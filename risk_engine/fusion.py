"""Multi-modal evidence fusion: edge CV + community (OSINT) reports + Sentinel-2 -> one risk matrix.

Design goal: every number in the output can be re-derived by hand from the stored evidence and the
constants in `config.FUSION_*`. There is no learned model here; that is deliberate. With no labelled
Karachi ground truth, a trained fuser could not be validated, so a transparent formula is the honest choice.

Pipeline
--------
1. COLLECT   evidence items from SQLite: CV detections (observations+detections), OSINT posts, and
             stored satellite checks. Only hazard labels in config.HAZARD_LABEL_MAP count (COCO "car" is not a hazard).
2. GROUP     items of the same hazard type within FUSION_MATCH_RADIUS_M of a cell's anchor form one hazard cell.
             The anchor is the earliest evidence, so a cell's id is stable across re-assessments.
3. SCORE per modality  (each in [0, cap_m]):
       item strength  s   = raw_confidence x reliability x geo_precision x 0.5^(age / half_life)
       one source     = best item from that independent source (same file / same author never counts twice)
       modality conf  c_m = min(cap_m, 1 - prod_over_sources(1 - s))          (noisy-OR)
   Satellite is one source: CONFIRMED_WET -> support x quality x applicability. NO_CHANGE / INCONCLUSIVE /
   NOT_APPLICABLE contribute 0 (they carry no *positive* evidence).
4. FUSE      evidence_confidence E = 1 - prod_m(1 - c_m)                          (noisy-OR across modalities)
             If clear satellite imagery of an observable hazard shows NO_CHANGE while other evidence is strong,
             E is reduced:  E *= 1 - lambda x quality x applicability   (flag `satellite_contradiction`).
5. IMPACT    weighted mean over the components that are actually KNOWN (severity, exposure, environment);
             unknown components are dropped and the weights renormalised, never silently guessed.
6. MATRIX    likelihood band L = band(E), impact band I = band(impact), both 1..5 (0.2-wide bins);
             category from L x I against config.FUSION_MATRIX_CATEGORIES; risk_score = 100 x E x impact.
7. RECORD    append a hash-chained entry to `evidence_ledger`, then an immutable row to `risk_matrix`
             that references it. See database/db.py for exactly what tamper-evidence does and doesn't give.

Known simplifications (stated, not hidden)
------------------------------------------
* Noisy-OR assumes modalities fail independently. Community reports and CV could both be wrong about
  the same viral photo; the per-author dedup, near-duplicate suppression and modality caps limit, but do not
  remove, that risk.
* All weights, caps, half-lives and thresholds are expert-set defaults, not calibrated on labelled outcomes.
* `E` is an evidence-strength index, NOT a calibrated probability that the hazard exists.
* Risk scores are experimental and not an official safety rating.
"""
from __future__ import annotations

import datetime as dt
import json
import math

import config
from database import db as kpdb
from geospatial import satellite as sat
from geospatial.clustering import haversine_m, valid_coord

UTC = dt.timezone.utc
MODALITIES = ("cv", "osint", "satellite")


# ====================================================================== small helpers
def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def parse_ts(value, default=None):
    """ISO-8601 -> aware UTC datetime. Naive strings are assumed to be UTC. None/garbage -> default."""
    if value is None:
        return default
    if isinstance(value, dt.datetime):
        d = value
    else:
        try:
            d = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return default
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)


def decay(age_hours, half_life_hours):
    """Exponential half-life decay in (0, 1]. Future-dated evidence (negative age) is not boosted."""
    return 0.5 ** (max(0.0, age_hours) / half_life_hours)


def band(x):
    """Maps [0,1] to a 1..5 band using 0.2-wide bins (1.0 falls in band 5)."""
    return 5 if x >= 1.0 else int(_clip01(x) * 5) + 1


def matrix_category(likelihood_band, impact_band):
    p = likelihood_band * impact_band
    return next((name for threshold, name in config.FUSION_MATRIX_CATEGORIES if p >= threshold), "Low")


def noisy_or(values):
    p = 1.0
    for v in values:
        p *= 1.0 - _clip01(v)
    return 1.0 - p


def geo_precision_factor(radius_m):
    r = 2000.0 if radius_m is None else float(radius_m)
    return next(f for limit, f in config.FUSION_GEO_PRECISION if r <= limit)


def config_fingerprint():
    """Short digest of every constant that influences a score; stored with each assessment so a
    score can be tied to the exact configuration that produced it."""
    keys = sorted(k for k in dir(config) if k.startswith("FUSION_")) + [
        "RISK_WEIGHTS", "BASE_SEVERITY", "RAIN_SENSITIVE", "SAT_APPLICABILITY", "HAZARD_LABEL_MAP"]
    return kpdb.sha256_hex(kpdb.canonical_json({k: getattr(config, k) for k in keys}))[:16]


# ====================================================================== evidence collection
def collect_cv(conn, since_iso=None):
    """CV evidence: one item per (observation, hazard label), strongest detection of that label in the observation."""
    q = ("SELECT o.id, o.source_id, o.timestamp, o.lat, o.lon, o.model, o.model_version, o.is_synthetic, "
         "d.id, d.label, d.confidence FROM observations o JOIN detections d ON d.observation_id=o.id "
         "WHERE o.lat IS NOT NULL AND o.lon IS NOT NULL")
    args = []
    if since_iso:
        q += " AND o.timestamp >= ?"
        args.append(since_iso)
    best = {}
    with kpdb.db_lock:
        rows = conn.execute(q, args).fetchall()
    for oid, src, ts, lat, lon, model, mver, synth, did, label, conf in rows:
        hazard = config.HAZARD_LABEL_MAP.get(str(label).lower())
        if hazard is None or not valid_coord(lat, lon, config.KARACHI_BBOX):
            continue
        k = (oid, hazard)
        if k not in best or (conf or 0) > best[k]["raw"]:
            best[k] = {"modality": "cv", "hazard": hazard, "lat": lat, "lon": lon, "ts": parse_ts(ts),
                       "raw": float(conf or 0.0), "source_key": f"cv:{src}", "geo_radius_m": 25.0,
                       "synthetic": bool(synth), "model": f"{model}:{mver}", "ref": f"cv:{oid}:{did}",
                       "label": label}
    return list(best.values())


def collect_osint(conn, since_iso=None):
    posts = kpdb.fetch_osint_posts(conn, since=since_iso, include_duplicates=False)
    items = []
    for r in posts.to_dict("records"):
        if r.get("status") not in (None, "active") or not r.get("hazard_type"):
            continue
        lat, lon = r.get("lat"), r.get("lon")
        if lat is None or lon is None or (isinstance(lat, float) and math.isnan(lat)):
            continue
        if not valid_coord(lat, lon, config.KARACHI_BBOX):
            continue
        author = r.get("author_hash")
        items.append({"modality": "osint", "hazard": r["hazard_type"], "lat": float(lat), "lon": float(lon),
                      "ts": parse_ts(r.get("posted_at")) or parse_ts(r.get("scraped_at")),
                      "raw": float(r.get("community_signal") or 0.0),
                      "source_key": f"osint:{author or 'anonymous'}",
                      "geo_radius_m": r.get("geo_radius_m"), "synthetic": bool(r.get("is_synthetic")),
                      "urgency": float(r.get("urgency") or 0.0), "platform": r.get("platform"),
                      "ref": f"osint:{r['id']}:{str(r.get('content_sha256'))[:12]}"})
    return items


# ====================================================================== grouping
def group_into_cells(items):
    """Anchor-based grouping (no chaining): sorted by time so the earliest item is the anchor."""
    cells = []
    for it in sorted(items, key=lambda i: (i["ts"] or dt.datetime.min.replace(tzinfo=UTC), i["ref"])):
        reach = max(config.FUSION_MATCH_RADIUS_M, min(it.get("geo_radius_m") or 0.0, 2000.0))
        for c in cells:
            if c["hazard"] == it["hazard"] and haversine_m((c["anchor_lat"], c["anchor_lon"]),
                                                          (it["lat"], it["lon"])) <= reach:
                c["items"].append(it)
                break
        else:
            cells.append({"hazard": it["hazard"], "anchor_lat": it["lat"], "anchor_lon": it["lon"], "items": [it]})
    for c in cells:
        c["cell_id"] = f"{c['hazard']}|{c['anchor_lat']:.4f}|{c['anchor_lon']:.4f}"
        c["lat"] = sum(i["lat"] for i in c["items"]) / len(c["items"])
        c["lon"] = sum(i["lon"] for i in c["items"]) / len(c["items"])
    return cells


# ====================================================================== scoring one cell
def _modality_confidence(items, modality, now, cv_reliability_by_model=None):
    """(confidence, detail dict) for CV or OSINT items of one cell."""
    best_per_source, detail = {}, {"n_items": len(items), "sources": {}}
    for it in items:
        if modality == "cv":
            reliability = (cv_reliability_by_model or {}).get(it.get("model"), config.FUSION_CV_UNVALIDATED_RELIABILITY)
            geo = 1.0
        else:
            reliability = config.FUSION_OSINT_SOURCE_RELIABILITY.get(it.get("platform"),
                                                                    config.FUSION_OSINT_SOURCE_RELIABILITY["default"])
            geo = geo_precision_factor(it.get("geo_radius_m"))
        age_h = ((now - it["ts"]).total_seconds() / 3600.0) if it["ts"] else config.FUSION_MAX_AGE_DAYS * 24.0
        s = _clip01(it["raw"]) * reliability * geo * decay(age_h, config.FUSION_HALF_LIFE_H[modality])
        k = it["source_key"]
        if s > best_per_source.get(k, -1.0):
            best_per_source[k] = s
    ranked = sorted(best_per_source.values(), reverse=True)
    if modality == "osint":
        ranked = ranked[:config.FUSION_OSINT_MAX_AUTHORS]
    conf = min(config.FUSION_MODALITY_CAP[modality], noisy_or(ranked))
    detail["sources"] = {k: round(v, 4) for k, v in sorted(best_per_source.items(), key=lambda kv: -kv[1])}
    detail["n_independent_sources"] = len(best_per_source)
    return conf, detail


def _satellite_confidence(ev, now):
    """(confidence, quality, detail). Only CONFIRMED_WET adds positive evidence."""
    if ev is None:
        return 0.0, 0.0, {"state": "not_checked"}
    detail = {"state": ev.verdict, "support": ev.support, "quality": ev.quality,
              "applicability": ev.applicability, "scene_id": ev.scene_id, "check_id": ev.check_id}
    if ev.verdict != sat.CONFIRMED_WET:
        return 0.0, ev.quality, detail
    t = parse_ts(ev.scene_time) or now
    age_h = (now - t).total_seconds() / 3600.0
    c = _clip01(ev.support * ev.quality * ev.applicability) * decay(age_h, config.FUSION_HALF_LIFE_H["satellite"])
    return min(config.FUSION_MODALITY_CAP["satellite"], c), ev.quality, detail


def score_cell(cell, satellite_ev=None, *, now=None, rainfall_mm=None, cv_reliability_by_model=None):
    """Pure function: one grouped cell (+ optional SatelliteEvidence) -> assessment dict (no DB access)."""
    now = now or dt.datetime.now(UTC)
    by_mod = {m: [i for i in cell["items"] if i["modality"] == m] for m in ("cv", "osint")}
    detail, strength, flags, expl = {}, {}, [], []

    for m in ("cv", "osint"):
        if by_mod[m]:
            strength[m], detail[m] = _modality_confidence(by_mod[m], m, now, cv_reliability_by_model)
        else:
            strength[m], detail[m] = 0.0, {"n_items": 0, "n_independent_sources": 0, "sources": {}}
    strength["satellite"], sat_quality, detail["satellite"] = _satellite_confidence(satellite_ev, now)

    # ---- fuse
    e_raw = noisy_or(strength.values())
    marginal = {}
    for m in MODALITIES:  # leave-one-out: how much does this modality add on top of the others?
        marginal[m] = e_raw - noisy_or(v for k, v in strength.items() if k != m)

    e = e_raw
    applicability = float(config.SAT_APPLICABILITY.get(cell["hazard"], 0.0))
    others = noisy_or([strength["cv"], strength["osint"]])
    if (satellite_ev is not None and satellite_ev.verdict == sat.NO_CHANGE
            and satellite_ev.quality >= config.FUSION_CONTRADICTION_MIN_QUALITY
            and others >= config.FUSION_CONTRADICTION_MIN_OTHER_EVIDENCE and applicability > 0):
        factor = 1.0 - config.FUSION_CONTRADICTION_LAMBDA * satellite_ev.quality * applicability
        e *= factor
        flags.append("satellite_contradiction")
        expl.append(f"Clear Sentinel-2 imagery shows no new water; evidence reduced x{factor:.2f} "
                    "(a 10 m pixel cannot see narrow street flooding, so this is a reduction, not a veto).")
    e = _clip01(e)

    # ---- impact (only known components)
    rain_component = (None if rainfall_mm is None else
                      _clip01(rainfall_mm / 100.0) * config.RAIN_SENSITIVE.get(cell["hazard"], 0.2))
    urgency = max([i.get("urgency", 0.0) for i in by_mod["osint"]] or [0.0])
    severity = _clip01(config.BASE_SEVERITY.get(cell["hazard"], 0.5)
                       + config.FUSION_URGENCY_SEVERITY_BOOST * urgency)
    components = {"severity": severity, "exposure": None, "environment": rain_component}
    known = {k: v for k, v in components.items() if v is not None and k in config.FUSION_IMPACT_COMPONENTS}
    wsum = sum(config.RISK_WEIGHTS[k] for k in known)
    impact = sum(config.RISK_WEIGHTS[k] * v for k, v in known.items()) / wsum
    if components["exposure"] is None:
        flags.append("exposure_unknown")
    if rain_component is None:
        flags.append("rainfall_unknown")

    # ---- matrix
    lb, ib = band(e), band(impact)
    category = matrix_category(lb, ib)
    score = round(100.0 * e * impact, 1)

    # ---- corroboration + flags
    agreeing = [m for m in MODALITIES if strength[m] >= config.FUSION_AGREE_MIN_CONTRIB]
    n_sources = (detail["cv"]["n_independent_sources"] + detail["osint"]["n_independent_sources"]
                 + (1 if strength["satellite"] > 0 else 0))
    corroboration = {0: "none", 1: "single-modality", 2: "two-modality"}.get(len(agreeing), "three-modality")
    if len(agreeing) <= 1:
        flags.append("single_modality")
    sv = satellite_ev.verdict if satellite_ev else None
    if sv == sat.INCONCLUSIVE:
        flags.append("satellite_inconclusive")
    if sv == sat.NOT_APPLICABLE or (sv is None and applicability == 0):
        flags.append("satellite_not_applicable")
    if sv is None and applicability > 0:
        flags.append("satellite_not_checked")
    if any(i["synthetic"] for i in cell["items"]) or (satellite_ev and satellite_ev.is_synthetic):
        flags.append("synthetic_data_included")
    if by_mod["cv"] and not cv_reliability_by_model:
        flags.append("cv_model_unvalidated")
    newest = max((i["ts"] for i in cell["items"] if i["ts"]), default=None)
    if newest and (now - newest).total_seconds() > 7 * 86400:
        flags.append("stale_evidence")

    # ---- explanation (deterministic text from the numbers above)
    lines = []
    for m, label in (("cv", "Edge CV"), ("osint", "Community reports"), ("satellite", "Sentinel-2")):
        if m == "satellite":
            lines.append(f"{label}: {detail[m]['state']}, contributes {strength[m]:.2f}")
        else:
            lines.append(f"{label}: {detail[m]['n_independent_sources']} independent source(s), "
                         f"contributes {strength[m]:.2f}")
    expl = lines + expl
    expl.append(f"Evidence strength E={e:.2f} (noisy-OR of modalities); impact={impact:.2f} from "
                f"{', '.join(known)}; matrix L{lb} x I{ib} -> {category}; score = 100 x E x impact = {score}")

    digest_payload = {"cell": cell["cell_id"], "refs": sorted(i["ref"] for i in cell["items"]),
                      "sat": [satellite_ev.check_id, satellite_ev.result_sha256] if satellite_ev else None,
                      "rain": rainfall_mm, "cfg": config_fingerprint()}
    return {
        "cell_id": cell["cell_id"], "hazard_type": cell["hazard"], "lat": round(cell["lat"], 6),
        "lon": round(cell["lon"], 6), "assessed_at": now.isoformat(timespec="seconds"),
        "cv_strength": round(strength["cv"], 4), "osint_strength": round(strength["osint"], 4),
        "satellite_strength": round(strength["satellite"], 4),
        "cv_contrib": round(marginal["cv"], 4), "osint_contrib": round(marginal["osint"], 4),
        "satellite_contrib": round(marginal["satellite"], 4),
        "evidence_confidence": round(e, 4), "severity": round(severity, 4), "exposure": None,
        "impact": round(impact, 4), "likelihood_band": lb, "impact_band": ib,
        "matrix_cell": f"L{lb}xI{ib}", "matrix_category": category, "risk_score": score,
        "risk_level": category, "corroboration": corroboration, "flags": flags, "explanation": " | ".join(expl),
        "n_independent_sources": n_sources,
        "is_synthetic": any(i["synthetic"] for i in cell["items"]) or bool(satellite_ev and satellite_ev.is_synthetic),
        "inputs_digest": kpdb.sha256_hex(kpdb.canonical_json(digest_payload)),
        "algorithm_version": config.FUSION_ALGORITHM_VERSION, "config_fingerprint": config_fingerprint(),
        "detail": detail,
    }


# ====================================================================== orchestration
def assess_all(conn, *, now=None, rainfall_mm=None, persist=True, cv_reliability_by_model=None):
    """Collect -> group -> attach satellite evidence -> score -> (optionally) record. Returns assessments,
    highest risk first. Each carries `persisted` (False when nothing changed since the last record)."""
    now = now or dt.datetime.now(UTC)
    since = (now - dt.timedelta(days=config.FUSION_MAX_AGE_DAYS)).isoformat(timespec="seconds")
    cells = group_into_cells(collect_cv(conn, since) + collect_osint(conn, since))
    out = []
    for cell in cells:
        sat_ev = sat.find_evidence(conn, cell["lat"], cell["lon"], cell["hazard"], since_iso=since)
        a = score_cell(cell, sat_ev, now=now, rainfall_mm=rainfall_mm, cv_reliability_by_model=cv_reliability_by_model)
        a["persisted"] = _record(conn, a) if persist else False
        out.append(a)
    return sorted(out, key=lambda a: -a["risk_score"])


def _ledger_payload(a):
    keys = ("cell_id", "assessed_at", "hazard_type", "evidence_confidence", "impact", "matrix_cell",
            "matrix_category", "risk_score", "inputs_digest", "algorithm_version", "config_fingerprint")
    return {k: a[k] for k in keys}


def _record(conn, a):
    """Appends ledger entry + risk_matrix row unless the newest row for this cell is identical in inputs and result."""
    with kpdb.db_lock:
        last = conn.execute("SELECT inputs_digest, matrix_cell, ledger_seq, ledger_hash FROM risk_matrix "
                            "WHERE cell_id=? ORDER BY id DESC LIMIT 1", (a["cell_id"],)).fetchone()
        if last and last[0] == a["inputs_digest"] and last[1] == a["matrix_cell"]:
            a["ledger_seq"], a["ledger_hash"] = last[2], last[3]
            return False
        seq, h = kpdb.append_ledger(conn, "risk_assessment", _ledger_payload(a))
        a["ledger_seq"], a["ledger_hash"] = seq, h
        kpdb.insert_risk_matrix_row(conn, **{k: v for k, v in a.items() if k not in ("detail", "persisted")})
        return True


# ====================================================================== integrity checks
def verify_matrix(conn):
    """Checks every risk_matrix row against the ledger entry it points to. Returns
    {"ok", "n_rows", "problems": [...]}. Complements kpdb.verify_ledger() (chain integrity)."""
    problems, n = [], 0
    with kpdb.db_lock:
        cur = conn.execute("SELECT * FROM risk_matrix ORDER BY id")
        cols = [d[0] for d in cur.description]
        for r in (dict(zip(cols, row)) for row in cur.fetchall()):
            n += 1
            led = conn.execute("SELECT payload, entry_hash FROM evidence_ledger WHERE seq=?", (r["ledger_seq"],)).fetchone()
            if led is None:
                problems.append({"id": r["id"], "problem": "referenced ledger entry is missing"})
                continue
            if led[1] != r["ledger_hash"]:
                problems.append({"id": r["id"], "problem": "ledger hash differs from the one recorded on the row"})
            payload = json.loads(led[0])
            diffs = [k for k, v in payload.items() if r.get(k) != v]
            if diffs:
                problems.append({"id": r["id"], "problem": f"row disagrees with its ledger entry on {diffs}"})
    return {"ok": not problems, "n_rows": n, "problems": problems}


def verify_evidence(conn):
    """Re-hashes stored OSINT text and satellite results against the digests recorded at ingestion."""
    problems = []
    with kpdb.db_lock:
        for pid, text, h in conn.execute("SELECT id, text, content_sha256 FROM osint_posts"):
            if kpdb.sha256_hex(text or "") != h:
                problems.append({"table": "osint_posts", "id": pid, "problem": "text no longer matches content_sha256"})
        cur = conn.execute("SELECT * FROM satellite_checks")
        cols = [d[0] for d in cur.description]
        for row in (dict(zip(cols, r)) for r in cur.fetchall()):
            if not sat.row_digest_matches(row):
                problems.append({"table": "satellite_checks", "id": row["id"], "problem": "result no longer matches result_sha256"})
    return {"ok": not problems, "problems": problems}


def full_audit(conn):
    """One call for the UI: chain + matrix-vs-ledger + evidence digests."""
    chain, matrix, evidence = kpdb.verify_ledger(conn), verify_matrix(conn), verify_evidence(conn)
    return {"ok": chain["ok"] and matrix["ok"] and evidence["ok"], "chain": chain, "matrix": matrix,
            "evidence": evidence, "head_seq": chain["head_seq"], "head_hash": chain["head_hash"]}
