"""SQLite persistence layer.

Original tables (v1): locations, models, incidents, observations, detections, weather,
risk_scores, experiments.

Added in v2 (multi-modal fusion):
    osint_posts       scraped community reports (author handles hashed, never stored raw)
    satellite_checks  Sentinel-2 cross-verification results (NDWI/NDVI metrics + verdict)
    evidence_ledger   append-only, hash-chained audit log of every fused assessment
    risk_matrix       the fused hazard risk matrix (one row per assessment of a hazard cell)

Tamper-evidence model (be precise about what this does and does not give you):
    * `evidence_ledger` and `risk_matrix` reject UPDATE/DELETE through SQLite triggers, so a casual
      edit (a stray SQL statement, a buggy script) fails loudly.
    * Every ledger entry stores H(prev_hash | ts | kind | payload). Editing, deleting or reordering any
      entry breaks the chain, and `verify_ledger()` reports exactly where.
    * If the environment variable KP_LEDGER_KEY is set, H is HMAC-SHA256 under that key, so someone who
      can rewrite the database file still cannot recompute a valid chain without the key.
    * Without a key it is plain SHA-256: tamper-EVIDENT against accidents and naive edits, but a
      determined attacker who rewrites the whole file can rebuild the chain. Publishing the chain head
      (`ledger_head()`) somewhere external (a git commit, a tweet) anchors it.
    Nothing here makes a local SQLite file "un-tamperable"; it makes tampering detectable.
"""
import datetime
import hashlib
import hmac
import json
import os
import sqlite3
import threading

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations(id INTEGER PRIMARY KEY, lat REAL, lon REAL, area TEXT, road_segment TEXT);
CREATE TABLE IF NOT EXISTS models(id INTEGER PRIMARY KEY, name TEXT, version TEXT, scope TEXT);
CREATE TABLE IF NOT EXISTS incidents(id INTEGER PRIMARY KEY, code TEXT UNIQUE, type TEXT, lat REAL, lon REAL,
  first_seen TEXT, last_seen TEXT, n_observations INTEGER, evidence_confidence REAL, is_synthetic INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY, incident_id INTEGER REFERENCES incidents(id),
  source_id TEXT, timestamp TEXT, lat REAL, lon REAL, evidence_type TEXT, model TEXT, model_version TEXT,
  confidence REAL, is_synthetic INTEGER NOT NULL, processing_version TEXT);
CREATE TABLE IF NOT EXISTS detections(id INTEGER PRIMARY KEY, observation_id INTEGER REFERENCES observations(id),
  label TEXT, confidence REAL, bbox TEXT);
CREATE TABLE IF NOT EXISTS weather(id INTEGER PRIMARY KEY, timestamp TEXT, lat REAL, lon REAL, rain_mm REAL, source TEXT);
CREATE TABLE IF NOT EXISTS risk_scores(id INTEGER PRIMARY KEY, incident_id INTEGER REFERENCES incidents(id),
  score REAL, level TEXT, details TEXT, computed_at TEXT);
CREATE TABLE IF NOT EXISTS experiments(id INTEGER PRIMARY KEY, name TEXT, date TEXT, model TEXT, dataset_version TEXT, params TEXT, metrics TEXT, notes TEXT);
CREATE INDEX IF NOT EXISTS ix_inc_type ON incidents(type);
CREATE INDEX IF NOT EXISTS ix_inc_loc ON incidents(lat, lon);
CREATE INDEX IF NOT EXISTS ix_obs_inc ON observations(incident_id);
CREATE INDEX IF NOT EXISTS ix_obs_ts ON observations(timestamp);
CREATE INDEX IF NOT EXISTS ix_det_obs ON detections(observation_id);

-- ------------------------------------------------------------------ v2: OSINT evidence
CREATE TABLE IF NOT EXISTS osint_posts(
  id INTEGER PRIMARY KEY,
  platform TEXT NOT NULL,            -- e.g. 'mock', 'rss', a forum name
  post_id TEXT NOT NULL,             -- id on that platform (unique per platform)
  url TEXT,
  author_hash TEXT,                  -- salted SHA-256 prefix; the raw handle is never stored
  posted_at TEXT,                    -- ISO-8601 UTC, as reported by the source
  scraped_at TEXT NOT NULL,
  text TEXT NOT NULL,                -- privacy-redacted (emails/phones removed)
  lang TEXT,                         -- 'en' | 'ur' | 'roman-ur' | 'mixed'
  image_urls TEXT,                   -- JSON list; URLs only, images are not stored
  hazard_type TEXT,
  hazard_strength REAL, urgency REAL, negativity REAL, community_signal REAL,
  status TEXT,                       -- 'active' | 'resolved' | 'irrelevant'
  keywords TEXT,                     -- JSON list of matched lexicon terms (explainability)
  lat REAL, lon REAL, geo_method TEXT, geo_radius_m REAL, geo_place TEXT,
  content_sha256 TEXT NOT NULL,      -- digest of `text`; lets the ledger detect later edits
  duplicate_of INTEGER,              -- id of the earlier near-identical post (counts once in fusion)
  is_synthetic INTEGER NOT NULL,
  UNIQUE(platform, post_id)
);
CREATE INDEX IF NOT EXISTS ix_osint_hazard ON osint_posts(hazard_type, posted_at);
CREATE INDEX IF NOT EXISTS ix_osint_loc ON osint_posts(lat, lon);

-- ------------------------------------------------------------------ v2: satellite evidence
CREATE TABLE IF NOT EXISTS satellite_checks(
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  trigger_post_id INTEGER,           -- osint_posts.id that triggered the check (nullable)
  hazard_type TEXT, lat REAL, lon REAL, aoi_radius_m REAL, event_time TEXT,
  cell_key TEXT,                     -- hazard|grid-cell, used as the cache key
  verdict TEXT,                      -- CONFIRMED_WET | NO_CHANGE | INCONCLUSIVE | NOT_APPLICABLE
  support REAL, quality REAL, applicability REAL,
  scene_time TEXT, scene_id TEXT, baseline_scene_ids TEXT, cloud_pct REAL, valid_fraction REAL,
  metrics_json TEXT NOT NULL,        -- full NDWI/NDVI/MNDWI numbers behind the verdict
  provider TEXT, is_synthetic INTEGER NOT NULL,
  result_sha256 TEXT NOT NULL,       -- digest of metrics_json + verdict (tamper-evidence)
  error TEXT
);
CREATE INDEX IF NOT EXISTS ix_sat_cell ON satellite_checks(cell_key, created_at);
CREATE INDEX IF NOT EXISTS ix_sat_loc ON satellite_checks(lat, lon);

-- ------------------------------------------------------------------ v2: ledger + risk matrix
CREATE TABLE IF NOT EXISTS evidence_ledger(
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload TEXT NOT NULL,             -- canonical JSON
  prev_hash TEXT NOT NULL,
  alg TEXT NOT NULL,                 -- 'sha256' | 'hmac-sha256'
  entry_hash TEXT NOT NULL UNIQUE
);
CREATE TABLE IF NOT EXISTS risk_matrix(
  id INTEGER PRIMARY KEY,
  cell_id TEXT NOT NULL,             -- hazard|anchor lat|anchor lon (stable across re-assessments)
  hazard_type TEXT, lat REAL, lon REAL, assessed_at TEXT,
  cv_strength REAL, osint_strength REAL, satellite_strength REAL,
  cv_contrib REAL, osint_contrib REAL, satellite_contrib REAL,
  evidence_confidence REAL, severity REAL, exposure REAL, impact REAL,
  likelihood_band INTEGER, impact_band INTEGER, matrix_cell TEXT, matrix_category TEXT,
  risk_score REAL, risk_level TEXT, corroboration TEXT, flags TEXT, explanation TEXT,
  n_independent_sources INTEGER, is_synthetic INTEGER,
  inputs_digest TEXT, ledger_seq INTEGER, ledger_hash TEXT,
  algorithm_version TEXT, config_fingerprint TEXT
);
CREATE INDEX IF NOT EXISTS ix_rm_cell ON risk_matrix(cell_id, id);
"""

# Append-only guards. Kept separate so reset_audit_tables() can drop/recreate them.
TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS trg_ledger_no_update BEFORE UPDATE ON evidence_ledger
BEGIN SELECT RAISE(ABORT, 'evidence_ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_ledger_no_delete BEFORE DELETE ON evidence_ledger
BEGIN SELECT RAISE(ABORT, 'evidence_ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_rm_no_update BEFORE UPDATE ON risk_matrix
BEGIN SELECT RAISE(ABORT, 'risk_matrix is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_rm_no_delete BEFORE DELETE ON risk_matrix
BEGIN SELECT RAISE(ABORT, 'risk_matrix is append-only'); END;
CREATE TRIGGER IF NOT EXISTS trg_osint_no_update BEFORE UPDATE ON osint_posts
BEGIN SELECT RAISE(ABORT, 'osint_posts rows are immutable evidence'); END;
CREATE TRIGGER IF NOT EXISTS trg_sat_no_update BEFORE UPDATE ON satellite_checks
BEGIN SELECT RAISE(ABORT, 'satellite_checks rows are immutable evidence'); END;
"""

_TRIGGER_NAMES = ("trg_ledger_no_update", "trg_ledger_no_delete", "trg_rm_no_update", "trg_rm_no_delete",
                  "trg_osint_no_update", "trg_sat_no_update")

# The Streamlit app shares one cached connection (check_same_thread=False) between the UI thread and
# the thread that runs the asyncio OSINT pipeline, so every v2 write goes through this lock.
_WRITE_LOCK = threading.RLock()
db_lock = _WRITE_LOCK          # public alias: wrap multi-statement read/modify sequences in `with db_lock:`
GENESIS_HASH = "0" * 64


def connect(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    c = sqlite3.connect(path, check_same_thread=False, timeout=10)
    try:
        c.execute("PRAGMA journal_mode=WAL")      # readers (UI) don't block the pipeline writer
        c.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        pass
    c.execute("PRAGMA busy_timeout=5000")
    c.executescript(SCHEMA)
    c.executescript(TRIGGERS)
    return c


# =============================================================== v1 helpers (unchanged API)
def insert_observation(conn, source_id, lat, lon, evidence_type, model, model_version,
                        confidence, is_synthetic, processing_version):
    """Persists one real (or synthetic, if explicitly flagged) observation. Returns its row id."""
    with _WRITE_LOCK:
        cur = conn.execute(
            "INSERT INTO observations(source_id,timestamp,lat,lon,evidence_type,model,model_version,"
            "confidence,is_synthetic,processing_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (source_id, utc_now_iso(), lat, lon, evidence_type, model,
             model_version, confidence, int(is_synthetic), processing_version))
        conn.commit()
        return cur.lastrowid


def insert_detection(conn, observation_id, label, confidence, bbox):
    with _WRITE_LOCK:
        cur = conn.execute("INSERT INTO detections(observation_id,label,confidence,bbox) VALUES (?,?,?,?)",
                           (observation_id, label, confidence, json.dumps(bbox)))
        conn.commit()
        return cur.lastrowid


def fetch_observations(conn):
    import pandas as pd
    return pd.read_sql_query("SELECT * FROM observations ORDER BY timestamp DESC", conn)


def fetch_detections(conn):
    import pandas as pd
    return pd.read_sql_query("SELECT * FROM detections ORDER BY id DESC", conn)


# =============================================================== small utilities
def utc_now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def canonical_json(obj):
    """Deterministic JSON (sorted keys, no whitespace, ASCII-only) used for every hashed payload."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_hex(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ledger_key():
    k = os.getenv(config.LEDGER_KEY_ENV, "")
    return k.encode("utf-8") if k else None


def _entry_hash(alg, prev_hash, seq, ts, kind, payload_json):
    msg = f"{prev_hash}|{seq}|{ts}|{kind}|{payload_json}".encode("utf-8")
    if alg == "hmac-sha256":
        key = _ledger_key()
        if key is None:
            raise KeyError("ledger entry was written with HMAC but KP_LEDGER_KEY is not set")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()
    return hashlib.sha256(msg).hexdigest()


# =============================================================== OSINT posts
_OSINT_COLS = ("platform", "post_id", "url", "author_hash", "posted_at", "scraped_at", "text", "lang",
               "image_urls", "hazard_type", "hazard_strength", "urgency", "negativity", "community_signal",
               "status", "keywords", "lat", "lon", "geo_method", "geo_radius_m", "geo_place",
               "content_sha256", "duplicate_of", "is_synthetic")


def insert_osint_post(conn, **fields):
    """Inserts one processed post. Returns the new row id, or None if (platform, post_id) already exists
    (re-scraping the same page must be idempotent)."""
    row = {c: fields.get(c) for c in _OSINT_COLS}
    for jf in ("image_urls", "keywords"):
        if isinstance(row[jf], (list, tuple)):
            row[jf] = json.dumps(list(row[jf]), ensure_ascii=False)
    row["scraped_at"] = row["scraped_at"] or utc_now_iso()
    row["is_synthetic"] = int(bool(row["is_synthetic"]))
    row["content_sha256"] = row["content_sha256"] or sha256_hex(row["text"] or "")
    with _WRITE_LOCK:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO osint_posts({','.join(_OSINT_COLS)}) "
            f"VALUES ({','.join('?' * len(_OSINT_COLS))})", [row[c] for c in _OSINT_COLS])
        conn.commit()
        return cur.lastrowid if cur.rowcount else None


def fetch_osint_posts(conn, since=None, hazard_type=None, include_duplicates=True, limit=None):
    import pandas as pd
    q, args = "SELECT * FROM osint_posts WHERE 1=1", []
    if since:
        q += " AND COALESCE(posted_at, scraped_at) >= ?"; args.append(since)
    if hazard_type:
        q += " AND hazard_type = ?"; args.append(hazard_type)
    if not include_duplicates:
        q += " AND duplicate_of IS NULL"
    q += " ORDER BY id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    return pd.read_sql_query(q, conn, params=args)


# =============================================================== satellite checks
_SAT_COLS = ("created_at", "trigger_post_id", "hazard_type", "lat", "lon", "aoi_radius_m", "event_time",
             "cell_key", "verdict", "support", "quality", "applicability", "scene_time", "scene_id",
             "baseline_scene_ids", "cloud_pct", "valid_fraction", "metrics_json", "provider",
             "is_synthetic", "result_sha256", "error")


def insert_satellite_check(conn, **fields):
    row = {c: fields.get(c) for c in _SAT_COLS}
    row["created_at"] = row["created_at"] or utc_now_iso()
    row["is_synthetic"] = int(bool(row["is_synthetic"]))
    if isinstance(row["baseline_scene_ids"], (list, tuple)):
        row["baseline_scene_ids"] = json.dumps(list(row["baseline_scene_ids"]))
    if not isinstance(row["metrics_json"], str):
        row["metrics_json"] = canonical_json(row["metrics_json"] or {})
    with _WRITE_LOCK:
        cur = conn.execute(f"INSERT INTO satellite_checks({','.join(_SAT_COLS)}) "
                           f"VALUES ({','.join('?' * len(_SAT_COLS))})", [row[c] for c in _SAT_COLS])
        conn.commit()
        return cur.lastrowid


def fetch_satellite_checks(conn, limit=None):
    import pandas as pd
    q = "SELECT * FROM satellite_checks ORDER BY id DESC" + (f" LIMIT {int(limit)}" if limit else "")
    return pd.read_sql_query(q, conn)


def latest_satellite_check(conn, cell_key, max_age_hours):
    """Most recent check for this cell newer than max_age_hours, as a dict, or None."""
    cutoff = (datetime.datetime.now(datetime.timezone.utc)
              - datetime.timedelta(hours=max_age_hours)).isoformat(timespec="seconds")
    cur = conn.execute("SELECT * FROM satellite_checks WHERE cell_key=? AND created_at>=? "
                       "ORDER BY id DESC LIMIT 1", (cell_key, cutoff))
    r = cur.fetchone()
    if r is None:
        return None
    return dict(zip([d[0] for d in cur.description], r))


# =============================================================== ledger (hash chain)
def append_ledger(conn, kind, payload):
    """Appends one entry to the hash chain and returns (seq, entry_hash). Thread-safe."""
    payload_json = payload if isinstance(payload, str) else canonical_json(payload)
    ts = utc_now_iso()
    alg = "hmac-sha256" if _ledger_key() else "sha256"
    with _WRITE_LOCK:
        if not conn.in_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            last = conn.execute("SELECT seq, entry_hash FROM evidence_ledger ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (last[0] + 1, last[1]) if last else (1, GENESIS_HASH)
            h = _entry_hash(alg, prev, seq, ts, kind, payload_json)
            conn.execute("INSERT INTO evidence_ledger(seq,ts,kind,payload,prev_hash,alg,entry_hash) "
                         "VALUES (?,?,?,?,?,?,?)", (seq, ts, kind, payload_json, prev, alg, h))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return seq, h


def ledger_head(conn):
    """(seq, hash) of the newest entry, or (0, GENESIS_HASH) for an empty chain. Publish this
    externally to anchor the chain."""
    r = conn.execute("SELECT seq, entry_hash FROM evidence_ledger ORDER BY seq DESC LIMIT 1").fetchone()
    return (r[0], r[1]) if r else (0, GENESIS_HASH)


def fetch_ledger(conn, limit=None):
    import pandas as pd
    q = "SELECT * FROM evidence_ledger ORDER BY seq DESC" + (f" LIMIT {int(limit)}" if limit else "")
    return pd.read_sql_query(q, conn)


def verify_ledger(conn):
    """Re-walks the whole chain. Returns
        {"ok": bool, "n_entries": int, "head_seq": int, "head_hash": str, "problems": [...], "keyed": bool}
    A problem is {"seq": int, "problem": str}. Detects edited payloads/timestamps, deleted entries
    (sequence gaps), reordered entries, and re-linked chains."""
    problems, prev, expected_seq, n = [], GENESIS_HASH, 1, 0
    for seq, ts, kind, payload, prev_hash, alg, entry_hash in conn.execute(
            "SELECT seq,ts,kind,payload,prev_hash,alg,entry_hash FROM evidence_ledger ORDER BY seq"):
        n += 1
        if seq != expected_seq:
            problems.append({"seq": seq, "problem": f"sequence gap: expected {expected_seq} (entry deleted or reordered?)"})
            expected_seq = seq
        if prev_hash != prev:
            problems.append({"seq": seq, "problem": "prev_hash does not match the previous entry (chain broken)"})
        try:
            recomputed = _entry_hash(alg, prev_hash, seq, ts, kind, payload)
        except KeyError as e:
            problems.append({"seq": seq, "problem": f"unverifiable: {e}"})
            recomputed = entry_hash
        if recomputed != entry_hash:
            problems.append({"seq": seq, "problem": "entry hash mismatch (payload or timestamp was modified)"})
        prev, expected_seq = entry_hash, expected_seq + 1
    head_seq, head_hash = ledger_head(conn)
    return {"ok": not problems, "n_entries": n, "head_seq": head_seq, "head_hash": head_hash,
            "problems": problems, "keyed": _ledger_key() is not None}


# =============================================================== risk matrix
_RM_COLS = ("cell_id", "hazard_type", "lat", "lon", "assessed_at", "cv_strength", "osint_strength",
            "satellite_strength", "cv_contrib", "osint_contrib", "satellite_contrib", "evidence_confidence",
            "severity", "exposure", "impact", "likelihood_band", "impact_band", "matrix_cell",
            "matrix_category", "risk_score", "risk_level", "corroboration", "flags", "explanation",
            "n_independent_sources", "is_synthetic", "inputs_digest", "ledger_seq", "ledger_hash",
            "algorithm_version", "config_fingerprint")


def insert_risk_matrix_row(conn, **fields):
    row = {c: fields.get(c) for c in _RM_COLS}
    if isinstance(row["flags"], (list, tuple)):
        row["flags"] = json.dumps(list(row["flags"]))
    row["is_synthetic"] = int(bool(row["is_synthetic"]))
    with _WRITE_LOCK:
        cur = conn.execute(f"INSERT INTO risk_matrix({','.join(_RM_COLS)}) "
                           f"VALUES ({','.join('?' * len(_RM_COLS))})", [row[c] for c in _RM_COLS])
        conn.commit()
        return cur.lastrowid


def fetch_risk_matrix(conn, latest_only=True):
    """The fused risk matrix. latest_only=True returns the newest assessment of each hazard cell."""
    import pandas as pd
    if latest_only:
        q = ("SELECT * FROM risk_matrix WHERE id IN (SELECT MAX(id) FROM risk_matrix GROUP BY cell_id) "
             "ORDER BY risk_score DESC")
    else:
        q = "SELECT * FROM risk_matrix ORDER BY id DESC"
    return pd.read_sql_query(q, conn)


def reset_audit_tables(conn):
    """DEMO/DEV ONLY. Wipes OSINT posts, satellite checks, the ledger and the risk matrix, including
    the audit trail. Drops and recreates the append-only triggers; never call this on real data."""
    with _WRITE_LOCK:
        conn.executescript("".join(f"DROP TRIGGER IF EXISTS {t};" for t in _TRIGGER_NAMES))
        conn.executescript("DELETE FROM risk_matrix; DELETE FROM evidence_ledger; "
                           "DELETE FROM satellite_checks; DELETE FROM osint_posts;")
        try:
            conn.execute("DELETE FROM sqlite_sequence WHERE name IN ('evidence_ledger')")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.executescript(TRIGGERS)
