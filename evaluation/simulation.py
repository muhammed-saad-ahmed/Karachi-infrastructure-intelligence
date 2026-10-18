"""Monte Carlo sensitivity engine + Bayesian (MCMC) calibration bounds for the evidence-fusion risk matrix.

WHY THIS EXISTS
    risk_engine/fusion.py is a transparent formula, but its constants (modality caps, half-lives, reliabilities, the
    contradiction strength, the impact weights, ...) are expert-set defaults in config/__init__.py. Those constants
    decide which cell of the 5x5 matrix a hazard lands in, and that cell is written into the hash-chained ledger.
    This module answers, with numbers, the question "how much does the output depend on guesses I cannot verify?"

WHAT IT DOES
    1. SENSITIVITY (Monte Carlo, default 10,000 iterations)
       Every uncertain constant is perturbed with Gaussian noise (probabilities in logit space, scales in log space,
       so they always stay valid) and the WHOLE fusion + matrix is re-evaluated for every draw, for every hazard
       cell, in vectorised NumPy (no Python loop over draws). Reported per cell:
         * 95% intervals for the evidence score E, the impact and the risk score
         * the probability of each matrix category and of keeping the baseline category (= stability)
         * the structural margin: how far E or impact must move before the category changes
       and overall: the largest relative perturbation the matrix tolerates while >= 95% of cells keep their category.
    2. LEAVE-ONE-OUT matrix
       The same draws are re-used (common random numbers) with one modality (CV / OSINT / satellite) removed. This
       shows each stream's marginal contribution WITH an interval and how much category stability is lost
       without it, under moderate and severe uncertainty.
    3. BAYESIAN CALIBRATION (random-walk Metropolis, 4 chains, R-hat + ESS diagnostics)
       Given LABELLED hazard cells (confirmed / refuted on the ground) it infers how reliable CV and OSINT evidence
       really are, with 95% credible intervals, and those posterior draws can drive the sensitivity run.
       Assumption stated up front: for calibration, the evidence score E is treated as the probability that the
       hazard is real (Bernoulli likelihood). E is built as a noisy-OR, which has that shape, but it is an index.

HONESTY
    * The sensitivity analysis measures robustness to the PERTURBATION MODEL below. It is not a calibration: a
      stable result can still be wrong if the formula's structure is wrong.
    * No labelled Karachi outcomes exist in this repository, so calibrate_reliabilities() has only been validated
      on synthetic labels generated from a known truth (see tests). Until real labels are logged, treat the
      calibration output as a tool, not as a finding.
    * Perturbation sizes (`sigma`) are a judgement call; the run reports results at a moderate and a severe level.

Entry points:  load_cells / compile_cells -> run_sensitivity -> report_tables / markdown_report / anchor_report
CLI:           python -m evaluation.simulation --synthetic 60 --n-iter 10000 --out reports/sensitivity.json
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import os

import numpy as np

import config
from database import db as kpdb
from geospatial import satellite as sat
from risk_engine import fusion

UTC = dt.timezone.utc
MODALITIES = fusion.MODALITIES                         # ("cv", "osint", "satellite")
_CAT_ORDER = ["Low", "Moderate", "High", "Critical"]  # index = severity rank used throughout


# ====================================================================== small numeric helpers
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _band(x):
    """Vectorised twin of fusion.band(): 1..5 in 0.2-wide bins, 1.0 -> 5."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x >= 1.0, 5, (x * 5).astype(np.int64) + 1)


def _category_index(likelihood_band, impact_band):
    """Vectorised twin of fusion.matrix_category(): 0 Low .. 3 Critical."""
    p = likelihood_band * impact_band
    conds = [p >= thr for thr, _ in config.FUSION_MATRIX_CATEGORIES]
    ranks = [_CAT_ORDER.index(name) for _, name in config.FUSION_MATRIX_CATEGORIES]
    return np.select(conds, ranks, default=0)


def _noisy_or_arr(*arrays):
    out = np.ones_like(arrays[0], dtype=np.float64)
    for a in arrays:
        out = out * (1.0 - np.clip(a, 0.0, 1.0))
    return 1.0 - out


# ====================================================================== settings
@dataclasses.dataclass
class SimulationSettings:
    n_iter: int = 10_000          # Monte Carlo draws at the moderate level
    sigma: float = 0.15           # relative perturbation scale (sd of a parameter as a share of its value)
    severe_sigma: float = 0.50    # "severe data uncertainty" level used for the leave-one-out ranking
    seed: int = 20261017
    ci: float = 0.95
    stability_target: float = 0.95
    ladder: tuple = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00)
    n_ladder: int = 2_000         # draws per ladder step (the ladder only needs the overall stability)

    def validate(self):
        if self.n_iter < 1 or self.n_ladder < 1:
            raise ValueError("n_iter and n_ladder must be >= 1")
        if not (0 < self.sigma <= 5 and 0 < self.severe_sigma <= 5):
            raise ValueError("sigma values must be in (0, 5]")
        if not 0.5 < self.ci < 1.0:
            raise ValueError("ci must be in (0.5, 1)")


# ====================================================================== compile cells -> padded arrays
@dataclasses.dataclass
class CompiledCells:
    """Everything fusion needs, as dense arrays of shape (C cells, S sources, K items per source)."""
    cell_ids: list
    hazards: list
    hazard_idx: np.ndarray            # (C,) index into `hazard_names`
    hazard_names: list
    cv: dict                          # raw, age, rel0, geo_idx : (C, S, K)
    osint: dict
    sat: dict                         # confirmed(C,), nochange(C,), support, quality, age, applic_item, applic_cfg
    urgency: np.ndarray               # (C,)
    rain: np.ndarray                  # (C,) environment component, NaN = unknown
    base_sev: np.ndarray              # (C,) baseline severity before urgency boost
    mean_rel: dict                    # typical baseline reliability per modality (for perturbation scaling)
    has_synthetic: bool = False
    baseline_fusion: list = dataclasses.field(default_factory=list)   # fusion.score_cell outputs (cross-check)

    @property
    def n_cells(self):
        return len(self.cell_ids)


def _pad(groups, fill):
    """groups: list (per cell) of list (per source) of list of tuples -> padded arrays."""
    n_c = len(groups)
    s_max = max([len(g) for g in groups] + [1])
    k_max = max([len(src) for g in groups for src in g] + [1])
    out = {k: np.full((n_c, s_max, k_max), v, dtype=np.float64) for k, v in fill.items()}
    for ci, g in enumerate(groups):
        for si, src in enumerate(g):
            for ki, item in enumerate(src):
                for key in fill:
                    out[key][ci, si, ki] = item[key]
    out["geo_idx"] = out["geo_idx"].astype(np.int64)
    return out


def _geo_bin(radius_m):
    r = 2000.0 if radius_m is None else float(radius_m)
    return next(i for i, (limit, _) in enumerate(config.FUSION_GEO_PRECISION) if r <= limit)


def compile_cells(cells, sat_evs=None, *, now=None, rainfall_mm=None, cv_reliability_by_model=None,
                  cross_check=True):
    """Turns grouped cells (fusion.group_into_cells output) + their satellite evidence into dense arrays.
    With cross_check=True the unperturbed fusion.score_cell() result for every cell is kept, so callers (and the
    tests) can verify that this vectorised engine reproduces the real fusion code exactly."""
    now = now or dt.datetime.now(UTC)
    sat_evs = sat_evs if sat_evs is not None else [None] * len(cells)
    if len(sat_evs) != len(cells):
        raise ValueError("sat_evs must be parallel to cells")
    hazard_names = sorted({c["hazard"] for c in cells})
    cv_groups, os_groups = [], []
    urg, rain, base_sev, h_idx = [], [], [], []
    cv_rels, os_rels = [], []
    sat_rows = {k: [] for k in ("confirmed", "nochange", "support", "quality", "age", "applic_item", "applic_cfg")}
    baseline_fusion = []
    for cell, ev in zip(cells, sat_evs):
        per = {"cv": {}, "osint": {}}
        for it in cell["items"]:
            if it["modality"] not in per:
                continue
            ts = it["ts"]
            age_h = ((now - ts).total_seconds() / 3600.0) if ts else config.FUSION_MAX_AGE_DAYS * 24.0
            if it["modality"] == "cv":
                rel = (cv_reliability_by_model or {}).get(it.get("model"), config.FUSION_CV_UNVALIDATED_RELIABILITY)
                geo_idx = len(config.FUSION_GEO_PRECISION)           # sentinel column == factor 1.0
                cv_rels.append(rel)
            else:
                rel = config.FUSION_OSINT_SOURCE_RELIABILITY.get(it.get("platform"),
                                                                 config.FUSION_OSINT_SOURCE_RELIABILITY["default"])
                geo_idx = _geo_bin(it.get("geo_radius_m"))
                os_rels.append(rel)
            per[it["modality"]].setdefault(it["source_key"], []).append(
                {"raw": float(np.clip(it["raw"], 0, 1)), "age": float(age_h), "rel0": float(rel), "geo_idx": geo_idx})
        cv_groups.append([v for _, v in sorted(per["cv"].items())])
        os_groups.append([v for _, v in sorted(per["osint"].items())])
        osint_items = [i for i in cell["items"] if i["modality"] == "osint"]
        urg.append(max([i.get("urgency", 0.0) for i in osint_items] or [0.0]))
        base_sev.append(config.BASE_SEVERITY.get(cell["hazard"], 0.5))
        rain.append(np.nan if rainfall_mm is None else
                    float(np.clip(rainfall_mm / 100.0, 0, 1)) * config.RAIN_SENSITIVE.get(cell["hazard"], 0.2))
        h_idx.append(hazard_names.index(cell["hazard"]))
        applic_cfg = float(config.SAT_APPLICABILITY.get(cell["hazard"], 0.0))
        confirmed = ev is not None and ev.verdict == sat.CONFIRMED_WET
        nochange = ev is not None and ev.verdict == sat.NO_CHANGE
        t = fusion.parse_ts(ev.scene_time) if ev is not None and ev.scene_time else None
        age = ((now - t).total_seconds() / 3600.0) if t else 0.0
        for k, v in (("confirmed", confirmed), ("nochange", nochange),
                     ("support", ev.support if ev else 0.0), ("quality", ev.quality if ev else 0.0),
                     ("age", age), ("applic_item", ev.applicability if ev else 0.0), ("applic_cfg", applic_cfg)):
            sat_rows[k].append(float(v))
        if cross_check:
            baseline_fusion.append(fusion.score_cell(cell, ev, now=now, rainfall_mm=rainfall_mm,
                                                     cv_reliability_by_model=cv_reliability_by_model))
    fill = {"raw": 0.0, "age": 0.0, "rel0": 0.5, "geo_idx": len(config.FUSION_GEO_PRECISION)}
    return CompiledCells(
        cell_ids=[c["cell_id"] for c in cells], hazards=[c["hazard"] for c in cells],
        hazard_idx=np.array(h_idx, dtype=np.int64), hazard_names=hazard_names,
        cv=_pad(cv_groups, fill), osint=_pad(os_groups, fill),
        sat={k: np.array(v, dtype=np.float64) for k, v in sat_rows.items()},
        urgency=np.array(urg), rain=np.array(rain), base_sev=np.array(base_sev),
        mean_rel={"cv": float(np.mean(cv_rels)) if cv_rels else float(config.FUSION_CV_UNVALIDATED_RELIABILITY),
                  "osint": float(np.mean(os_rels)) if os_rels else float(config.FUSION_OSINT_SOURCE_RELIABILITY["default"])},
        has_synthetic=any(i["synthetic"] for c in cells for i in c["items"]) or any(e is not None and e.is_synthetic for e in sat_evs),
        baseline_fusion=baseline_fusion)


def load_cells(conn, *, now=None, rainfall_mm=None, cv_reliability_by_model=None):
    """Compiles every hazard cell currently supported by evidence in the database (read-only)."""
    now = now or dt.datetime.now(UTC)
    since = (now - dt.timedelta(days=config.FUSION_MAX_AGE_DAYS)).isoformat(timespec="seconds")
    cells = fusion.group_into_cells(fusion.collect_cv(conn, since) + fusion.collect_osint(conn, since))
    evs = [sat.find_evidence(conn, c["lat"], c["lon"], c["hazard"], since_iso=since) for c in cells]
    return compile_cells(cells, evs, now=now, rainfall_mm=rainfall_mm, cv_reliability_by_model=cv_reliability_by_model)


# ====================================================================== parameters and perturbation
def baseline_theta(comp):
    """The unperturbed parameter set (floats) and the kind of each parameter ('prob' | 'pos' | 'shift')."""
    theta, kinds = {}, {}

    def put(name, value, kind):
        theta[name], kinds[name] = float(value), kind

    for m in ("cv", "osint", "satellite"):
        put(f"cap_{m}", config.FUSION_MODALITY_CAP[m], "prob")
        put(f"hl_{m}", config.FUSION_HALF_LIFE_H[m], "pos")
    put("rel_shift_cv", 0.0, "shift")
    put("rel_shift_osint", 0.0, "shift")
    for i in range(1, len(config.FUSION_GEO_PRECISION)):
        put(f"geo_{i}", config.FUSION_GEO_PRECISION[i][1], "prob")
    put("lam", config.FUSION_CONTRADICTION_LAMBDA, "prob")
    put("q_min", config.FUSION_CONTRADICTION_MIN_QUALITY, "prob")
    put("other_min", config.FUSION_CONTRADICTION_MIN_OTHER_EVIDENCE, "prob")
    put("w_sev", config.RISK_WEIGHTS["severity"], "pos")
    put("w_env", config.RISK_WEIGHTS["environment"], "pos")
    put("urg_boost", config.FUSION_URGENCY_SEVERITY_BOOST, "prob")
    for h in comp.hazard_names:
        put(f"sev_{h}", config.BASE_SEVERITY.get(h, 0.5), "prob")
    return theta, kinds


def perturb(theta0, kinds, comp, sigma, n, rng, *, scales=None, override=None):
    """Gaussian perturbation of every parameter. Returns {name: array(n,)}.

    sigma is a RELATIVE size: the sd of a probability-like parameter p is about sigma*p (applied in logit space,
    sd_logit = sigma/(1-p), so it stays inside (0, 1)); positive scales (half-lives, weights) get log-normal noise
    with sd_log = sigma; reliability shifts are additive logit offsets with the sd a probability of the typical
    reliability would have. scales: optional per-parameter multipliers; override: {name: array(n,)} to inject
    externally-sampled parameters (e.g. MCMC posterior draws) instead of perturbing them."""
    scales = scales or {}
    override = override or {}
    out = {}
    for name, x0 in theta0.items():
        if name in override:
            v = np.asarray(override[name], dtype=np.float64)
            if v.shape != (n,):
                raise ValueError(f"override for {name} must have shape ({n},)")
            out[name] = v
            continue
        z = rng.standard_normal(n)                      # always drawn, so common random numbers line up
        s = sigma * scales.get(name, 1.0)
        kind = kinds[name]
        if kind == "prob":
            out[name] = _expit(_logit(x0) + z * s / max(1.0 - x0, 0.05))
        elif kind == "pos":
            out[name] = x0 * np.exp(z * s)
        else:                                            # shift (additive logit offset)
            mean_rel = comp.mean_rel["cv" if name.endswith("cv") else "osint"]
            out[name] = x0 + z * s / max(1.0 - mean_rel, 0.05)
    return out


def _single(theta0):
    """theta0 as arrays of length 1 (the unperturbed evaluation)."""
    return {k: np.array([v], dtype=np.float64) for k, v in theta0.items()}


# ====================================================================== the vectorised fusion
def _modality_strength(block, hl, cap, rel_shift, geo_table, top_m=None):
    raw, age, rel0, gidx = block["raw"], block["age"], block["rel0"], block["geo_idx"]
    n = hl.shape[0]
    rel = _expit(_logit(rel0)[None] + rel_shift[:, None, None, None])               # (N,C,S,K)
    geo = np.take(geo_table, gidx, axis=1)                                          # (N,C,S,K)
    decay = 0.5 ** (np.maximum(age, 0.0)[None] / hl[:, None, None, None])
    s = raw[None] * rel * geo * decay
    per_source = s.max(axis=3)                                                      # best item per independent source
    if top_m is not None and per_source.shape[2] > top_m:
        per_source = -np.sort(-per_source, axis=2)[:, :, :top_m]
    conf = 1.0 - np.prod(1.0 - per_source, axis=2)
    return np.minimum(cap[:, None], conf)                                           # (N,C)


def strengths(theta, comp):
    """(s_cv, s_osint, s_sat), each (N, C): per-modality confidence for every draw and cell."""
    n = next(iter(theta.values())).shape[0]
    geo_table = np.ones((n, len(config.FUSION_GEO_PRECISION) + 1))
    for i in range(1, len(config.FUSION_GEO_PRECISION)):
        geo_table[:, i] = theta[f"geo_{i}"]
    s_cv = _modality_strength(comp.cv, theta["hl_cv"], theta["cap_cv"], theta["rel_shift_cv"], geo_table)
    s_os = _modality_strength(comp.osint, theta["hl_osint"], theta["cap_osint"], theta["rel_shift_osint"], geo_table,
                              top_m=config.FUSION_OSINT_MAX_AUTHORS)
    d = comp.sat
    raw = np.clip(d["support"] * d["quality"] * d["applic_item"], 0.0, 1.0)
    decay = 0.5 ** (np.maximum(d["age"], 0.0)[None] / theta["hl_satellite"][:, None])
    s_sat = np.minimum(theta["cap_satellite"][:, None], raw[None] * decay) * d["confirmed"][None]
    return s_cv, s_os, s_sat


def finish(theta, comp, s_cv, s_os, s_sat, *, sat_info=True):
    """Fuses modality strengths into E, impact, bands, category and score. `sat_info=False` also switches off the
    satellite contradiction rule (used when the satellite stream is removed in the leave-one-out analysis)."""
    d = comp.sat
    e_raw = _noisy_or_arr(s_cv, s_os, s_sat)
    others = _noisy_or_arr(s_cv, s_os)
    gate = ((d["nochange"][None] > 0) & (d["quality"][None] >= theta["q_min"][:, None])
            & (others >= theta["other_min"][:, None]) & (d["applic_cfg"][None] > 0)) if sat_info else False
    factor = 1.0 - theta["lam"][:, None] * d["quality"][None] * d["applic_cfg"][None]
    e = np.clip(np.where(gate, e_raw * factor, e_raw), 0.0, 1.0)
    sev_tab = np.stack([theta[f"sev_{h}"] for h in comp.hazard_names], axis=1)       # (N,H)
    sev = np.clip(sev_tab[:, comp.hazard_idx] + theta["urg_boost"][:, None] * comp.urgency[None], 0.0, 1.0)
    known = ~np.isnan(comp.rain)
    rain = np.where(known, comp.rain, 0.0)[None]
    w_s, w_e = theta["w_sev"][:, None], theta["w_env"][:, None]
    impact = np.where(known[None], (w_s * sev + w_e * rain) / (w_s + w_e), sev)
    lb, ib = _band(e), _band(impact)
    return {"E": e, "impact": impact, "L": lb, "I": ib, "cat": _category_index(lb, ib), "score": 100.0 * e * impact}


def evaluate(theta, comp, drop=None):
    """Full fusion for N parameter vectors. drop: None or one of MODALITIES to remove that evidence stream."""
    s = list(strengths(theta, comp))
    if drop is not None:
        s[MODALITIES.index(drop)] = np.zeros_like(s[0])
    out = finish(theta, comp, *s, sat_info=(drop != "satellite"))
    out["strengths"] = np.stack(s)
    return out


# ====================================================================== sensitivity run
def _ci(a, ci):
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    return np.percentile(a, [lo, hi], axis=0)


def _chunks(n, per):
    i = 0
    while i < n:
        yield min(per, n - i)
        i += per


def _draw_eval(theta0, kinds, comp, n, sigma, rng, *, drops=(), chunk=None, override=None, scales=None):
    """Runs n perturbed evaluations in chunks. -> {None|modality: {"E","cat","score","impact","L","I"}} arrays (n, C)."""
    if comp.n_cells == 0:
        raise ValueError("no hazard cells to simulate")
    size = comp.cv["raw"].shape[1] * comp.cv["raw"].shape[2] + comp.osint["raw"].shape[1] * comp.osint["raw"].shape[2]
    per = chunk or max(25, int(3e6 / max(1, comp.n_cells * size)))
    keys = (None,) + tuple(drops)
    acc = {k: {f: [] for f in ("E", "cat", "score", "impact", "L", "I")} for k in keys}
    done = 0
    for m in _chunks(n, per):
        ov = None if override is None else {k: np.asarray(v)[done:done + m] for k, v in override.items()}
        th = perturb(theta0, kinds, comp, sigma, m, rng, override=ov, scales=scales)
        for k in keys:
            r = evaluate(th, comp, drop=k)
            for f in acc[k]:
                acc[k][f].append(r[f].astype(np.float32) if f in ("E", "score", "impact") else r[f].astype(np.int8))
        done += m
    return {k: {f: np.concatenate(v) for f, v in d.items()} for k, d in acc.items()}


def category_margin(E0, impact0):
    """Smallest absolute move of E (impact held fixed) or of impact (E held fixed) that changes the matrix category.
    Returns (margin_E, margin_impact); inf if no move within [0, 1] changes it."""
    L0, I0 = int(_band(np.array(E0))), int(_band(np.array(impact0)))
    cat0 = int(_category_index(np.array(L0), np.array(I0)))
    best = {"E": math.inf, "I": math.inf}
    for L in range(1, 6):
        if L != L0 and int(_category_index(np.array(L), np.array(I0))) != cat0:
            d = ((L - 1) / 5.0 - E0) if L > L0 else (E0 - L / 5.0)
            best["E"] = min(best["E"], abs(d))
    for I in range(1, 6):
        if I != I0 and int(_category_index(np.array(L0), np.array(I))) != cat0:
            d = ((I - 1) / 5.0 - impact0) if I > I0 else (impact0 - I / 5.0)
            best["I"] = min(best["I"], abs(d))
    return best["E"], best["I"]


def _rel_sd(E):
    """sd/mean of E per cell over the draws (0 where the mean is ~0)."""
    mean = E.mean(axis=0)
    return np.where(mean > 1e-9, E.std(axis=0) / np.maximum(mean, 1e-9), 0.0)


def _round(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


def run_sensitivity(comp, settings=None, *, override=None, scales=None, progress=None):
    """Monte Carlo sensitivity of the whole fusion + matrix. Returns a JSON-serialisable report dict (see module docstring)."""
    st = settings or SimulationSettings()
    st.validate()
    rng = np.random.default_rng(st.seed)
    theta0, kinds = baseline_theta(comp)
    report = {"meta": {
        "n_iter": st.n_iter, "sigma": st.sigma, "severe_sigma": st.severe_sigma, "seed": st.seed, "ci": st.ci,
        "n_cells": comp.n_cells, "n_parameters_perturbed": len(theta0),
        "parameters": {k: _round(v, 6) for k, v in theta0.items()},
        "config_fingerprint": fusion.config_fingerprint(), "fusion_algorithm_version": config.FUSION_ALGORITHM_VERSION,
        "synthetic_data_included": comp.has_synthetic,
        "perturbation_model": "logit-normal (probabilities) / log-normal (scales) / additive logit (reliability), "
                              "independent per parameter; sd is relative to the value (see perturb())",
        "created_at": dt.datetime.now(UTC).isoformat(timespec="seconds")},
        "cells": [], "stability": {}, "loo": {}, "ladder": []}
    if comp.n_cells == 0:
        report["note"] = "no hazard cells with evidence: nothing to simulate"
        return report

    base = evaluate(_single(theta0), comp)
    E0, imp0, cat0 = base["E"][0], base["impact"][0], base["cat"][0]
    sc0 = base["score"][0]
    margins = [category_margin(float(E0[i]), float(imp0[i])) for i in range(comp.n_cells)]

    # ---- moderate level (the headline run), with leave-one-out on the same draws
    res = _draw_eval(theta0, kinds, comp, st.n_iter, st.sigma, rng, drops=MODALITIES, override=override, scales=scales)
    full = res[None]
    stability = (full["cat"] == cat0[None]).mean(axis=0)
    e_lo, e_hi = _ci(full["E"], st.ci)
    s_lo, s_hi = _ci(full["score"], st.ci)
    i_lo, i_hi = _ci(full["impact"], st.ci)
    e_sd, s_sd = full["E"].std(axis=0), full["score"].std(axis=0)
    cat_probs = np.stack([(full["cat"] == k).mean(axis=0) for k in range(4)], axis=1)
    l_probs = np.stack([(full["L"] == k).mean(axis=0) for k in range(1, 6)], axis=1)
    for i in range(comp.n_cells):
        me, mi = margins[i]
        report["cells"].append({
            "cell_id": comp.cell_ids[i], "hazard_type": comp.hazards[i],
            "baseline": {"E": _round(E0[i]), "impact": _round(imp0[i]), "category": _CAT_ORDER[int(cat0[i])],
                         "risk_score": _round(sc0[i], 2), "matrix_cell": f"L{int(base['L'][0][i])}xI{int(base['I'][0][i])}"},
            "E": {"mean": _round(full["E"][:, i].mean()), "sd": _round(e_sd[i]), "lo": _round(e_lo[i]), "hi": _round(e_hi[i])},
            "impact": {"lo": _round(i_lo[i]), "hi": _round(i_hi[i])},
            "risk_score": {"mean": _round(full["score"][:, i].mean(), 2), "sd": _round(s_sd[i], 2),
                           "lo": _round(s_lo[i], 2), "hi": _round(s_hi[i], 2)},
            "category_probs": {name: _round(cat_probs[i, k]) for k, name in enumerate(_CAT_ORDER)},
            "likelihood_band_probs": [_round(p) for p in l_probs[i]],
            "stability": _round(stability[i]),
            "margin": {"E": _round(me), "impact": _round(mi),
                       "E_in_sd": _round(me / e_sd[i]) if e_sd[i] > 0 and math.isfinite(me) else None}})
    report["stability"] = {
        "overall_category_stability": _round(stability.mean()),
        "cells_with_stability_ge_target": _round((stability >= st.stability_target).mean()),
        "worst_cell_stability": _round(stability.min()),
        "mean_risk_score_sd": _round(s_sd.mean(), 3), "mean_E_sd": _round(e_sd.mean(), 4),
        "mean_E_ci_width": _round((e_hi - e_lo).mean(), 4),
        "median_margin_E": _round(np.median([m[0] for m in margins if math.isfinite(m[0])] or [math.nan])),
        "stability_target": st.stability_target}

    # ---- leave-one-out at the moderate level and at the severe level
    def loo_block(r, i_cells, sigma_label):
        rows = []
        summary = {}
        for m in MODALITIES:
            drop = r[m]
            contrib = r[None]["E"] - drop["E"]
            stab_full = (r[None]["cat"] == cat0[None]).mean(axis=0)
            stab_drop = (drop["cat"] == cat0[None]).mean(axis=0)
            var_full, var_drop = r[None]["score"].var(axis=0), drop["score"].var(axis=0)
            c_lo, c_hi = _ci(contrib, st.ci)
            active = base["strengths"][MODALITIES.index(m), 0] > 1e-9
            summary[m] = {
                "n_cells_with_stream": int(active.sum()),
                "mean_marginal_contribution": _round(contrib[:, active].mean()) if active.any() else None,
                "mean_stability_with": _round(stab_full[active].mean()) if active.any() else None,
                "mean_stability_without": _round(stab_drop[active].mean()) if active.any() else None,
                "stability_lost_when_removed": _round((stab_full - stab_drop)[active].mean()) if active.any() else None,
                "mean_score_sd_with": _round(np.sqrt(var_full[active]).mean(), 3) if active.any() else None,
                "mean_score_sd_without": _round(np.sqrt(var_drop[active]).mean(), 3) if active.any() else None,
                # relative uncertainty (sd / mean of E): does the stream make the evidence score proportionally firmer?
                "mean_E_relative_sd_with": _round(_rel_sd(r[None]["E"])[active].mean()) if active.any() else None,
                "mean_E_relative_sd_without": _round(_rel_sd(drop["E"])[active].mean()) if active.any() else None}
            for i in i_cells:
                if sigma_label == "moderate":
                    rows.append((i, m, contrib[:, i], c_lo[i], c_hi[i], stab_full[i], stab_drop[i], active[i]))
        return summary, rows

    cells_idx = range(comp.n_cells)
    summ_mod, rows = loo_block(res, cells_idx, "moderate")
    by_cell = {}
    for i, m, contrib, lo, hi, sf, sd, active in rows:
        by_cell.setdefault(i, {"cell_id": comp.cell_ids[i], "hazard_type": comp.hazards[i]})[m] = {
            "baseline_strength": _round(base["strengths"][MODALITIES.index(m), 0][i]),
            "marginal_contribution": {"mean": _round(contrib.mean()), "lo": _round(lo), "hi": _round(hi)},
            "stability_with": _round(sf), "stability_without": _round(sd), "stream_present": bool(active)}
    sev = _draw_eval(theta0, kinds, comp, st.n_iter, st.severe_sigma, rng, drops=MODALITIES, override=override, scales=scales)
    summ_sev, _ = loo_block(sev, [], "severe")
    ranking = sorted(((m, v["stability_lost_when_removed"]) for m, v in summ_sev.items() if v["stability_lost_when_removed"] is not None),
                     key=lambda t: -t[1])
    report["loo"] = {"matrix": [by_cell[i] for i in sorted(by_cell)], "summary_moderate": summ_mod, "summary_severe": summ_sev,
                     "stabilizer_ranking_severe": [{"stream": m, "stability_lost_when_removed": v} for m, v in ranking],
                     "severe_overall_category_stability": _round((sev[None]["cat"] == cat0[None]).mean())}

    # ---- tolerance ladder: how large can the perturbation get before categories start to flip?
    ladder = []
    for sg in st.ladder:
        r = _draw_eval(theta0, kinds, comp, st.n_ladder, sg, rng, override=override, scales=scales)[None]
        stab = (r["cat"] == cat0[None]).mean(axis=0)
        ladder.append({"sigma": sg, "overall_stability": _round(stab.mean()),
                       "cells_stable_fraction": _round((stab >= st.stability_target).mean())})
        if progress:
            progress(f"ladder sigma={sg}")
    report["ladder"] = ladder
    tol = _interp_tolerance(ladder, st.stability_target)
    report["stability"]["tolerance_sigma"] = _round(tol, 3)
    report["stability"]["tolerance_below_ladder"] = bool(tol <= 0.0)
    report["stability"]["tolerance_definition"] = (
        f"largest relative parameter perturbation (sd as a share of each value) at which the average cell keeps its "
        f"matrix category with probability >= {st.stability_target:.0%}")
    return report


def _interp_tolerance(ladder, target):
    xs = [(l["sigma"], l["overall_stability"]) for l in ladder]
    if xs[0][1] < target:
        return 0.0
    for (s0, p0), (s1, p1) in zip(xs, xs[1:]):
        if p0 >= target > p1:
            return s0 + (p0 - target) / (p0 - p1) * (s1 - s0)
    return xs[-1][0]                                        # never dropped below the target on the ladder


# ====================================================================== MCMC (random-walk Metropolis)
def metropolis(logpost, x0, n_steps, *, rng, n_chains=4, burn_in=None, init_scale=0.2, thin=1, target_accept=0.30):
    """Vectorised random-walk Metropolis. logpost(x: (n_chains, d)) -> (n_chains,). x0: (d,) start (each chain is
    started from x0 plus noise). The proposal scale is adapted per chain during burn-in towards `target_accept`
    and then frozen, so the retained draws come from a valid (non-adaptive) chain. Returns a dict with
    samples (n_chains, n_keep, d), accept_rate, rhat (d,), ess (d,)."""
    x0 = np.atleast_1d(np.asarray(x0, dtype=np.float64))
    d = x0.size
    burn_in = n_steps // 2 if burn_in is None else int(burn_in)
    if not 0 <= burn_in < n_steps:
        raise ValueError("burn_in must be in [0, n_steps)")
    x = x0[None] + rng.normal(0, init_scale, (n_chains, d))
    lp = np.asarray(logpost(x), dtype=np.float64)
    if not np.all(np.isfinite(lp)):
        raise ValueError("log-posterior is not finite at the starting points")
    scale = np.full(n_chains, init_scale)
    kept, acc_kept, tot_kept = [], np.zeros(n_chains), 0
    acc_win = np.zeros(n_chains)
    for t in range(n_steps):
        prop = x + rng.normal(0, 1, (n_chains, d)) * scale[:, None]
        lp_prop = np.asarray(logpost(prop), dtype=np.float64)
        lp_prop = np.where(np.isfinite(lp_prop), lp_prop, -np.inf)
        accept = np.log(rng.random(n_chains)) < (lp_prop - lp)
        x = np.where(accept[:, None], prop, x)
        lp = np.where(accept, lp_prop, lp)
        if t < burn_in:
            acc_win += accept
            if (t + 1) % 50 == 0:
                rate = acc_win / 50.0
                scale *= np.exp(np.clip(rate - target_accept, -0.3, 0.3))
                acc_win[:] = 0
        else:
            acc_kept += accept
            tot_kept += 1
            if (t - burn_in) % thin == 0:
                kept.append(x.copy())
    samples = np.stack(kept, axis=1)                         # (chains, keep, d)
    return {"samples": samples, "accept_rate": float((acc_kept / max(tot_kept, 1)).mean()),
            "rhat": _rhat(samples), "ess": _ess(samples), "final_scale": scale.tolist()}


def _rhat(samples):
    """Gelman-Rubin potential scale reduction on split chains. (chains, n, d) -> (d,)."""
    c, n, d = samples.shape
    if n < 4:
        return np.full(d, np.nan)
    half = n // 2
    s = np.concatenate([samples[:, :half], samples[:, half:2 * half]], axis=0)       # 2c split chains
    m, ln = s.shape[0], s.shape[1]
    means, var = s.mean(axis=1), s.var(axis=1, ddof=1)
    w, b = var.mean(axis=0), ln * means.var(axis=0, ddof=1)
    var_plus = (ln - 1) / ln * w + b / ln
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_plus / w)


def _ess(samples):
    """Effective sample size per dimension (Geyer initial positive sequence on the pooled autocorrelation)."""
    c, n, d = samples.shape
    out = np.zeros(d)
    for j in range(d):
        x = samples[:, :, j] - samples[:, :, j].mean(axis=1, keepdims=True)
        f = np.fft.rfft(x, n=2 * n, axis=1)
        acov = np.fft.irfft(f * np.conj(f), axis=1)[:, :n] / n
        rho = acov.mean(axis=0) / max(acov[:, 0].mean(), 1e-300)
        s, k = 0.0, 1
        while k + 1 < n:
            pair = rho[k] + rho[k + 1]
            if pair < 0:
                break
            s += pair
            k += 2
        out[j] = c * n / (1.0 + 2.0 * s)
    return out


def calibrate_reliabilities(comp, labels, *, n_steps=3000, n_chains=4, prior_sd=0.7, seed=7, ci=0.95):
    """Bayesian calibration of how reliable CV and OSINT evidence is, from labelled hazard cells.

    labels: array-like of 0/1 per compiled cell (1 = a field check confirmed the hazard, 0 = refuted).
    Parameters: delta_cv, delta_osint = additive LOGIT shifts of every item's reliability (0 = expert default).
    Prior: Normal(0, prior_sd) on each shift (a mild pull towards the expert defaults).
    Likelihood: y_c ~ Bernoulli(E_c(theta)) with E clipped to [1e-3, 1-1e-3] - E is TREATED as a probability here.
    Returns posterior summaries (mean, sd, credible interval), implied reliabilities for a typical item,
    convergence diagnostics (rhat < 1.05 and ess > 200 are the usual bars) and the raw draws."""
    y = np.asarray(labels, dtype=np.float64)
    if y.shape != (comp.n_cells,):
        raise ValueError(f"labels must have shape ({comp.n_cells},)")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("labels must be 0 or 1")
    if comp.n_cells < 10:
        raise ValueError("need at least 10 labelled cells to calibrate anything")
    theta0, _ = baseline_theta(comp)

    def logpost(x):
        n = x.shape[0]
        th = {k: np.full(n, v) for k, v in theta0.items()}
        th["rel_shift_cv"], th["rel_shift_osint"] = x[:, 0], x[:, 1]
        e = np.clip(evaluate(th, comp)["E"], 1e-3, 1 - 1e-3)
        ll = (y[None] * np.log(e) + (1 - y[None]) * np.log(1 - e)).sum(axis=1)
        return ll - 0.5 * ((x / prior_sd) ** 2).sum(axis=1)

    rng = np.random.default_rng(seed)
    out = metropolis(logpost, np.zeros(2), n_steps, rng=rng, n_chains=n_chains)
    flat = out["samples"].reshape(-1, 2)
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    names = ("delta_cv", "delta_osint")
    base_rel = (comp.mean_rel["cv"], comp.mean_rel["osint"])
    post = {}
    for j, nm in enumerate(names):
        implied = _expit(_logit(base_rel[j]) + flat[:, j])
        post[nm] = {"mean": _round(flat[:, j].mean()), "sd": _round(flat[:, j].std()),
                    "lo": _round(np.percentile(flat[:, j], lo)), "hi": _round(np.percentile(flat[:, j], hi)),
                    "implied_reliability": {"default": _round(base_rel[j]), "mean": _round(implied.mean()),
                                            "lo": _round(np.percentile(implied, lo)), "hi": _round(np.percentile(implied, hi))}}
    converged = bool(np.all(out["rhat"] < 1.05) and np.all(out["ess"] > 200))
    return {"posterior": post, "accept_rate": _round(out["accept_rate"], 3), "rhat": [_round(v, 4) for v in out["rhat"]],
            "ess": [_round(v, 1) for v in out["ess"]], "converged": converged, "n_labelled": int(comp.n_cells),
            "n_positive": int(y.sum()), "prior_sd": prior_sd,
            "assumption": "E is treated as the probability that the hazard is real (Bernoulli likelihood).",
            "samples": flat}


def posterior_override(calibration, n):
    """Turns calibration draws into an `override` dict for run_sensitivity (resampling with replacement)."""
    flat = np.asarray(calibration["samples"])
    idx = np.random.default_rng(0).integers(0, len(flat), n)
    return {"rel_shift_cv": flat[idx, 0], "rel_shift_osint": flat[idx, 1]}


# ====================================================================== synthetic cells (offline / tests / demo)
def synthetic_cells(n_cells=40, seed=1, now=None):
    """Synthetic grouped cells + satellite evidence for offline use. Everything is flagged synthetic."""
    rng = np.random.default_rng(seed)
    now = now or dt.datetime.now(UTC)
    hazards = ["waterlogging", "sewage_overflow", "pothole", "open_manhole", "open_drain", "garbage"]
    cells, evs = [], []
    for i in range(n_cells):
        hz = hazards[int(rng.integers(len(hazards)))]
        lat, lon = 24.80 + 0.2 * rng.random(), 67.00 + 0.2 * rng.random()
        items = []
        for k in range(int(rng.poisson(1.0))):
            items.append({"modality": "cv", "hazard": hz, "lat": lat, "lon": lon,
                          "ts": now - dt.timedelta(hours=float(rng.exponential(120))), "raw": float(rng.uniform(.35, .95)),
                          "source_key": f"cv:{i}:{k}", "geo_radius_m": 25.0, "synthetic": True, "model": "synthetic:1",
                          "ref": f"cv:{i}:{k}", "label": hz})
        for k in range(int(rng.poisson(2.5))):
            items.append({"modality": "osint", "hazard": hz, "lat": lat, "lon": lon,
                          "ts": now - dt.timedelta(hours=float(rng.exponential(30))), "raw": float(rng.uniform(.3, .95)),
                          "source_key": f"osint:{i}:{k % 3}", "geo_radius_m": float(rng.choice([30, 800, 1500, 3000])),
                          "synthetic": True, "urgency": float(rng.random() * .6), "platform": "mock", "ref": f"osint:{i}:{k}"})
        if not items:
            items.append({"modality": "osint", "hazard": hz, "lat": lat, "lon": lon, "ts": now, "raw": .6,
                          "source_key": f"osint:{i}:0", "geo_radius_m": 800.0, "synthetic": True, "urgency": 0.1,
                          "platform": "mock", "ref": f"osint:{i}:x"})
        cells.append({"hazard": hz, "anchor_lat": lat, "anchor_lon": lon, "items": items, "cell_id": f"{hz}|{lat:.4f}|{lon:.4f}|{i}",
                      "lat": lat, "lon": lon})
        if config.SAT_APPLICABILITY.get(hz, 0.0) > 0 and rng.random() < 0.7:
            verdict = sat.CONFIRMED_WET if rng.random() < 0.55 else (sat.NO_CHANGE if rng.random() < 0.6 else sat.INCONCLUSIVE)
            evs.append(sat.SatelliteEvidence(
                verdict=verdict, hazard_type=hz, lat=lat, lon=lon, aoi_radius_m=500.0, event_time=now.isoformat(),
                support=float(rng.uniform(.4, 1)) if verdict == sat.CONFIRMED_WET else 0.0, quality=float(rng.uniform(.5, 1)),
                applicability=float(config.SAT_APPLICABILITY[hz]), scene_time=(now - dt.timedelta(hours=float(rng.uniform(6, 90)))).isoformat(),
                scene_id=f"SYN-{i}", is_synthetic=True, check_id=i, result_sha256="x" * 8))
        else:
            evs.append(None)
    return cells, evs


# ====================================================================== presentation + integrity
def report_tables(report):
    """pandas DataFrames for the UI: {'cells', 'loo', 'ladder', 'loo_summary_moderate', 'loo_summary_severe'}."""
    import pandas as pd
    cells = pd.DataFrame([{
        "cell": c["cell_id"], "hazard": c["hazard_type"], "category": c["baseline"]["category"],
        "matrix": c["baseline"]["matrix_cell"], "E": c["baseline"]["E"], "E_lo": c["E"]["lo"], "E_hi": c["E"]["hi"],
        "risk": c["baseline"]["risk_score"], "risk_lo": c["risk_score"]["lo"], "risk_hi": c["risk_score"]["hi"],
        "P(same category)": c["stability"], "margin_E": c["margin"]["E"], "margin_impact": c["margin"]["impact"]}
        for c in report.get("cells", [])])
    rows = []
    for r in report.get("loo", {}).get("matrix", []):
        for m in MODALITIES:
            v = r.get(m)
            if v and v["stream_present"]:
                rows.append({"cell": r["cell_id"], "stream": m, "baseline_strength": v["baseline_strength"],
                             "contribution": v["marginal_contribution"]["mean"],
                             "contrib_lo": v["marginal_contribution"]["lo"], "contrib_hi": v["marginal_contribution"]["hi"],
                             "stability_with": v["stability_with"], "stability_without": v["stability_without"]})
    loo = pd.DataFrame(rows)
    sm = lambda key: pd.DataFrame(report.get("loo", {}).get(key, {})).T
    return {"cells": cells, "loo": loo, "ladder": pd.DataFrame(report.get("ladder", [])),
            "loo_summary_moderate": sm("summary_moderate"), "loo_summary_severe": sm("summary_severe")}


def report_digest(report):
    """SHA-256 of the report's reproducible content (timestamps excluded), so it can be anchored and re-verified."""
    body = {k: v for k, v in report.items() if k != "calibration"}
    body["meta"] = {k: v for k, v in report["meta"].items() if k != "created_at"}
    return kpdb.sha256_hex(kpdb.canonical_json(body))


def anchor_report(conn, report):
    """Appends a compact summary + digest of the report to the hash-chained ledger, so the sensitivity analysis that
    justified a configuration is itself tamper-evident. Returns (seq, hash). Does not touch any other table."""
    s = report.get("stability", {})
    payload = {"digest": report_digest(report), "n_iter": report["meta"]["n_iter"], "sigma": report["meta"]["sigma"],
               "seed": report["meta"]["seed"], "n_cells": report["meta"]["n_cells"],
               "overall_category_stability": s.get("overall_category_stability"),
               "tolerance_sigma": s.get("tolerance_sigma"), "config_fingerprint": report["meta"]["config_fingerprint"],
               "synthetic_data_included": report["meta"]["synthetic_data_included"]}
    return kpdb.append_ledger(conn, "sensitivity_report", payload)


def markdown_report(report):
    m, s = report["meta"], report.get("stability", {})
    lines = [f"# Fusion sensitivity report", "",
             f"{m['n_iter']:,} Monte Carlo draws, relative sigma {m['sigma']}, {m['n_cells']} hazard cells, "
             f"{m['n_parameters_perturbed']} parameters perturbed, seed {m['seed']}.", ""]
    if m["synthetic_data_included"]:
        lines += ["> Includes synthetic data: figures describe the simulator, not Karachi.", ""]
    if not report.get("cells"):
        return "\n".join(lines + [report.get("note", "No cells.")])
    lines += ["## Structural stability", "",
              f"* Average probability a cell keeps its matrix category: **{s['overall_category_stability']:.1%}**",
              f"* Cells that keep it with probability >= {s['stability_target']:.0%}: **{s['cells_with_stability_ge_target']:.0%}** "
              f"(worst cell: {s['worst_cell_stability']:.1%})",
              (f"* Tolerance: **below the smallest tested perturbation ({report['ladder'][0]['sigma']})** - even tiny "
               f"parameter changes move some cells across a band edge ({s['tolerance_definition']})"
               if s.get("tolerance_below_ladder") else
               f"* Tolerance: categories stay stable up to a relative perturbation of **{s['tolerance_sigma']}** "
               f"({s['tolerance_definition']})"),
              f"* Mean {int(m['ci'] * 100)}% interval width of the evidence score E: {s['mean_E_ci_width']}; "
              f"mean risk-score sd: {s['mean_risk_score_sd']}", "",
              "## Which stream stabilises the matrix?", "",
              f"Under severe uncertainty (sigma {m['severe_sigma']}), average category stability over all cells is "
              f"{report['loo']['severe_overall_category_stability']:.1%}. Stability lost when a stream is removed "
              "(only cells where that stream carries evidence):", ""]
    for r in report["loo"]["stabilizer_ranking_severe"]:
        lines.append(f"* {r['stream']}: {r['stability_lost_when_removed']:+.3f}")
    lines += ["", "Positive = removing the stream makes the category less reproducible. That measures how much the result "
              "DEPENDS on the stream (it also moves the evidence level, so it is not a pure variance effect); compare "
              "`mean_E_relative_sd_with/without` in the JSON for the proportional-uncertainty view. Neither says a stream "
              "is more accurate.", ""]
    return "\n".join(lines)


# ====================================================================== CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m evaluation.simulation", description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=config.DB_PATH, help="SQLite database with fused evidence (default: app DB)")
    ap.add_argument("--synthetic", type=int, default=0, help="ignore the DB and simulate N synthetic cells")
    ap.add_argument("--n-iter", type=int, default=10_000)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--rain", type=float, default=None, help="rainfall mm to include the environment component")
    ap.add_argument("--out", help="write the JSON report here")
    ap.add_argument("--markdown", help="write a markdown summary here")
    ap.add_argument("--anchor", action="store_true", help="append the report digest to the evidence ledger of --db")
    a = ap.parse_args(argv)
    conn = None
    if a.synthetic:
        cells, evs = synthetic_cells(a.synthetic, a.seed)
        comp = compile_cells(cells, evs, rainfall_mm=a.rain)
    else:
        if not os.path.exists(a.db):
            raise SystemExit(f"database {a.db!r} not found; use --synthetic N for an offline run")
        conn = kpdb.connect(a.db)
        comp = load_cells(conn, rainfall_mm=a.rain)
    rep = run_sensitivity(comp, SimulationSettings(n_iter=a.n_iter, sigma=a.sigma, seed=a.seed))
    rep["digest"] = report_digest(rep)
    if a.anchor:
        if conn is None:
            raise SystemExit("--anchor needs a real database, not --synthetic")
        print("ledger entry:", anchor_report(conn, rep))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2)
    md = markdown_report(rep)
    if a.markdown:
        with open(a.markdown, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)


if __name__ == "__main__":
    _main()
