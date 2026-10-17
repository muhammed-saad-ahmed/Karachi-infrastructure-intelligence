# Karachi Pulse — Urban Infrastructure Intelligence

> **An independent research prototype exploring low-cost, auditable AI infrastructure-risk intelligence for Karachi.**
>
> *Not an official municipal system, safety rating, government service, or production-ready hazard detection system.*

**Author:** Muhammad Saad
**Version:** 2.0.0 (multi-modal edition)
**License:** MIT

---

## Overview

**Karachi Pulse** explores whether three cheap, independent evidence streams can be combined into an **auditable** urban-hazard picture for a resource-constrained megacity:

1. **Edge computer vision** — a YOLOv8-nano detector exported to ONNX and run on a CPU/phone, so there is no inference server to pay for.
2. **Community reports (OSINT)** — public civic-hazard posts in English, Roman Urdu and Urdu, collected politely, geolocated offline and de-duplicated.
3. **Satellite cross-verification** — Sentinel-2 (free, via Microsoft Planetary Computer) NDWI/NDVI change detection to check whether a reported flood left a mark on the ground.

They are fused by a transparent, hand-checkable algorithm into a **5×5 likelihood × impact hazard risk matrix**, stored in SQLite with an **append-only, hash-chained audit ledger** so any later edit is detectable.

The project is built around one principle:

> **A system should clearly distinguish what it knows, what it estimates, what it has measured, and what it does not yet know.**

Detection confidence, evidence strength and risk stay separate quantities; "no information" (clouds, unobservable hazard) is never confused with "no hazard"; and every score ships with the numbers that produced it.

The system is a foundation for real-world data collection — **not** a claim that Karachi's infrastructure can already be reliably detected or predicted by this prototype.

**Pipeline:** Detection → Tracking → *Community reports* → *Satellite verification* → Evidence Fusion → Geospatial Reasoning → Risk Matrix → Evaluation

Full write-up (motivation, related work, methodology, limitations): `docs/research_report.md`. Technical spec of the v2 modules: `docs/multimodal_fusion.md`.

---

## Why Karachi?

In December 2025, *Dawn* reported that **23 people, including 8 children, had died in Karachi that year after falling into open manholes and uncovered sewers** — a recurring, documented problem, not a hypothetical one. Waterlogging from blocked drains is a second, well-documented failure mode (see `docs/research_report.md`).

Karachi Pulse is an attempt to explore whether a low-cost technical layer could eventually *complement* existing infrastructure-monitoring processes. It is **not** intended to replace municipal inspection, emergency services, engineers, or official infrastructure databases.

---

## What the System Does

### 1. Edge ML runtime (`computer_vision/edge_onnx.py`)
YOLOv8-nano → ONNX → ONNX Runtime on CPU. Aspect-preserving letterbox, single-call preprocessing, YOLOv8 head decoding (raw and NMS-free exports), NumPy class-aware NMS with a bounded candidate count, ORT session tuned for edge (full graph optimisation, ≤ 4 threads, no idle spinning), latency benchmark (p50/p95/FPS), and **static INT8 quantization** that refuses to run without real calibration images and is always followed by an INT8-vs-FP32 agreement check. The registry prefers `models/custom/*.onnx`, then `*.pt`, then the COCO baseline; a model's *identity* comes from its own class names, so a COCO ONNX is never shown as a "Karachi hazard model". Training/export/quantization run on free Colab/Kaggle (`scripts/train_edge_model.py`); the deployed app needs only `onnxruntime`.

**Limitation:** no trained pothole / open-manhole / sewage weights ship with this repository — that needs a properly collected, consented, geotagged, labelled dataset. The module is the deployment path for the weights you train.

### 2. OSINT pipeline (`scripts/osint_pipeline.py`, `scripts/osint_text.py`, `geospatial/gazetteer.py`)
Asynchronous, polite collection (robots.txt incl. Crawl-delay, per-host rate limit, retries with exponential backoff + jitter, circuit breaker, size caps) of RSS/Atom and CSS-selector HTML sources (optional Playwright for JavaScript pages). A deterministic **mock stream** simulates a rate-limited public feed so everything runs offline. Posts get transparent lexicon-based hazard/urgency/sentiment scores, offline geocoding (explicit coordinates, map links, or a Karachi neighbourhood gazetteer with an explicit uncertainty radius), salted author pseudonyms, redaction of emails/phones/@mentions, and near-duplicate suppression so a viral copy-paste cannot fake corroboration.

**Limitations:** the text analysis is a lexicon, not a trained model (sarcasm, long-range negation and Roman-Urdu spelling variants fool it); gazetteer coordinates are approximate. **Scrape only what you are allowed to** — most social networks forbid it in their terms; prefer RSS, official complaint portals and open data.

### 3. Satellite fusion (`geospatial/satellite.py`)
For a strong, located report of area-scale waterlogging, fetch the clear Sentinel-2 L2A scene nearest the report and a median composite of clear dry-period scenes, mask clouds with the SCL band, compute **NDWI** (McFeeters 1996), **MNDWI** (Xu 2006) and **NDVI** (Tucker 1979), and count *newly wet, non-vegetated* pixels (speckle removed). Four explicit verdicts: `CONFIRMED_WET`, `NO_CHANGE`, `INCONCLUSIVE` (clouds / no scene / provider error), `NOT_APPLICABLE` (potholes and manholes are smaller than a 10 m pixel). Results are cached per ~550 m cell to be kind to the free API, and every result carries a digest.

**Limitations:** a 10 m pixel cannot see narrow street flooding; floods come with clouds; revisit takes days; thresholds are expert-set, not calibrated. The live provider needs network access and the optional packages in `requirements-satellite.txt`; an offline, clearly-labelled synthetic provider is used for tests and demos.

### 4. Multi-modal evidence fusion (`risk_engine/fusion.py`)
Capped, time-decayed **noisy-OR over independent sources** (one image / one author counts once), then across modalities; an explicit, bounded contradiction rule when clear satellite imagery of an observable hazard shows no change; impact computed only from *known* components (unknowns are dropped and flagged, never guessed). The output is a 5×5 likelihood × impact matrix with per-modality strengths, leave-one-out marginal contributions, flags, a plain-text explanation generated from the numbers, and an input digest + configuration fingerprint. Stored in append-only SQLite tables and recorded in a hash-chained ledger (optionally HMAC-keyed). **Tamper-evident, not tamper-proof** — see `docs/multimodal_fusion.md` for exactly what is and is not guaranteed.

### 5. Original platform (v1, unchanged)
Detection registry, IoU tracking, haversine clustering, the original transparent risk engine, Flood Lab, Traffic Lab, condition-wise error-analysis evaluation, dataset tooling (validation, perceptual-hash duplicates, **geographic** train/val/test split), experiment tracking and the recurring-hotspot rule.

---

## Current Data Status

| Component | Status |
|---|---|
| Detection pipeline, IoU tracking, geospatial clustering, risk engine | **Implemented** |
| SQLite persistence, dataset tooling, evaluation + error analysis, experiment tracking | **Implemented** |
| ONNX edge runtime (decode, NMS, benchmark, registry integration) | **Implemented**, tested end-to-end with a stub ONNX model |
| ONNX export / INT8 quantization | **Implemented — not exercised in the dev sandbox** (needs `ultralytics` / `onnx`); run on Colab and check `compare_models` |
| OSINT pipeline (fetch, parse, analyse, geocode, dedup, store) | **Implemented**, tested offline; live sources are yours to configure |
| Satellite verification — maths, verdict logic, caching, simulator | **Implemented**, tested |
| Satellite verification — live Planetary Computer provider | **Implemented — not exercised in the dev sandbox (no network)**; validate on your machine |
| Evidence fusion + risk matrix + hash-chained ledger | **Implemented**, tested (including tamper detection) |
| Traffic / flood analysis | **Experimental** |
| Hazard-specific trained model | **Not trained** |
| Real Karachi hazard dataset | **Not yet collected** |
| Calibration of fusion weights / satellite thresholds | **Not done** (needs labelled outcomes) |
| mAP / confusion matrix | **Not implemented** |

## Synthetic Data Policy

Where real observations are unavailable the prototype uses synthetic demonstration data, always explicitly labelled **SYNTHETIC**. Everything the offline demo writes — simulated posts, simulated satellite scenes, simulated CV detections — is stored with `is_synthetic = 1`, shown as such in the UI, and any risk-matrix cell containing synthetic evidence carries the flag `synthetic_data_included`.

## Honesty Statement

> **If the system cannot measure something, it says so.**

Metrics that cannot currently be computed display **NOT EVALUATED**, never an invented number. This repository does not claim detection accuracy, real-world pothole/manhole/sewage detection performance, flood-detection accuracy, government validation, municipal deployment, academic validation, calibrated probabilities, or production-ready emergency prediction. It demonstrates an **engineering, evaluation and audit framework** — not a finished city-scale monitoring system.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt     # light runtime: no PyTorch, CPU only
cp .env.example .env                # then set KP_PRIVACY_SALT (and optionally KP_LEDGER_KEY)
python -m pytest -q                 # 86 tests (31 original + 55 v2) should pass
streamlit run app.py
```

Optional extras:

| Need | Install |
|---|---|
| Live Sentinel-2 verification | `pip install -r requirements-satellite.txt` |
| Training / ONNX export / INT8 / COCO baseline detector | `pip install -r requirements-train.txt` (use a free Colab/Kaggle GPU for training) |
| Running the tests | `pip install -r requirements-dev.txt` |
| JavaScript-rendered OSINT pages | `pip install playwright && playwright install chromium` |

> Without `ultralytics` and without an ONNX model in `models/custom/`, the app starts with **no vision model loaded** (the sidebar says so) — the Risk Matrix, OSINT Feed, Satellite Check and all analytics pages still work.

## Quick start: the multi-modal demo (offline, $0, no network)

1. `streamlit run app.py` → **Risk Matrix** → **Run offline demo**.
   It simulates community posts, simulated Sentinel-2 scenes and simulated CV detections, fuses them, and shows the 5×5 matrix, the map, a per-cell "why this score?", and the integrity audit. Everything is flagged synthetic.
2. **OSINT Feed** → choose *Offline simulator*, pick a satellite mode, **Run pipeline now**.
3. **Satellite Check** → enter a coordinate (default: Gulshan-e-Iqbal) → **Verify location** to see NDWI/event/baseline/newly-wet panels.
4. Try breaking it: in a Python shell, drop a trigger and edit a row of `risk_matrix`, then reopen **Risk Matrix** — the audit panel reports exactly what no longer matches.

## From the command line

```bash
# Edge model: train on free GPU, export, quantize, verify, benchmark
python -m scripts.train_edge_model --data data/hazard_dataset.yaml --epochs 100 --calib data/raw
python -m computer_vision.edge_onnx export   --weights best.pt --out models/custom/hazard_n.onnx
python -m computer_vision.edge_onnx quantize --model models/custom/hazard_n.onnx --calib data/raw --out models/custom/hazard_n.int8.onnx
python -m computer_vision.edge_onnx compare  --fp32 models/custom/hazard_n.onnx --int8 models/custom/hazard_n.int8.onnx --images data/raw
python -m computer_vision.edge_onnx bench    --model models/custom/hazard_n.onnx
python -m computer_vision.edge_onnx infer    --model models/custom/hazard_n.onnx --image road.jpg

# OSINT
python -m scripts.osint_pipeline --mock 60 --satellite synthetic          # offline
cp data/osint_sources.example.json data/osint_sources.json                 # edit with sources you are allowed to read
python -m scripts.osint_pipeline --sources data/osint_sources.json --once --satellite live
```

From Python / a notebook:

```python
from database.db import connect, fetch_risk_matrix, verify_ledger
from risk_engine.fusion import assess_all, full_audit
conn = connect("data/karachi_pulse.db")
assess_all(conn, rainfall_mm=40)          # fuse everything currently stored
fetch_risk_matrix(conn)                   # latest assessment per hazard cell
full_audit(conn)                          # chain + matrix-vs-ledger + evidence digests
```

## Project Structure

```text
karachi-pulse/
│
├── app.py                        Streamlit application (15 pages)
├── requirements.txt              light runtime (no PyTorch)
├── requirements-satellite.txt    optional: live Sentinel-2
├── requirements-train.txt        optional: training / ONNX export / COCO baseline
├── requirements-dev.txt          optional: pytest
├── README.md
├── LICENSE
├── .env.example
├── .gitignore
│
├── app_ui/                       Design system + reusable UI components
│   ├── theme.py                  Color tokens, typography, global CSS
│   ├── components.py             Cards, gauges, dossiers, detection rendering
│   └── fusion_pages.py           Risk Matrix, OSINT Feed, Satellite Check, Edge panel
│
├── computer_vision/
│   ├── preprocessing.py          Validation, decode, face blur, frame sampling
│   ├── detection.py              Model registry (ONNX > custom .pt > baseline), benchmarking
│   ├── edge_onnx.py              ONNX Runtime edge detector, export, INT8 quantization, benchmark
│   └── tracking.py               IoU-based temporal tracking
│
├── config/
│   └── __init__.py               All tunable weights/thresholds — nothing hard-coded inline
│
├── risk_engine/
│   ├── scoring.py                Risk scoring, evidence confidence, road health
│   ├── fusion.py                 Multi-modal evidence fusion -> 5x5 risk matrix, integrity audits
│   └── traffic.py                Vehicle density categorization
│
├── geospatial/
│   ├── clustering.py             Haversine clustering, coordinate validation
│   ├── gazetteer.py              Offline Karachi geocoder (coordinates / neighbourhood names)
│   └── satellite.py              Sentinel-2 NDWI/NDVI cross-verification, providers, caching
│
├── database/
│   └── db.py                     SQLite schema, helpers, append-only triggers, hash-chained ledger
│
├── evaluation/
│   └── metrics.py                Precision/Recall/F1, IoU matching, condition-wise error analysis
│
├── prediction/
│   └── hotspots.py               Recurring-hotspot flagging (recurrence rule)
│
├── experiments/
│   └── tracker.py                Real experiment logger (log.json created on first use)
│
├── scripts/
│   ├── generate_demo_data.py     Synthetic demo data + offline multi-modal demo seeder
│   ├── dataset_tools.py          Validation, duplicate detection, geographic split, stats
│   ├── osint_pipeline.py         Async polite scraper, parsers, mock stream, pipeline, CLI
│   ├── osint_text.py             Lexicon analysis (EN / Roman Urdu / Urdu), redaction, dedup
│   └── train_edge_model.py       Train -> export -> quantize on free Colab/Kaggle
│
├── models/
│   └── custom/                   Drop a trained .onnx (preferred) or .pt here to override the baseline
│
├── data/
│   ├── raw/                      Real images go here (git-ignored)
│   ├── annotations/              Ground-truth labels for a real dataset
│   ├── demo/                     Synthetic demonstration CSV
│   ├── hazard_dataset.example.yaml   YOLO dataset config template
│   └── osint_sources.example.json    OSINT source config template
│
├── docs/
│   ├── methodology.md            Technical methodology + limitations
│   ├── multimodal_fusion.md      v2 specification: edge runtime, OSINT, satellite, fusion, ledger
│   └── research_report.md        Full research write-up with verified citations
│
└── tests/
    ├── test_core.py              31 original unit tests
    ├── test_v2.py                55 tests: ONNX runtime, OSINT, satellite, fusion, ledger tamper detection
    └── tiny_onnx.py              Builds a stub ONNX model (no `onnx` package needed) for runtime tests
```

## Database (v2 tables)

| Table | Purpose |
|---|---|
| `osint_posts` | Redacted community reports with pseudonymous author, hazard/urgency/signal, location + radius, `content_sha256`, `duplicate_of`, `is_synthetic`. `UNIQUE(platform, post_id)`; updates blocked. |
| `satellite_checks` | Verdict, support/quality, scene ids, full NDWI/NDVI metrics, `result_sha256`, provider, `is_synthetic`; updates blocked. |
| `evidence_ledger` | Append-only hash chain (SHA-256, or HMAC-SHA-256 with `KP_LEDGER_KEY`); updates and deletes blocked. |
| `risk_matrix` | One immutable row per assessment: strengths, contributions, E, impact, bands, category, score, flags, explanation, `inputs_digest`, ledger link, algorithm version, config fingerprint. |

Original v1 tables (`incidents`, `observations`, `detections`, …) are unchanged. Full column lists: `database/db.py`.

## Using the Dataset Tools

```python
from scripts.dataset_tools import dataset_summary, validate_images, detect_duplicates, geographic_split
dataset_summary("data/raw")           # image count, corrupted count, resolution spread
validate_images("data/raw")           # which files fail to decode
detect_duplicates("data/raw")         # near-duplicate pairs via perceptual hashing
geographic_split(records)             # train/val/test by location, not randomly — avoids leakage
```

## Logging an Experiment

```python
from experiments.tracker import log_experiment, list_experiments
log_experiment("hazard-n-v1", model="hazard_n.onnx", dataset_version="v0", params={"imgsz": 640}, metrics={"f1": 0.0})
list_experiments()
```

## Evaluating a Real Model

1. Collect consented, geotagged Karachi hazard images; split with `geographic_split`; label as `pothole / open_manhole / sewage_overflow` (`data/hazard_dataset.example.yaml`).
2. Train on free Colab/Kaggle with `scripts/train_edge_model.py` (exports ONNX; add `--calib` for INT8).
3. Copy the `.onnx` into `models/custom/` — the app auto-detects it, no code changes. The Model Lab page benchmarks it on your machine.
4. In **Model Lab**, upload held-out labelled images plus a ground-truth JSON for real Precision/Recall/F1 and condition-wise error analysis (`rain`, `low_light`, `blur`, `occlusion`, …). Only then does the fused matrix reflect a validated detector.

## Privacy, Ethics and Responsible Use

* No face recognition, person identification or license-plate identification is implemented or planned; faces are blurred before analysis (best-effort).
* OSINT: public pages only, robots.txt honoured, rate-limited, identifiable User-Agent; author handles salted-hashed (set `KP_PRIVACY_SALT`), emails/phones/@mentions removed, images not downloaded. Reports describe places, not people.
* Respect each source's terms of service. This tool must not be pointed at platforms that prohibit scraping.
* Risk scores are experimental and **not** an official safety rating. Do not use them for emergency decisions.

## License

MIT — see `LICENSE`.
