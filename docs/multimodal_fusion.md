# Multi-modal pipeline (v2): edge CV, OSINT, Sentinel-2, evidence fusion

This document is the technical specification for the four v2 modules. Every constant mentioned lives in
`config/__init__.py`; every claim here is backed by a test in `tests/test_v2.py`.

```text
 phone / laptop                        public web                       Sentinel-2 archive
 ───────────────                       ──────────                       ──────────────────
 image / video                         RSS / HTML / mock feed           Planetary Computer (free)
      │ ONNX Runtime (CPU)                  │ async, polite fetch               ▲
      ▼                                     ▼                                    │ only if the report is
 CV detections ───────────┐      posts → analysis → geocode → dedup             │ strong, located and
 (observations,           │            (osint_posts) ─────── alert ────────────►│ observable at 10 m
  detections)             │                  │                                  │
                          │                  │                       satellite_checks (NDWI/NDVI verdict)
                          ▼                  ▼                                  │
                       ┌────────────────────────────────────────────────────────▼┐
                       │ risk_engine/fusion.py: group → score → 5×5 matrix        │
                       └──────────────┬───────────────────────────────────────────┘
                                      ▼
                  risk_matrix (append-only rows)  +  evidence_ledger (hash chain)
```

## 1. Edge ML runtime — `computer_vision/edge_onnx.py`

| Step | What happens | Why |
|---|---|---|
| Train (free Colab/Kaggle) | `scripts/train_edge_model.py` fine-tunes `yolov8n.pt` on `pothole / open_manhole / sewage_overflow` | needs a labelled dataset — **not shipped** |
| Export | `export_to_onnx`: static 640×640 input, opset 12, simplified, FP32 | broadest runtime support (CPU, mobile, WASM) |
| Quantize (optional) | `quantize_int8_static`: QDQ INT8, per-channel weights, percentile calibration on **real** images; refuses < `EDGE_CALIBRATION_MIN_IMAGES` | dynamic quantization barely helps conv nets; noise calibration wrecks accuracy |
| Verify | `compare_models`: latency **and** detection agreement INT8↔FP32 | quantization must never be trusted blindly |
| Run | `OnnxDetector`: letterbox → one `blobFromImage` call → ORT session (graph opt `ENABLE_ALL`, ≤ 4 threads, no idle spinning) → decode → NumPy class-aware NMS | no PyTorch, no server |
| Measure | `benchmark`: p50/p95/FPS on **this** machine | speed claims must be measured on the target device |

Details that matter:

* **Decode** supports the YOLOv8/11 raw head `(1, 4+nc, N)` and the NMS-free `(1, N, 6)` head (`end2end=True`).
  Candidates are capped (`EDGE_MAX_CANDIDATES`) before NMS so CPU cost is bounded even on noisy frames.
* **Identity from the model, not the folder:** class names are read from the ONNX metadata. A model with none of the
  hazard classes is reported as `yolo-onnx-baseline`, never as a "custom Karachi model".
* The registry order is `models/custom/*.onnx` → `models/custom/*.pt` → COCO baseline.
* **Deploying with $0:** `onnxruntime` runs in Streamlit Community Cloud, on a Raspberry Pi, on Android (ONNX Runtime
  Mobile) and in the browser (`onnxruntime-web`, WebAssembly) — a browser build would remove the server entirely
  (not implemented here).

## 2. OSINT pipeline — `scripts/osint_pipeline.py`, `scripts/osint_text.py`, `geospatial/gazetteer.py`

**Collection** (`HttpFetcher`, async, stdlib transport): robots.txt (RFC 9309 semantics, `Crawl-delay`), ≥ `OSINT_MIN_DELAY_S`
per host, global concurrency cap, retries on 429/5xx/timeouts with exponential backoff + jitter and `Retry-After`,
per-host circuit breaker, http/https only, size cap. RSS/Atom parsing rejects `DOCTYPE`/`ENTITY` (entity-expansion attacks).
HTML sources use CSS selectors from a JSON spec; `html_js` sources render with Playwright. `MockStream` is a deterministic
synthetic feed (irrelevant posts, resolved posts, copy-paste duplicates, multi-author bursts) for offline work.

**Analysis** (lexicon-based, no ML — see "Honesty" below): English, Roman Urdu and Urdu script.

```
hazard_strength  = noisy-OR of matched hazard-term weights
urgency          = noisy-OR of injury/child/accident cues
negativity       = noisy-OR of complaint/frustration cues
community_signal = clip( hazard_strength × (0.75 + 0.25 × max(negativity, urgency)) + 0.10 × hazard_strength × [has image] )
status           = resolved if a resolution phrase is not negated within 3 tokens, else active
```

**Geocoding** is offline: explicit coordinates / map links (30 m) or a Karachi neighbourhood gazetteer (centroid + uncertainty
radius; longest alias wins, so "North Nazimabad" ≠ "Nazimabad"). The word "Karachi" alone yields **no** location.
Gazetteer coordinates are approximate and meant to be edited.

**Privacy:** author handles → salted SHA-256 prefix (set `KP_PRIVACY_SALT`); emails, phone numbers and @mentions removed
from stored text; images are not downloaded (URLs only).

**De-duplication:** 3-word-shingle Jaccard ≥ `OSINT_DEDUP_JACCARD` against the last `OSINT_DEDUP_MEMORY` posts. Copies are
stored (provenance) but `duplicate_of` excludes them from fusion.

**Trigger:** an active, located, non-duplicate post with `community_signal ≥ OSINT_ALERT_THRESHOLD` about a
satellite-observable hazard starts a Sentinel-2 check (at most 2 concurrent, cached per ~550 m cell).

## 3. Satellite fusion — `geospatial/satellite.py`

```
NDWI  = (Green − NIR)  / (Green + NIR)     McFeeters 1996      water > 0
MNDWI = (Green − SWIR) / (Green + SWIR)    Xu 2006             urban-robust water
NDVI  = (NIR − Red)    / (NIR + Red)       Tucker 1979         vegetation
```

1. AOI = square window around the report (≥ 500 m half-width, up to 1.5 km when the report only names a neighbourhood).
2. Event scene = clear scene nearest the report (≤ 10 days before, ≤ 3 days after); baseline = median of up to 4 clear scenes from
   `event−120d … event−20d`.
3. Clouds/shadows/cirrus/snow masked with the Scene Classification Layer. < 50 % clear AOI ⇒ `INCONCLUSIVE`.
4. A pixel is **newly wet** iff event NDWI > 0, baseline NDWI ≤ 0, ΔNDWI ≥ 0.10, event NDVI ≤ 0.10, and it is clear in both. Connected
   patches smaller than 8 px are speckle and dropped.
5. `support = 0.6 × clip(newly-wet fraction / 3 %) + 0.4 × clip(largest patch / 150 px)`; `support ≥ 0.35` ⇒ `CONFIRMED_WET`, else `NO_CHANGE`.
6. `quality = clear-sky fraction × baseline-depth factor`.

| Verdict | Meaning | Effect in fusion |
|---|---|---|
| `CONFIRMED_WET` | clear imagery shows new hectare-scale water | adds `support × quality × applicability` |
| `NO_CHANGE` | clear imagery shows none | adds 0; may *reduce* evidence (never veto) |
| `INCONCLUSIVE` | clouds / no scene / provider error | adds 0, no reduction |
| `NOT_APPLICABLE` | hazard smaller than a pixel (pothole, manhole…) | adds 0, no reduction |

Limits: 10 m pixels cannot see narrow street flooding; floods come with clouds (Sentinel-1 SAR would help — not implemented);
revisit takes days; thresholds are expert-set, not calibrated on a labelled Karachi flood. `SyntheticSatelliteProvider` is an
offline simulator (all output flagged synthetic). `PlanetaryComputerProvider` reads only the AOI window of each COG.
**The live provider was not exercised in the development sandbox (no network); run `Satellite Check` with live mode on your machine
and compare against a flood you can verify before trusting it.**

## 4. Evidence fusion — `risk_engine/fusion.py`

```
item strength   s  = raw_confidence × reliability × geo_precision × 0.5^(age / half_life)
per source         = best item of that independent source (one file / one author counts once)
modality conf   c_m = min( cap_m , 1 − Π_sources (1 − s) )                         cap: cv .90 · osint .70 · satellite .80
fused evidence  E  = 1 − Π_m (1 − c_m)
contradiction      if clear satellite NO_CHANGE (quality ≥ .5) and CV+OSINT ≥ .5:  E ← E × (1 − 0.30 × quality × applicability)
impact             = Σ w_k·v_k / Σ w_k  over KNOWN components of {severity, exposure, environment}   (unknown ones dropped, flagged)
likelihood band L  = band(E) ∈ 1..5 ;  impact band I = band(impact) ∈ 1..5 ;  category from L×I: ≥20 Critical · ≥12 High · ≥6 Moderate · else Low
risk_score         = 100 × E × impact
```

Half-lives: CV 14 d, OSINT 36 h, satellite 5 d. Severity = base severity (+ up to 0.15 for urgency cues). Each assessment
stores the per-modality strengths, their leave-one-out marginal contributions, a plain-text explanation generated from the numbers,
flags (`single_modality`, `satellite_inconclusive`, `satellite_contradiction`, `exposure_unknown`, `synthetic_data_included`,
`cv_model_unvalidated`, `stale_evidence`, …), the digest of its inputs and a fingerprint of the configuration that produced it.

Deliberate simplifications: noisy-OR assumes modalities fail independently; `E` is an evidence-strength **index**, not a calibrated
probability; every weight is an expert-set default. The right next step is to calibrate on a labelled set of confirmed/refuted reports.

## 5. Storage and tamper-evidence — `database/db.py`

| Table | Purpose |
|---|---|
| `osint_posts` | processed community reports (redacted text, pseudonymous author, hazard/urgency/signal, location + radius, `content_sha256`, `duplicate_of`, `is_synthetic`). `UNIQUE(platform, post_id)` ⇒ re-scraping is idempotent. Update-blocked by trigger. |
| `satellite_checks` | verdict, support/quality, scene ids, full NDWI/NDVI metrics JSON, `result_sha256`, provider, `is_synthetic`. Update-blocked by trigger. |
| `evidence_ledger` | append-only hash chain: `H(prev_hash \| seq \| ts \| kind \| canonical_json(payload))`; SHA-256, or HMAC-SHA-256 when `KP_LEDGER_KEY` is set. |
| `risk_matrix` | one immutable row per assessment of a hazard cell: strengths, contributions, E, impact, bands, category, score, flags, explanation, `inputs_digest`, `ledger_seq/hash`, algorithm version, config fingerprint. |

Re-assessment appends a new row **only if** the inputs or the matrix cell changed; history is never overwritten.

What the integrity checks give you — and what they do not:

* SQL triggers make casual `UPDATE`/`DELETE` fail loudly.
* `verify_ledger` detects edited and deleted entries even if the triggers were dropped (tested); sequence gaps and broken `prev_hash` links also flag reordering.
* `verify_matrix` ties every row to its ledger entry; `verify_evidence` re-hashes OSINT text and satellite results.
* With `KP_LEDGER_KEY`, someone who rewrites the database file cannot forge a valid chain without the key. Without it, a determined
  attacker who rewrites the whole file *can* rebuild a plain SHA-256 chain: publish `ledger_head()` outside the database (a git
  commit, a public post) to anchor it.
* **This is tamper-evident, not "un-tamperable".** A local SQLite file cannot be both writable by the app and immune to its owner.

## Honesty about what is and isn't validated

* No hazard-trained weights, no real Karachi hazard dataset, no labelled flood events: no accuracy figure is claimed for any component.
* Text analysis is a lexicon. Sarcasm, long-range negation, Roman-Urdu spelling variants and quoted old incidents will fool it.
* The satellite thresholds, fusion weights, caps, half-lives and gazetteer coordinates are unvalidated defaults.
* Everything the offline demo writes is flagged `is_synthetic` and is surfaced as such in the UI and in the risk matrix flags.
* Not an official safety rating; not for emergency decisions.
