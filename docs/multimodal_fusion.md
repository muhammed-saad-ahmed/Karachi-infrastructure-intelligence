# Multi-modal pipeline (v2): edge CV, OSINT, Sentinel-2, evidence fusion

This document is the technical specification for the four v2 modules (sections 1-5) and the v2.1 hardening (section 6). Every constant mentioned lives in
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

## 6. v2.1 hardening (five refactors of the modules above)

### 6.1 Satellite: spectral unmixing + shadow mask (`geospatial/satellite.py`)
* **LSMA.** Each pixel's four bands (green, red, NIR, SWIR) are modelled as a mix of *water, vegetation, impervious, shade* endmembers (`config.SAT_ENDMEMBERS`). Abundances are solved by **fully constrained least squares** (non-negative, sum to 1): for each of the 2^m-1 candidate supports, the sum-to-one equality-constrained solution is found with a Lagrange multiplier; non-negative solutions are kept and the smallest residual wins. It is exact (matches SLSQP in `tests/test_v3.py`) and vectorised. A pixel is *newly wet* when its water fraction is >= `SAT_LSMA_MIN_WATER` and rose by >= `SAT_LSMA_MIN_DELTA` over the dry-period median. A poor spectral fit (`SAT_LSMA_MAX_RMSE`) marks the pixel unreliable instead of guessing. `KP_SAT_METHOD=ndwi` restores the v2.0 behaviour.
* **Shadow mask.** Building/cloud shadow and water are both dark and a shadowed concrete pixel unmixes to roughly 86% "water", so unmixing alone does not fix the false positive. A pixel is flagged structural shadow only if it is dark **and** thin (not inside a dark body >= `SAT_SHADOW_MIN_BODY_PX` wide, via morphological opening) **and** has high local variance **and** has a bright neighbour (roof/concrete) within that distance. SCL classes 2/3 are reported separately. Reported per check: `shadow_frac_event`, `structural_shadow_frac_event`, `shadow_false_positive_px`, `naive_ndwi_gain_frac` vs `water_equiv_gain_frac`.
* **What was shown:** on the offline simulator with dense simulated buildings the legacy method gave support ~0.5-0.58 (false `CONFIRMED_WET`); with the filter it is 0, and a real simulated flood next to shadows still gives its true clusters.
* **Not shown:** performance on real Sentinel-2 scenes. The endmembers are generic textbook-style spectra, not scene-derived (use image-derived pure pixels for a real deployment), the shadow thresholds were tuned on **my own simulator** (circular), and a flooded narrow street between tall buildings can be masked as shadow (a false negative).

### 6.2 Monte-Carlo sensitivity and calibration (`evaluation/simulation.py`)
`python -m evaluation.simulation --synthetic 60 --n-iter 10000 --out reports/sensitivity.md --anchor` (or the **Sensitivity** page).
* The whole fusion + 5x5 matrix is re-implemented as dense NumPy arrays (cells x sources x items x draws) and cross-checked against `fusion.score_cell` (E within 5e-5, identical categories, with and without rain). 10,000 draws over 60 cells take about 4 s on a laptop CPU.
* Every expert parameter (reliabilities, half-lives, caps, severity table, urgency boost, contradiction rule...) is perturbed: logit-normal for probabilities, log-normal for scales, additive logit for reliability shifts. Output per cell: mean and **95% interval** of E, impact and risk score, category probabilities, P(same category as baseline), and the margin to the nearest band edge.
* **Stability margin / variance tolerance:** a ladder of perturbation sizes gives the largest relative sd at which the average cell keeps its category with probability >= 95% (`tolerance_sigma`; flagged `below_ladder` if even the smallest rung fails).
* **Leave-one-out matrix** (CV vs OSINT vs satellite) with common random numbers, plus a severe-shock variant and a ranking of which stream stabilises the matrix. It measures *dependence*, not correctness.
* **"MCMC":** the sampling above is plain Monte Carlo *sensitivity analysis*, not a posterior. A true Bayesian step exists separately: `calibrate_reliabilities(comp, labels)` runs 4-chain random-walk Metropolis (adaptive burn-in, split R-hat, FFT effective sample size) for how reliable CV and OSINT evidence is, given field-checked 0/1 labels. It treats E as a Bernoulli probability, which the fused score is **not** known to be, and needs a few hundred real labels; with synthetic labels it demonstrates the machinery only. Read `converged` before trusting it.
* The synthetic result (category stability ~79% at sd 15%, tolerance ~0.03) says the matrix is fragile near band edges on **synthetic** cells. Do not quote it as a property of Karachi.
* `anchor_report` appends a digest of the report to the hash-chained ledger (kind `sensitivity_report`).

### 6.3 Semantic text analysis (`scripts/osint_text.py`)
* A post is split into clauses and candidate 1-4 word spans. Each span is embedded and compared by **cosine similarity** with banks of seed phrases (hazards, urgency, negativity, resolved, praise); the nearest seed's similarity is mapped to 0..1 and scaled by that seed's prior. Add seeds without code: `data/hazard_seeds.json` or `SeedBank.add`.
* **Two embedders, same interface.** `OnnxSentenceEmbedder` runs a multilingual MiniLM / MobileBERT sentence encoder on ONNX Runtime/CPU (mean pooling, L2 norm). `HashedNgramEmbedder` is the dependency-free fallback. **The model is not bundled** (licence, size, and it could not be downloaded in the build environment), so the fallback is what runs out of the box. The fallback is **orthographic, not semantic**: it handles spelling variants (khadda/khada/khara, gaddha/gadha/gaddhay, paani/pani) but not synonyms or true cross-lingual meaning. Every result carries `backend`.
* **Structural parser** (closed-class words, not a hazard dictionary): negation (English pre-negators, Roman-Urdu/Urdu post-negators, comma-bounded clauses, handles seeds that contain a negator such as "manhole ka dhakkan nahi" = cover missing), intensity multipliers (bohat, very, thora, slight; bounded to 0.6-1.3), and **sarcasm** (praise in the same sentence as an un-negated hazard, or in an adjacent sentence marked by `!!`/`?!`; thanks-for-fixing is *resolved*, not sarcasm; contrast words like "but" end a sentence). Sarcasm halves urgency and floors negativity at 0.6. Ambiguous cases (denied hazard + praise/irony) set `needs_review`.
* **Install the real encoder (once, on any machine with internet; free Colab is fine):**
  ```bash
  pip install "optimum[exporters]" tokenizers
  optimum-cli export onnx --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 models/text/
  ```
  Then `pip install -r requirements-text.txt`. The app loads `models/text/model.onnx` + `tokenizer.json` automatically (override with `KP_TEXT_MODEL_DIR`); if the model is missing or fails it falls back and logs a warning. With the ONNX backend the cosine-to-score band (`SIM_BAND`) **must be re-checked** on a handful of your own posts: it is set from general knowledge of MiniLM similarity ranges, not measured here.
* **Validated:** 30 hand-written examples (written by me, who also wrote the rules) behave as intended, and `tests/test_v3.py` covers variants, negation, intensity, sarcasm, the ONNX path with a stub encoder, and graceful fallback. **Not validated:** any labelled Karachi corpus. Long-range negation (> 3 words), code-switched irony and quoted old incidents still fool it.

### 6.4 INT8 calibration pipeline (`scripts/dataset_formatter.py`, `computer_vision/edge_onnx.py`)
```bash
python -m scripts.dataset_formatter build --src data/raw --out data/calib --n 50
python -m scripts.dataset_formatter int8  --model models/custom/hazard_n.onnx --src data/raw --work reports/int8
```
* **Augmentation** (seeded, NumPy/OpenCV): random zoom/crop, flip, +-5 deg rotation, then one Karachi lighting profile - `harsh_noon`, `golden_hour`, `dusk`, `night_sodium` (lamp glow + noise), `monsoon_overcast`, `dust_haze` (atmospheric scattering I*t + A*(1-t)), `headlight_glare`, `flyover_shade` - plus sensor noise and a JPEG round trip. With no source images it draws procedural road scenes and marks the set `synthetic`.
* **Calibration set:** exactly `EDGE_CALIBRATION_MIN_IMAGES` (50) JPEGs + `manifest.json` with every recipe and a hash. It is `deployment_grade` only if built from >= `EDGE_CALIB_MIN_UNIQUE_SOURCES` (20) distinct **real** photos. 50 augmentations of 3 photos are not 50 photos.
* **INT8 vs FP32 agreement on the raw output tensors** (before NMS can hide error): cosine, relative L2 (nrmse), SQNR (dB), max abs diff, and top-k anchor Jaccard, mean and **worst** image; plus detection agreement (`compare_models`) and interleaved latency (p50/p95, speed-up).
* **Gates** (`evaluate_int8`, `assert_int8_ok`, thresholds `EDGE_INT8_*` in config) are asserted by `tests/test_int8_pipeline.py`. INT8 slower than FP32 is a *warning* by default (many CPUs lack fast INT8), a failure with `EDGE_INT8_REQUIRE_SPEEDUP`.
* **Not run here:** the real `quantize_int8_static` needs the `onnx` package, which the build sandbox did not have. The pipeline's metrics and gates were tested against a stand-in (the FP32 net with 8-bit-rounded weights); the end-to-end quantization test (`test_real_static_quantization_end_to_end_when_onnx_installed`) runs on your machine once `onnx` is installed (`pip install -r requirements-dev.txt`). Agreement with FP32 is not accuracy: evaluate the final model on labelled data.

### 6.5 Local differential privacy on coordinates (`geospatial/gazetteer.py`, `database/db.py`)
* **Mechanism:** planar-Laplace noise (Andres et al., *geo-indistinguishability*, 2013), sampled with the Lambert W_{-1} inverse CDF. Parameters: level l within radius r, so epsilon = l / r per metre. Defaults l = ln 4, r = 100 m: epsilon ~ 0.0139/m, **mean displacement 144 m, 95% within 342 m, hard cap 666 m** (the 0.999 quantile, which makes the output (epsilon, delta)-geo-indistinguishable). Results are remapped into the Karachi bounding box.
* **Where it is applied:** once, at ingestion, to *precise* locations only (`explicit` coordinates in text/link or `source`-provided). Neighbourhood gazetteer centroids already carry their own uncertainty and are left alone. `database/db.py` enforces it again at the storage boundary (idempotent: `geo_method` becomes `explicit+ldp`), so no code path can store an exact coordinate. Coordinates are also scrubbed from the stored text and URL (the pipeline geocodes from the raw text first). **Schema unchanged.**
* **Clustering stays usable, but it is not perfect.** The recorded uncertainty radius grows to `hypot(original, sqrt(2) * r95)` (two independently noised reports of one spot differ by ~sqrt 2 x the single-report noise), so fusion still groups neighbours, at the price of a weaker evidence weight for them. Several reports of one hot-spot recover its centroid to well under 120 m (tested); one report is only locatable to a few hundred metres. **Perfect clustering under local differential privacy is mathematically impossible**; this is a stated trade-off, tunable with `KP_LDP_LEVEL` / `KP_LDP_RADIUS_M`.
* **Repeat reporters:** independent noise on repeated reports of the same spot would let an observer average it away (privacy loss composes k x epsilon). The noise is therefore *sticky*: seeded by HMAC(`KP_PRIVACY_SALT`, author hash + ~55 m grid cell), so the same author at the same spot always gets the same displacement. Anonymous reports get fresh noise. Different spots by one author still compose.
* **Scope of the guarantee:** it protects the stored coordinate. It does not protect against someone who can identify the author from the text itself, from timing, or from a distinctive photo, and it is only as strong as the salt is secret. Synthetic demo posts are exempt unless `KP_LDP_SYNTHETIC=1`. Set `KP_LDP=0` to disable.

## Honesty about what is and isn't validated

* No hazard-trained weights, no real Karachi hazard dataset, no labelled flood events: no accuracy figure is claimed for any component.
* Text analysis (v2.1) is embedding-based but runs on an orthographic fallback unless you install the ONNX encoder; sarcasm, long-range negation and quoted old incidents still fool it, and its thresholds were tuned on hand-written examples.
* LSMA endmembers, shadow thresholds and every simulation parameter are expert-set or tuned on my own simulator; none has been checked against labelled real Sentinel-2 scenes or field-verified Karachi events.
* The sensitivity analysis shows how much conclusions depend on unvalidated parameters, not that they are right.
* The satellite thresholds, fusion weights, caps, half-lives and gazetteer coordinates are unvalidated defaults.
* Everything the offline demo writes is flagged `is_synthetic` and is surfaced as such in the UI and in the risk matrix flags.
* Not an official safety rating; not for emergency decisions.

