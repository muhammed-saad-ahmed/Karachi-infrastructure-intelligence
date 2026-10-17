# Karachi Pulse: Low-Cost AI Urban Risk Intelligence
### A Research Prototype — Status Report and Methodology

**Author:** Muhammad Saad
**Status:** Working engineering prototype (v2) with a demo/baseline vision model, an edge ONNX deployment path, a community-report (OSINT) pipeline, Sentinel-2 cross-verification, a multi-modal evidence-fusion risk matrix, and synthetic demonstration data. No hazard-specific model has been trained and no real Karachi observations have been collected — both are stated plainly wherever they matter, in this document and in the application itself.

---

## 1. Abstract

Karachi, like many resource-constrained megacities, lacks continuous, city-wide sensing of road infrastructure hazards. Municipal inspection is manual, sparse, and reactive. Smartphone cameras are, by contrast, nearly universal. This prototype investigates whether a pipeline combining smartphone imagery, off-the-shelf computer vision, temporal tracking, and a transparent, uncertainty-aware risk-scoring engine could plausibly support (not replace) municipal infrastructure monitoring. The system — object detection and tracking, geospatial clustering, an explainable risk engine, and an evaluation framework — is built and functional. What is **not** yet done, and is not claimed to be done, is a hazard-specific trained model or any real Karachi dataset; the prototype runs in an explicitly labelled demo/baseline mode until those exist.

## 2. Introduction

Urban infrastructure failure in Karachi is not abstract. As of December 2025, Dawn newspaper reported that 23 people — including 8 children — had died that year alone after falling into open manholes and uncovered sewers in the city [1], a recurring story across multiple months and neighbourhoods [2], [3]. Separately, waterlogging from clogged or encroached drains is a documented, worsening problem in specific Karachi localities such as Manzoor Colony and Orangi Nala, studied directly by local urban-planning groups [4]. These are real, sourced, ongoing problems — not a premise invented for this project.

## 3. Problem Statement

Karachi's Municipal Corporation relies on manual inspection and citizen complaints to find road and drainage hazards. This is slow, inconsistent, and reactive: by the time a hazard is reported through official channels, it has often already caused harm, as the manhole incidents above illustrate. A complementary, low-cost sensing layer — even an imperfect one — could plausibly shorten that detection gap, provided its outputs are honest about their own uncertainty.

## 4. Related Work

**Pothole and road-damage detection.** A substantial body of computer-vision research addresses pothole/road-damage detection from 2D images, 3D point clouds, and hybrid methods; a 2024 systematic review in *Sensors* surveys this landscape and finds hybrid (image-processing + deep-learning) approaches currently report the strongest accuracy [5]. The RDD2020 dataset — 26,336 smartphone-captured road images from India, Japan, and the Czech Republic, annotated for four damage types including potholes — is a widely used public benchmark for this task and the basis of the IEEE Global Road Damage Detection Challenge [6]. A 2024 comparative study specifically evaluated lightweight YOLO variants (YOLOv8-nano, YOLOv8-small, YOLOv7-tiny) for on-device smartphone pothole detection, reporting average precision in the low-to-mid 80% range for the smallest models — evidence that a nano-scale detector is a reasonable starting architecture for this problem class, though not evidence about performance on Karachi-specific imagery [7].

**Crowdsourced flood risk.** Separately from pothole detection, researchers have used crowdsourced and sensor-fused data for flood-risk prediction: a 2023 study combined crowdsourced traffic reports with topographic and rainfall features to predict road-level flooding risk during real storm events in Texas using tree-based machine learning models [8]. Other work has used crowdsourced street photographs of submerged traffic signs to estimate flood depth directly from images [9]. MIT's Urban Risk Lab operates RiskMap, a live, multi-city citizen-reporting platform for flood and infrastructure hazards that has been used operationally during real storm events [10] — a direct, real-world precedent for the citizen-sensing concept this prototype explores.

**Karachi- and Pakistan-specific work.** This is not the first Karachi-focused effort. "Raasta," a Habib University final-year project, built an Android application collecting accelerometer, gyroscope, and GPS data specifically on Karachi roads to classify road anomalies including potholes, displayed on a map [11]. The Khyber Pakhtunkhwa provincial government's Planning and Development Department has separately worked on a smartphone application for citizen-driven pothole identification [12]. Locally, the Nala Mapping Project by TTRC and the Urban Innovation Lab has directly surveyed flood-prone drainage infrastructure in specific Karachi neighbourhoods [4].

## 5. Research Gap

Karachi Pulse does not claim to be the first system to look at road hazards or flooding in Pakistan — the work above shows it isn't. The gap it targets is the specific **combination**: multi-hazard visual detection (not only potholes) **plus** temporal deduplication across frames/reports **plus** explicit multi-source evidence fusion **plus** a transparent, component-level risk score **plus** rainfall-scenario modelling for waterlogging, all in one pipeline with a documented evaluation framework. Prior Karachi-specific work (e.g. Raasta [11]) focuses on sensor-based road-anomaly classification; this prototype instead focuses on the vision + risk-reasoning + evidence-fusion layer, intended as a complementary approach rather than a replacement.

## 6. Research Question

Can low-cost smartphone imagery, combined with environmental, temporal, traffic, and geospatial data, be used to identify, verify, prioritize, and predict urban infrastructure risks in a resource-constrained megacity — and can this be done with an explicit, auditable accounting of uncertainty rather than a black-box score?

## 7. System Architecture

CAMERA → COMPUTER VISION (detection/classification) → TEMPORAL TRACKING (IoU-based, merges repeated frames/reports into single incidents) → GEOLOCATION (haversine clustering) → EVIDENCE FUSION (independent-source confidence, not frame count) → RISK ENGINE (configurable weighted components with stated reasons) → URBAN INTELLIGENCE (map, dossier, road-health index, flood/traffic labs).

Implementation: Python, OpenCV, ONNX Runtime (edge inference; Ultralytics only for training/export), Streamlit, Plotly, SQLite, pytest. Full module layout in the README.

**v2 extension - multi-modal evidence.** Camera detections are no longer the only evidence stream. Public community reports (English / Roman Urdu / Urdu) are collected asynchronously and politely, analysed with a transparent lexicon, geocoded offline and de-duplicated (`scripts/osint_pipeline.py`). Reports of area-scale waterlogging trigger a Sentinel-2 check (NDWI/NDVI change against a dry-period baseline, cloud-masked; `geospatial/satellite.py`). The three streams are fused by `risk_engine/fusion.py` (capped, time-decayed noisy-OR over independent sources; explicit contradiction rule; 5x5 likelihood-by-impact matrix) and recorded in append-only SQLite tables with a hash-chained ledger. Specification: `docs/multimodal_fusion.md`.

## 8. Dataset

**Current state: none, real.** `data/demo/` contains algorithmically generated synthetic points inside Karachi's bounding box, explicitly flagged `is_synthetic=True` in both the data and every place it is displayed. No image in this repository is claimed to be a real Karachi photograph unless a user uploads one in a live session, in which case it is processed and discarded, not stored as a dataset.

**What real data collection would require:** geotagged photographs/short clips of Karachi roads, ideally with a geographic (not random) train/validation/test split to avoid leakage between nearby frames of the same physical location — the dataset tooling for this split already exists in `scripts/` and is unit-tested, but has no real data to operate on yet.

## 9. Computer Vision Pipeline

Image: decode → validate → optional face blur → detect → confidence filter → status (`"Insufficient visual evidence"` rather than a forced label when nothing clears threshold). Video: frame sampling → per-frame detection → IoU tracking → one incident per track, not per frame. Detector is a registry (`computer_vision/detection.py`): a custom model in `models/custom/*.pt` is always preferred if present; otherwise a selectable COCO baseline (`yolov8n`/`yolov8s`) runs in explicitly labelled DEMO/BASELINE mode, which can recognize general objects (people, vehicles) but has no hazard-specific classes (pothole, waterlogging, etc.).

## 10. Risk Engine

`risk_engine/scoring.py` computes a 0–100 score from five configurable, weighted components (severity, exposure, recurrence, environmental context, multi-source confirmation), each traceable back to its contributing points. Evidence confidence is computed as a noisy-OR over *independent sources only* — repeated frames or reports from the same source do not inflate it, which is verified by a dedicated unit test. AI confidence, evidence confidence, and risk score are kept as three distinct, separately displayed quantities, never conflated.

## 11. Geospatial Analysis

Haversine-distance clustering (`geospatial/clustering.py`) merges same-type observations within a configurable radius into one incident, with the merge reason logged per member. Coordinate validation rejects out-of-range or out-of-bounds points before they reach the map.

## 12. Experimental Design

No hazard-detection experiment has been run, because no hazard-labelled dataset exists yet (Section 8). The evaluation framework (Section 13) is built and unit-tested so that the moment real labelled data exists, Precision/Recall/F1/IoU-matching can be computed immediately without further engineering.

## 13. Evaluation Metrics

`evaluation/metrics.py` implements IoU-based detection matching, Precision/Recall/F1, and MAE/RMSE for numeric predictions — all unit-tested against synthetic ground truth (see `tests/test_core.py`). The in-app Model Lab page lets a user upload their own labelled images plus a ground-truth JSON and get real, live-computed Precision/Recall/F1 against the active detector. `evaluate_with_conditions` additionally buckets results by condition tag (rain, low light, blur, occlusion, unusual angle, night, heavy traffic, or custom tags) so performance under specific failure-prone conditions is measured separately rather than averaged away — this is the error-analysis tool described in Section 15. Both functions only produce numbers when real labelled data is supplied by the user; with none supplied, the page honestly shows NOT EVALUATED. mAP, a confusion matrix, and a full per-class breakdown are **not implemented** in this pass; they require a larger labelled set and a full confidence/IoU sweep than this lightweight tool currently performs.

## 14. Results

**None yet, and none are claimed.** The application correctly displays "NOT EVALUATED" for every metric that has no real trained model or labelled dataset behind it. This is a deliberate design decision, not an oversight — see Section 17 of the original project brief, which this report follows.

## 15. Error Analysis

The tooling is built and unit-tested (`evaluate_with_conditions`, Section 13): given condition-tagged ground truth, it computes real, separately-measured Precision/Recall/F1 per condition (lighting, occlusion, rain, blur, angle, etc.), surfaced as a sorted bar chart in Model Lab so the weakest conditions are immediately visible. What has **not** happened is running it against a genuine labelled Karachi test set — that requires the real dataset described in Section 8, which does not yet exist. The distinction matters: the capability is real and ready; the result is not yet produced, and this document does not claim otherwise.

## 16. Limitations

- No hazard-trained model; baseline detector is general-purpose COCO only.
- No real Karachi observations; all map/incident data is synthetic and labelled as such.
- Risk-engine and traffic-density weights are expert-set defaults, not calibrated against measured outcomes.
- Face-blur privacy step is best-effort Haar-cascade detection; license plates are not reliably anonymized.
- Flood and traffic modules are scenario/estimation tools, not validated forecasts.
- Evaluation tooling is real but has only been exercised against synthetic/unit-test data, not a genuine labelled Karachi set.
- (v2) Community-report analysis is a lexicon, not a trained model; the gazetteer is approximate; satellite thresholds and fusion weights are expert-set and uncalibrated; the evidence-strength index is not a calibrated probability.
- (v2) 10 m Sentinel-2 pixels cannot observe potholes, manholes or narrow street flooding, and clouds frequently block imagery during floods; these cases yield NOT_APPLICABLE / INCONCLUSIVE rather than negative evidence.
- (v2) The risk matrix is tamper-evident (hash chain, append-only triggers, optional HMAC key), not tamper-proof.

## 17. Ethical Considerations

No face recognition, person identification, or license-plate identification is implemented or planned. An optional face-blur step runs before any detection. Synthetic data is never presented as real anywhere in the application. Risk scores are explicitly labelled as experimental and not an official safety rating, to avoid the output being mistaken for a municipal or government determination.

## 18. Future Work

1. Collect a small (hundreds, not thousands) real, geotagged Karachi road-hazard image set with consent and clear provenance.
2. Fine-tune a YOLO model on that set; drop the resulting weights into `models/custom/` — the application already auto-detects and prioritizes it with no code changes needed.
3. Run the existing (already-built, already-tested) evaluation pipeline against a held-out, geographically-split test set.
4. Calibrate traffic-density and flood-rainfall thresholds against any obtainable ground truth (even a small manual count).
5. Extend the evidence-fusion model with real multi-user reports once the system has actual users.
6. Calibrate the fusion weights, caps and half-lives, and the satellite thresholds, against a labelled set of confirmed/refuted reports and at least one verifiable Karachi flood event.
7. Validate the live Sentinel-2 provider end to end; add Sentinel-1 SAR (sees through cloud) for flood verification.
8. Train the hazard detector (pothole / open manhole / sewage overflow) on a consented, geotagged Karachi dataset, quantize it to INT8 with real calibration images, and report its measured accuracy.
9. Replace the lexicon with a multilingual classifier only once a labelled Karachi text set exists to validate it.

## 19. Conclusion

Karachi Pulse is an honestly-scoped engineering prototype: a complete, tested pipeline for detection (with an edge ONNX deployment path), tracking, geospatial clustering, community-report ingestion, satellite cross-verification, multi-modal evidence fusion, transparent risk scoring, and evaluation — built around real, documented problems in Karachi — that currently runs in demo mode because the two inputs that would make it "real" (a trained hazard model and real local data) require data-collection work this document does not pretend has happened.

## References

1. "The open manhole that laid bare Karachi's systemic rot," *Dawn*, Dec. 10, 2025. https://asianews.network/the-open-manhole-that-laid-bare-karachis-systemic-rot/
2. "Eight-year-old boy dies after falling into open manhole in Karachi's Korangi," *Dawn*, Dec. 29, 2025. https://english.aaj.tv/news/amp/330450124
3. "Open manhole claims another young life in Karachi," *The News*, Jan. 5, 2025. https://www.thenews.pk/latest/1268989-open-manhole-claims-another-young-life-in-karachi
4. "Nala Mapping," Transformation and Trauma Response Collective (TTRC) / Urban Innovation Lab. https://resdev.org/nala-mapping/
5. Y. Safyari, M. Mahdianpari, and H. Shiri, "A Review of Vision-Based Pothole Detection Methods Using Computer Vision and Machine Learning," *Sensors*, vol. 24, no. 17, p. 5652, 2024. https://doi.org/10.3390/s24175652
6. D. Arya, H. Maeda, S. K. Ghosh, D. Toshniwal, and Y. Sekimoto, "RDD2020: An annotated image dataset for automatic road damage detection using deep learning," *Data in Brief*, vol. 36, p. 107133, 2021. https://pmc.ncbi.nlm.nih.gov/articles/PMC8166755
7. A. U. Amri and G. P. Kusuma, "Comparative study of pothole detection using deep learning on smartphone," *IAES Int. J. Electr. Comput. Eng.*, vol. 37, no. 2, pp. 995–1004, 2025. https://doi.org/10.11591/ijeecs.v37.i2.pp995-1004
8. F. Yuan, C.-C. Lee, W. Mobley, H. Farahmand, Y. Xu, R. Blessing, S. Dong, A. Mostafavi, and S. D. Brody, "Predicting road flooding risk with crowdsourced reports and fine-grained traffic data," *Computational Urban Science*, vol. 3, no. 1, p. 15, 2023. https://doi.org/10.1007/s43762-023-00082-1
9. "Crowdsourced-based Deep Convolutional Networks for Urban Flood Depth Mapping," arXiv:2209.09200, 2022. https://arxiv.org/pdf/2209.09200
10. MIT Urban Risk Lab, "RiskMap." https://urbanrisklab.org/riskmap/
11. A. Khan and Z. O. Karim, "Raasta" (Final Year Project), Habib University. https://habib.edu.pk/graduate-directory/graduate/abeer-khan/
12. "Pothole Fixer – A New App For Fixing Broken Roads In KPK," Zameen.com, Jan. 29, 2021. https://www.zameen.com/blog/pothole-fixer-smartphone-application.html

*All sources above were retrieved and verified at the time of writing. No citation in this document was invented; where a claim could not be backed by a verifiable source, it is marked NOT EVALUATED or left as future work instead.*
