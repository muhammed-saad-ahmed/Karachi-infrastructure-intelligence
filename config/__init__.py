"""All tunable parameters live here (no weights hard-coded elsewhere).

Sections
--------
1. Core / risk engine            (original prototype)
2. Edge ML runtime (ONNX)        computer_vision/edge_onnx.py
3. OSINT scraping pipeline       scripts/osint_pipeline.py
4. Satellite fusion (Sentinel-2) geospatial/satellite.py
5. Multi-modal evidence fusion   risk_engine/fusion.py

Every number below is an *expert-set default*, not a calibrated value. They are kept here so a
reviewer can see - and change - every assumption in one place.
"""
import os


def _load_dotenv(path=None):
    """Minimal .env reader (stdlib only, no python-dotenv dependency). Real environment variables win; empty
    values are ignored so `KP_PRIVACY_SALT=` in .env.example does not blank out a default."""
    path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)
    except OSError:
        pass


_load_dotenv()

# ============================================================== 1. core / risk engine
DB_PATH = os.getenv("KP_DB_PATH", "data/karachi_pulse.db")
PROCESSING_VERSION = "2.0.0"
KARACHI_BBOX = (24.70, 66.80, 25.60, 67.60)  # lat_min, lon_min, lat_max, lon_max
MAX_IMAGE_MB, MAX_VIDEO_MB, MAX_FILES, MAX_VIDEO_SECONDS = 10, 100, 10, 60
DUP_RADIUS_M = 25.0
MIN_CONFIDENCE = 0.35
RISK_WEIGHTS = {"severity": 0.35, "exposure": 0.20, "recurrence": 0.15,
                "environment": 0.15, "confirmation": 0.15}
RISK_LEVELS = [(85, "Critical"), (60, "High"), (30, "Moderate"), (0, "Low")]
BASE_SEVERITY = {"pothole": 0.7, "road_damage": 0.6, "waterlogging": 0.65, "garbage": 0.45,
                 "open_drain": 0.8, "sign_damage": 0.4, "streetlight": 0.35, "hazard": 0.5,
                 # added in v2: the two hazards the edge model is trained to find
                 "open_manhole": 0.9, "sewage_overflow": 0.75}
RAIN_SENSITIVE = {"waterlogging": 1.0, "open_drain": 0.8, "garbage": 0.5, "pothole": 0.4,
                  "sewage_overflow": 0.9, "open_manhole": 0.5}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "train"}
# Vehicles-per-analyzed-frame bands — experimental and NOT calibrated against real traffic counts.
TRAFFIC_DENSITY_THRESHOLDS = {"Low": (0, 4), "Moderate": (5, 10), "High": (11, 20), "Severe": (21, float("inf"))}
BENCHMARK_RUNS = 10
ROAD_HEALTH_WEIGHTS = {"surface": 0.30, "drainage": 0.20, "cleanliness": 0.15, "lighting": 0.10,
                       "traffic_exposure": 0.10, "recurrence": 0.10, "safety_evidence": 0.05}

# Detector label -> hazard type used by the risk engine and the fusion layer.
# Labels that are not listed here (e.g. COCO "car", "person") are NOT hazards and are ignored by fusion.
HAZARD_LABEL_MAP = {
    "pothole": "pothole", "potholes": "pothole",
    "open_manhole": "open_manhole", "manhole": "open_manhole", "missing_manhole_cover": "open_manhole",
    "sewage": "sewage_overflow", "sewage_overflow": "sewage_overflow", "sewer_overflow": "sewage_overflow",
    "waterlogging": "waterlogging", "flood": "waterlogging", "standing_water": "waterlogging",
    "garbage": "garbage", "trash": "garbage",
    "open_drain": "open_drain", "road_damage": "road_damage", "crack": "road_damage",
}

# ============================================================== 2. edge ML runtime (ONNX)
EDGE_MODEL_DIR = "models/custom"          # *.onnx here are auto-detected before *.pt (see detection.py)
EDGE_IMGSZ = 640                          # export/inference size; 416 is ~2.3x faster, lower small-object recall
EDGE_SCORE_THRESHOLD = 0.25               # pre-NMS candidate filter (display threshold is MIN_CONFIDENCE)
EDGE_NMS_IOU = 0.45
EDGE_MAX_CANDIDATES = 300                 # cap boxes entering NMS -> bounded CPU cost even on noisy frames
EDGE_MAX_DETECTIONS = 100
EDGE_NUM_THREADS = int(os.getenv("KP_EDGE_THREADS", "0"))  # 0 = auto (min(4, cpu_count))
EDGE_HAZARD_CLASSES = ("pothole", "open_manhole", "sewage_overflow")  # fallback names if ONNX metadata has none
EDGE_CALIBRATION_MIN_IMAGES = 50          # static INT8 quantization refuses to run on fewer real images

# ============================================================== 3. OSINT scraping pipeline
OSINT_USER_AGENT = ("KarachiPulseResearchBot/2.0 (+https://github.com/muhammed-saad-ahmed/"
                    "Karachi-infrastructure-intelligence; independent research prototype)")
OSINT_MIN_DELAY_S = 2.0                   # minimum gap between two requests to the SAME host
OSINT_MAX_CONCURRENCY = 4                 # global cap on in-flight requests
OSINT_REQUEST_TIMEOUT_S = 15.0
OSINT_MAX_RETRIES = 3
OSINT_BACKOFF_BASE_S = 1.0
OSINT_BACKOFF_MAX_S = 30.0
OSINT_MAX_BODY_BYTES = 2 * 1024 * 1024
OSINT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
OSINT_ALERT_THRESHOLD = 0.45              # community_signal needed to trigger a satellite check
OSINT_DEDUP_JACCARD = 0.90                # near-duplicate text threshold (counts once in fusion)
OSINT_DEDUP_MEMORY = 500                  # recent posts kept for near-duplicate comparison
# Salt for hashing author handles. CHANGE THIS in .env (KP_PRIVACY_SALT); raw handles are never stored.
OSINT_PRIVACY_SALT = os.getenv("KP_PRIVACY_SALT", "change-me-in-dotenv")

# ============================================================== 4. satellite fusion (Sentinel-2)
SAT_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAT_COLLECTION = "sentinel-2-l2a"
SAT_AOI_RADIUS_M = 500.0                  # half-width of the square area of interest (1 km box ~ 100x100 px)
SAT_AOI_MAX_RADIUS_M = 1500.0             # cap when a report is only localised to a neighbourhood (3 km box, 300x300 px)
SAT_PIXEL_M = 10.0                        # Sentinel-2 B03/B04/B08 native resolution
SAT_EVENT_LOOKBACK_DAYS = 10              # scenes up to N days BEFORE the report can verify it
SAT_EVENT_LOOKAHEAD_DAYS = 3              # ... and up to N days after (used when a check is re-run later)
SAT_BASELINE_DAYS = (120, 20)             # baseline composite window: event-120d .. event-20d
SAT_BASELINE_MAX_SCENES = 4
SAT_MAX_CLOUD_PCT = 40.0                  # scene-level eo:cloud_cover filter (pixel-level mask applied after)
SAT_MIN_VALID_FRACTION = 0.50             # clear-sky pixels required inside the AOI, else INCONCLUSIVE
SAT_NDWI_THRESHOLD = 0.0                  # McFeeters (1996): NDWI > 0 -> open water
SAT_MIN_DELTA_NDWI = 0.10                 # a pixel only counts as "newly wet" if NDWI rose by >= this
SAT_MAX_NDVI_WATER = 0.10                 # ...and only if it does not look vegetated (NDVI <= this) in the event scene
SAT_GAIN_FULL_FRAC = 0.03                 # 3% of the AOI newly wet (~3 ha of a 100 ha box) -> full area score (expert-set)
SAT_CLUSTER_FULL_PX = 150                 # one connected newly-wet patch of 150 px (1.5 ha) -> full patch score
SAT_MIN_CLUSTER_PX = 8                    # patches smaller than this are treated as speckle
SAT_CONFIRM_THRESHOLD = 0.35              # support >= this -> verdict CONFIRMED_WET
SAT_CACHE_TTL_HOURS = 24.0                # reuse a conclusive check for the same ~500 m cell for this long
SAT_INCONCLUSIVE_RETRY_HOURS = 6.0        # ...and an inconclusive one for this long (be kind to the free API)
SAT_CELL_DEG = 0.005                      # ~550 m grid used as the cache key
# How much a 10 m optical satellite can say about each hazard. 0 = not observable from orbit
# (a pothole or an open manhole is far smaller than a pixel) -> verdict NOT_APPLICABLE, never "negative".
SAT_APPLICABILITY = {"waterlogging": 1.0, "sewage_overflow": 0.5, "open_drain": 0.5,
                     "pothole": 0.0, "open_manhole": 0.0, "garbage": 0.0, "road_damage": 0.0,
                     "sign_damage": 0.0, "streetlight": 0.0, "hazard": 0.0}

# ============================================================== 5. multi-modal evidence fusion
FUSION_ALGORITHM_VERSION = "1.0.0"
# Max trust any single modality may contribute to the noisy-OR (so no sensor can reach 100% alone).
FUSION_MODALITY_CAP = {"cv": 0.90, "osint": 0.70, "satellite": 0.80}
# Evidence older than this loses half its weight (exponential decay), in hours.
FUSION_HALF_LIFE_H = {"cv": 24.0 * 14, "osint": 36.0, "satellite": 24.0 * 5}
FUSION_MATCH_RADIUS_M = 150.0             # evidence within this distance (same hazard type) joins one cell
FUSION_CV_UNVALIDATED_RELIABILITY = 0.60  # CV reliability until a measured F1 is logged for that model
FUSION_OSINT_SOURCE_RELIABILITY = {"default": 0.60, "mock": 0.60}
FUSION_OSINT_MAX_AUTHORS = 10             # independent authors counted per cell (diminishing returns anyway)
# OSINT geocode precision factor by the radius of uncertainty of the location (metres -> factor).
FUSION_GEO_PRECISION = ((100.0, 1.00), (1000.0, 0.60), (2000.0, 0.35), (float("inf"), 0.20))
FUSION_URGENCY_SEVERITY_BOOST = 0.15      # max severity increase from urgency cues (injury, child, ...)
FUSION_CONTRADICTION_LAMBDA = 0.30        # evidence reduction when clear satellite imagery contradicts the reports
FUSION_CONTRADICTION_MIN_QUALITY = 0.50
FUSION_CONTRADICTION_MIN_OTHER_EVIDENCE = 0.50
FUSION_AGREE_MIN_CONTRIB = 0.20           # a modality "agrees" if its contribution is at least this
FUSION_MAX_AGE_DAYS = 90                  # evidence older than this is ignored entirely
FUSION_REASSESS_EPS = 0.02                # do not append a new ledger entry unless evidence moved by > this
# Impact = weighted mean of these components, using the weights in RISK_WEIGHTS and ONLY the components
# that are actually known (unknown ones are excluded and weights renormalised, never silently guessed).
# 'recurrence' and 'confirmation' are left out on purpose: independent confirmation IS the likelihood term.
FUSION_IMPACT_COMPONENTS = ("severity", "exposure", "environment")
# 5x5 matrix: product of (likelihood band 1-5) x (impact band 1-5) -> category (first threshold met)
FUSION_MATRIX_CATEGORIES = ((20, "Critical"), (12, "High"), (6, "Moderate"), (1, "Low"))
# Hash-chain key: if set, ledger hashes are HMAC-SHA256 (an attacker who rewrites the DB cannot
# recompute the chain without the key). If unset, plain SHA-256 is used (tamper-EVIDENT, not tamper-proof).
LEDGER_KEY_ENV = "KP_LEDGER_KEY"
