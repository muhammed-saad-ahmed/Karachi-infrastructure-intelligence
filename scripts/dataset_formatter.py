"""Automated INT8 calibration pipeline: arbitrary local images -> augmented calibration set -> INT8 model -> checks.

PROBLEM
    Static INT8 quantization (computer_vision/edge_onnx.py) needs >= EDGE_CALIBRATION_MIN_IMAGES representative images.
    Collecting a big labelled Karachi dataset takes months, so the quantization path - the part most likely to break
    - was never exercised. This script lets the WHOLE chain run, and be tested, from whatever photos exist today.

WHAT IT DOES
    1. collect      any mix of image files / folders (duplicates removed by hash, unreadable files skipped)
    2. augment      NumPy/OpenCV pipeline that imitates Karachi lighting: harsh noon sun, golden hour, dusk, sodium-lit
                    night, monsoon overcast, dust haze, headlight glare, flyover shade; plus random zoom/crop/flip/
                    rotation, sensor noise and JPEG compression. Seeded, so a run is reproducible.
    3. calibrate    writes N (default 50) JPEGs + a manifest and hands the folder to quantize_int8_static()
    4. verify       compares the raw output tensors of INT8 and FP32 (cosine, relative L2, SQNR, top-k anchor overlap),
                    detection agreement, and measures latency of both (FP32 vs INT8, in milliseconds)
    5. gate         evaluate_int8() turns those numbers into pass/fail checks that pytest asserts on
                    (tests/test_int8_pipeline.py) and the CLI uses for its exit code

HONESTY (also written into every manifest and report)
    * 50 augmentations of a handful of photos are NOT 50 independent photos. They exercise the pipeline and give a
      sane activation range, but a model calibrated this way is `deployment_grade: false` until the set contains at
      least EDGE_CALIB_MIN_UNIQUE_SOURCES distinct REAL photos. Procedurally generated scenes (used when you pass no
      images at all) are flagged `synthetic` and can never be deployment grade.
    * Agreement with FP32 is not accuracy. Evaluate the final .onnx on labelled images (Model Lab) before trusting it.
    * Augmentation imitates lighting conditions; it cannot invent road surfaces or potholes the photos do not contain.

CLI
    python -m scripts.dataset_formatter build --src data/raw --out data/calib --n 50
    python -m scripts.dataset_formatter int8  --model models/custom/hazard_n.onnx --src data/raw --work reports/int8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

import config

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
LIGHTING_PROFILES = ("harsh_noon", "golden_hour", "dusk", "night_sodium", "monsoon_overcast", "dust_haze",
                     "headlight_glare", "flyover_shade", "neutral")


# ====================================================================== collecting images
def collect_images(sources):
    """Reads every readable image under `sources` (files and/or folders). Returns (list of BGR arrays, list of names).
    Exact duplicates (same pixel bytes) are dropped so 'unique source' counts are honest."""
    import cv2
    paths = []
    for src in ([sources] if isinstance(sources, (str, os.PathLike)) else list(sources or [])):
        src = str(src)
        if os.path.isdir(src):
            for root, _, files in sorted(os.walk(src)):
                paths += [os.path.join(root, f) for f in sorted(files) if f.lower().endswith(IMAGE_EXTS)]
        elif os.path.isfile(src) and src.lower().endswith(IMAGE_EXTS):
            paths.append(src)
    images, names, seen = [], [], set()
    for p in paths:
        img = cv2.imread(p)
        if img is None or img.size == 0:
            continue
        h = hashlib.sha256(img.tobytes()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        images.append(img)
        names.append(os.path.basename(p))
    return images, names


def procedural_road_scene(rng, size=640):
    """A synthetic road photo substitute (sky, buildings, asphalt, lane markings, dark pothole-like blobs, wet patches)
    for running the pipeline with NO real images. Obviously fake; flagged `synthetic` everywhere it is used.
    Works for any size >= 32 (all feature sizes scale with the image)."""
    import cv2
    s = max(32, int(size))
    k = s / 640.0
    px = lambda v: max(1, int(round(v * k)))                      # a length in 640-px units -> pixels
    img = np.zeros((s, s, 3), np.float32)
    horizon = int(s * rng.uniform(0.30, 0.42))
    sky = np.linspace(0.85, 0.65, horizon)[:, None, None] * np.array([0.95, 0.85, 0.7])[None, None, :]
    img[:horizon] = np.broadcast_to(sky, (horizon, s, 3))
    x = 0
    while x < s:                                                         # building skyline
        w = int(rng.integers(px(30), px(110) + 1)); h = int(rng.integers(px(20), max(px(20) + 1, horizon - 2)))
        img[max(0, horizon - h):horizon, x:x + w] = rng.uniform(0.25, 0.7, 3)
        x += w
    asphalt = rng.uniform(0.28, 0.42)
    img[horizon:] = asphalt + np.linspace(0, 0.12, s - horizon)[:, None, None]

    def where(margin_x, top):
        lo_y, hi_y = horizon + top, max(horizon + top + 1, s - px(20))
        return int(rng.integers(margin_x, max(margin_x + 1, s - margin_x))), int(rng.integers(lo_y, hi_y))

    for _ in range(int(rng.integers(0, 4))):                             # potholes: dark blob with a lighter rim
        cx, cy = where(px(40), px(40))
        ax, ay = int(rng.integers(px(12), px(45) + 1)), int(rng.integers(px(6), px(22) + 1))
        cv2.ellipse(img, (cx, cy), (ax + 2, ay + 2), 0, 0, 360, (asphalt + 0.1,) * 3, -1)
        cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (asphalt * 0.35,) * 3, -1)
    for _ in range(int(rng.integers(0, 3))):                             # wet patches
        cx, cy = where(px(40), px(30))
        cv2.ellipse(img, (cx, cy), (int(rng.integers(px(30), px(90) + 1)), int(rng.integers(px(8), px(25) + 1))),
                    0, 0, 360, (0.55, 0.58, 0.62), -1)
    for y in range(horizon + px(10), s, px(40)):                         # dashed centre line, widening towards the viewer
        half = max(1, int((y - horizon) * 0.02))
        img[y:y + px(16), s // 2 - half:s // 2 + half] = 0.9
    img += rng.normal(0, 0.015, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ====================================================================== augmentation
def _as_float(img):
    return img.astype(np.float32) / 255.0


def _to_u8(f):
    return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _tint(f, bgr_gain):
    return f * np.array(bgr_gain, np.float32)[None, None, :]


def _contrast(f, c, pivot=0.5):
    return (f - pivot) * c + pivot


def _blob(shape, rng, strength, radius_frac=(0.08, 0.25), center=None):
    """Additive gaussian light blob (street lamp, headlight)."""
    h, w = shape[:2]
    cy, cx = center if center else (rng.uniform(0.1, 0.9) * h, rng.uniform(0.1, 0.9) * w)
    sigma = rng.uniform(*radius_frac) * max(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    return (strength * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))))[..., None].astype(np.float32)


def apply_lighting(img, profile, rng):
    """One Karachi lighting condition applied to a BGR uint8 image. Returns (image, params dict)."""
    f = _as_float(img)
    p = {"profile": profile}
    if profile == "harsh_noon":
        c, b, g = rng.uniform(1.25, 1.55), rng.uniform(0.03, 0.12), rng.uniform(0.80, 0.92)
        f = np.power(np.clip(_contrast(f, c) + b, 0, 1), g)
        p.update(contrast=c, brightness=b, gamma=g)
    elif profile == "golden_hour":
        br = rng.uniform(0.85, 1.05)
        f = _tint(f, (0.78, 1.0, 1.18)) * br                           # BGR gains: warm = more red, less blue
        p.update(brightness=br)
    elif profile == "dusk":
        br = rng.uniform(0.42, 0.68)
        f = _tint(f, (1.12, 0.96, 0.88)) * br
        p.update(brightness=br)
    elif profile == "night_sodium":
        br, noise = rng.uniform(0.18, 0.38), rng.uniform(0.015, 0.04)
        f = _tint(f, (0.55, 0.95, 1.25)) * br
        for _ in range(int(rng.integers(1, 4))):
            f = f + _blob(f.shape, rng, rng.uniform(0.35, 0.8), (0.04, 0.12)) * np.array([0.35, 0.7, 1.0], np.float32)
        f = f + rng.normal(0, noise, f.shape).astype(np.float32)
        p.update(brightness=br, sensor_noise=noise)
    elif profile == "monsoon_overcast":
        c, sat, br = rng.uniform(0.60, 0.80), rng.uniform(0.40, 0.70), rng.uniform(0.78, 0.95)
        grey = f.mean(axis=2, keepdims=True)
        f = _contrast(grey + (f - grey) * sat, c) * br
        f = _tint(f, (1.06, 1.0, 0.95))
        p.update(contrast=c, saturation=sat, brightness=br)
    elif profile == "dust_haze":
        t = rng.uniform(0.50, 0.80)
        airlight = np.array([0.58, 0.70, 0.78], np.float32)               # dusty tan in BGR
        f = f * t + airlight[None, None, :] * (1 - t)                      # atmospheric scattering: I t + A (1 - t)
        p.update(transmission=t)
    elif profile == "headlight_glare":
        f = f * rng.uniform(0.45, 0.65)
        h, w = f.shape[:2]
        f = f + _blob(f.shape, rng, rng.uniform(0.7, 1.0), (0.10, 0.22), center=(rng.uniform(0.45, 0.8) * h, rng.uniform(0.2, 0.8) * w))
        p.update(glare=True)
    elif profile == "flyover_shade":
        h, w = f.shape[:2]
        ramp = np.linspace(0, 1, w, dtype=np.float32)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        depth = rng.uniform(0.35, 0.65)
        f = f * (1 - depth * np.clip(ramp * 1.6 - 0.2, 0, 1))[None, :, None]
        p.update(shade_depth=depth)
    elif profile != "neutral":
        raise ValueError(f"unknown lighting profile {profile!r}")
    return _to_u8(f), p


def random_geometry(img, rng, size=640):
    """Random zoom/crop (square), horizontal flip and a small rotation, resized to size x size. -> (image, params)."""
    import cv2
    h, w = img.shape[:2]
    zoom = rng.uniform(1.0, 1.6)                                         # >1 = look closer (crop smaller window)
    side = max(8, int(min(h, w) / zoom))
    y0 = int(rng.integers(0, max(1, h - side + 1)))
    x0 = int(rng.integers(0, max(1, w - side + 1)))
    crop = img[y0:y0 + side, x0:x0 + side]
    out = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    flip = bool(rng.random() < 0.5)
    if flip:
        out = out[:, ::-1]
    angle = float(rng.uniform(-5, 5))
    m = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    out = cv2.warpAffine(np.ascontiguousarray(out), m, (size, size), borderMode=cv2.BORDER_REFLECT)
    return out, {"zoom": round(zoom, 3), "crop_xy": [x0, y0], "flip": flip, "rotation_deg": round(angle, 2)}


def augment(img, rng, size=640, profile=None):
    """Full recipe for one calibration image: geometry -> lighting -> sensor noise -> JPEG round trip.
    Returns (BGR uint8 image of shape (size, size, 3), recipe dict that fully describes what was done)."""
    import cv2
    if img is None or getattr(img, "ndim", 0) != 3 or img.shape[2] != 3:
        raise ValueError("augment() expects a BGR image of shape (H, W, 3)")
    out, geo = random_geometry(img, rng, size)
    profile = profile or LIGHTING_PROFILES[int(rng.integers(len(LIGHTING_PROFILES)))]
    out, light = apply_lighting(out, profile, rng)
    sigma = float(rng.uniform(0.0, 0.02))
    if sigma > 0:
        out = _to_u8(_as_float(out) + rng.normal(0, sigma, out.shape).astype(np.float32))
    quality = int(rng.integers(60, 96))
    ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out, {"geometry": geo, "lighting": light, "noise_sigma": round(sigma, 4), "jpeg_quality": quality}


# ====================================================================== building the calibration set
def build_calibration_set(sources, out_dir, n=None, size=None, seed=0, allow_synthetic=True):
    """Writes `n` augmented JPEGs (default config.EDGE_CALIBRATION_MIN_IMAGES) to out_dir plus manifest.json and returns
    the manifest. With no readable source images it generates procedural scenes (flagged synthetic) if allowed.
    Round-robins over the sources so every distinct photo is used as evenly as possible."""
    import cv2
    n = int(n or config.EDGE_CALIBRATION_MIN_IMAGES)
    size = int(size or config.EDGE_IMGSZ)
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    images, names = collect_images(sources) if sources else ([], [])
    synthetic = False
    if not images:
        if not allow_synthetic:
            raise ValueError("no readable source images found and allow_synthetic=False")
        synthetic = True
        images = [procedural_road_scene(np.random.default_rng(seed * 1000 + i), size) for i in range(min(n, 12))]
        names = [f"procedural_{i:02d}" for i in range(len(images))]
    os.makedirs(out_dir, exist_ok=True)
    entries = []
    for i in range(n):
        j = i % len(images)
        profile = LIGHTING_PROFILES[i % len(LIGHTING_PROFILES)] if n >= len(LIGHTING_PROFILES) else None
        img, recipe = augment(images[j], rng, size, profile)
        fname = f"calib_{i:03d}.jpg"
        if not cv2.imwrite(os.path.join(out_dir, fname), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
            raise OSError(f"could not write {os.path.join(out_dir, fname)}")
        entries.append({"file": fname, "source": names[j], "recipe": recipe,
                        "sha256": hashlib.sha256(img.tobytes()).hexdigest()})
    unique = len(images)
    grade = (not synthetic) and unique >= config.EDGE_CALIB_MIN_UNIQUE_SOURCES
    manifest = {"n_images": n, "size": size, "seed": seed, "n_unique_sources": unique, "synthetic": synthetic,
                "lighting_profiles": sorted({e["recipe"]["lighting"]["profile"] for e in entries}),
                "deployment_grade": grade,
                "caveat": ("Pipeline-validation calibration set: " + (
                    "all images are procedurally generated." if synthetic else
                    f"{unique} distinct source photo(s) - fewer than the {config.EDGE_CALIB_MIN_UNIQUE_SOURCES} needed "
                    "for deployment-grade calibration.") if not grade else
                    "Built from enough distinct real photos; still verify accuracy on labelled data."),
                "images": entries}
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# ====================================================================== the full INT8 pipeline + gates
def evaluate_int8(report, *, min_cosine=None, max_nrmse=None, min_topk=None, require_speedup=None):
    """Turns a pipeline report into explicit checks: -> {"passed": bool, "checks": [{name, ok, value, bar}], "warnings": []}.
    The bars come from config.EDGE_INT8_* unless overridden."""
    min_cosine = config.EDGE_INT8_MIN_COSINE if min_cosine is None else min_cosine
    max_nrmse = config.EDGE_INT8_MAX_NRMSE if max_nrmse is None else max_nrmse
    min_topk = config.EDGE_INT8_MIN_TOPK_JACCARD if min_topk is None else min_topk
    require_speedup = config.EDGE_INT8_REQUIRE_SPEEDUP if require_speedup is None else require_speedup
    ag, lat = report["agreement"], report["latency"]
    checks = [
        {"name": "no non-finite INT8 outputs", "ok": ag["n_nonfinite_outputs"] == 0, "value": ag["n_nonfinite_outputs"], "bar": 0},
        {"name": "worst-image cosine >= bar", "ok": ag["cosine_worst"] is not None and ag["cosine_worst"] >= min_cosine,
         "value": ag["cosine_worst"], "bar": min_cosine},
        {"name": "worst-image relative L2 error <= bar", "ok": ag["nrmse_worst"] is not None and ag["nrmse_worst"] <= max_nrmse,
         "value": ag["nrmse_worst"], "bar": max_nrmse}]
    if ag.get("topk_jaccard_worst") is not None:
        checks.append({"name": "worst-image top-k anchor overlap >= bar", "ok": ag["topk_jaccard_worst"] >= min_topk,
                       "value": ag["topk_jaccard_worst"], "bar": min_topk})
    warnings = []
    if lat.get("speedup_x") is not None and lat["speedup_x"] < 1.0:
        msg = f"INT8 is slower than FP32 on this CPU ({lat['speedup_x']}x); it likely lacks fast INT8 instructions"
        if require_speedup:
            checks.append({"name": "INT8 at least as fast as FP32", "ok": False, "value": lat["speedup_x"], "bar": 1.0})
        else:
            warnings.append(msg)
    if not report["calibration"]["deployment_grade"]:
        warnings.append(report["calibration"]["caveat"])
    return {"passed": all(c["ok"] for c in checks), "checks": checks, "warnings": warnings}


def assert_int8_ok(report, **bars):
    """pytest helper: raises AssertionError with the full metric table if any gate fails."""
    verdict = evaluate_int8(report, **bars)
    if not verdict["passed"]:
        failed = [c for c in verdict["checks"] if not c["ok"]]
        raise AssertionError("INT8 acceptance failed: " + "; ".join(
            f"{c['name']} (value {c['value']}, bar {c['bar']})" for c in failed)
            + "\nagreement=" + json.dumps(report["agreement"]) + "\nlatency=" + json.dumps(report["latency"]))
    return verdict


def run_int8_pipeline(fp32_model, work_dir, *, sources=None, n_images=None, seed=0, runs=30, int8_model=None,
                      agreement_images=20, quantizer=None):
    """augment -> quantize -> verify. Needs `onnx` + `onnxruntime` (quantization). `quantizer` lets a caller supply the
    function that produces the INT8 file from (fp32_path, int8_path, calib_dir) - used by tests with a stand-in.
    Returns a report dict: calibration manifest, quantization info, tensor agreement, detection agreement, latency, verdict."""
    from computer_vision import edge_onnx as eo
    if not os.path.exists(fp32_model):
        raise FileNotFoundError(fp32_model)
    os.makedirs(work_dir, exist_ok=True)
    calib_dir = os.path.join(work_dir, "calibration")
    manifest = build_calibration_set(sources, calib_dir, n=n_images, seed=seed)
    int8_path = int8_model or os.path.join(work_dir, os.path.splitext(os.path.basename(fp32_model))[0] + ".int8.onnx")
    quant = (quantizer or eo.quantize_int8_static)(fp32_model, int8_path, calib_dir)
    if isinstance(quant, str) or quant is None:
        quant = {"int8_model": int8_path}
    images, _ = collect_images(calib_dir)
    sample = images[:agreement_images]
    report = {"calibration": {k: v for k, v in manifest.items() if k != "images"}, "quantization": quant,
              "agreement": eo.tensor_agreement(fp32_model, int8_path, sample),
              "latency": eo.latency_comparison(fp32_model, int8_path, runs=runs)}
    try:
        report["detections"] = eo.compare_models(fp32_model, int8_path, sample)
    except Exception as exc:                       # e.g. a non-detector model: tensor agreement above still stands
        report["detections"] = {"skipped": f"{type(exc).__name__}: {exc}"}
    report["verdict"] = evaluate_int8(report)
    with open(os.path.join(work_dir, "int8_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


# ====================================================================== CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scripts.dataset_formatter", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="augment local images into a calibration set")
    b.add_argument("--src", nargs="*", default=[], help="image files / folders (none = procedural synthetic scenes)")
    b.add_argument("--out", required=True)
    b.add_argument("--n", type=int, default=config.EDGE_CALIBRATION_MIN_IMAGES)
    b.add_argument("--size", type=int, default=config.EDGE_IMGSZ)
    b.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("int8", help="calibrate -> quantize -> verify FP32 vs INT8")
    i.add_argument("--model", required=True)
    i.add_argument("--work", required=True)
    i.add_argument("--src", nargs="*", default=[])
    i.add_argument("--n", type=int, default=config.EDGE_CALIBRATION_MIN_IMAGES)
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--runs", type=int, default=30)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        m = build_calibration_set(a.src, a.out, a.n, a.size, a.seed)
        print(json.dumps({k: v for k, v in m.items() if k != "images"}, indent=2))
        return 0
    rep = run_int8_pipeline(a.model, a.work, sources=a.src, n_images=a.n, seed=a.seed, runs=a.runs)
    print(json.dumps({k: rep[k] for k in ("calibration", "agreement", "latency", "verdict")}, indent=2))
    return 0 if rep["verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(_main())
