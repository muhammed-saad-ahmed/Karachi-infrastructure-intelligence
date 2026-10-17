"""Edge runtime for a YOLOv8-nano hazard detector (pothole / open manhole / sewage overflow) on ONNX Runtime.

Why this exists
---------------
The original prototype ran Ultralytics + PyTorch (a heavy dependency stack, awkward on a phone or a $0 host).
Once a model is exported to ONNX, inference needs only `onnxruntime` + NumPy + OpenCV:
the same file can run on a laptop CPU, a Raspberry Pi, an Android phone (ONNX Runtime Mobile) or
in the browser (onnxruntime-web / WebAssembly) - so there is no inference server to pay for.

What is implemented (all pure Python 3.10+, no PyTorch at inference time)
-------------------------------------------------------------------------
    letterbox / preprocess      aspect-preserving resize + pad, BGR->RGB, 0-1, NCHW (single OpenCV call)
    decode_output               YOLOv8 raw head (1, 4+nc, N) or end-to-end (1, N, 6) -> boxes in ORIGINAL pixels
    nms                         class-aware, NumPy only, bounded candidate count => bounded CPU cost
    OnnxDetector                drop-in `Detector` for the existing registry (detection.get_detector)
    export_to_onnx              .pt -> .onnx (needs `ultralytics`, only on the machine that trains/exports)
    quantize_int8_static        QDQ INT8 using REAL calibration images (refuses to run on too few)
    compare_models              FP32 vs INT8: latency AND detection agreement (so quantization is never trusted blindly)
    benchmark                   latency percentiles + FPS on THIS machine

Honesty notes
-------------
* No trained pothole/sewage weights ship with this repository. This module is the deployment path for the
  weights you train (see scripts/train_edge_model.py). Until then the app stays in baseline mode.
* INT8 quantization can lower accuracy, especially on small objects like potholes. `compare_models` reports
  agreement with FP32 - that is NOT accuracy against ground truth. Evaluate the final file in Model Lab with
  real labelled images before trusting it.
* Reported speed-ups depend on the CPU; measure with `benchmark` on the target device.

CLI
---
    python -m computer_vision.edge_onnx export   --weights best.pt --out models/custom/hazard_n.onnx
    python -m computer_vision.edge_onnx quantize --model models/custom/hazard_n.onnx --calib data/raw --out models/custom/hazard_n.int8.onnx
    python -m computer_vision.edge_onnx compare  --fp32 models/custom/hazard_n.onnx --int8 models/custom/hazard_n.int8.onnx --images data/raw
    python -m computer_vision.edge_onnx bench    --model models/custom/hazard_n.onnx
    python -m computer_vision.edge_onnx infer    --model models/custom/hazard_n.onnx --image road.jpg
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import threading
import time

import numpy as np

import config
from computer_vision.detection import Detector

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


# ====================================================================== pre-processing
def letterbox(img, size=(640, 640), color=114):
    """Resize keeping aspect ratio, pad to `size` (h, w) with grey. Returns (canvas, ratio, (pad_left, pad_top)).
    Same geometry Ultralytics uses at training time, so boxes match what the model learned."""
    import cv2
    th, tw = (size, size) if isinstance(size, int) else size
    h, w = img.shape[:2]
    r = min(th / h, tw / w)
    nh, nw = max(1, round(h * r)), max(1, round(w * r))
    resized = img if (nh, nw) == (h, w) else cv2.resize(
        img, (nw, nh), interpolation=cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR)
    top, left = (th - nh) // 2, (tw - nw) // 2
    canvas = np.full((th, tw, 3), color, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, (left, top)


def preprocess(img_bgr, size=(640, 640)):
    """BGR uint8 image -> (float32 NCHW tensor in [0,1], ratio, pad). One OpenCV call does BGR->RGB, /255, HWC->CHW."""
    import cv2
    canvas, r, pad = letterbox(img_bgr, size)
    blob = cv2.dnn.blobFromImage(canvas, scalefactor=1 / 255.0, size=(canvas.shape[1], canvas.shape[0]),
                                 swapRB=True, crop=False)
    return np.ascontiguousarray(blob, dtype=np.float32), r, pad


# ====================================================================== post-processing
def nms(boxes, scores, iou_thr):
    """Greedy NMS on xyxy boxes (NumPy only). Returns indices to keep, highest score first."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        w = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
        h = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        inter = w * h
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def decode_output(raw, *, ratio, pad, orig_shape, conf_thr=None, iou_thr=None, max_candidates=None,
                  max_det=None, end2end=False):
    """Turn a raw model output into [(x1,y1,x2,y2, score, class_id)] in ORIGINAL image pixels.

    raw shapes accepted:
        (1, 4+nc, N) or (4+nc, N)   YOLOv8/YOLO11 head: cx,cy,w,h then per-class scores (no objectness)
        (1, N, 4+nc)                already transposed
        (1, N, 6) with end2end=True NMS-free export: x1,y1,x2,y2,score,class
    `end2end` must be explicit: a 2-class raw head is also 6 wide, so it cannot be inferred reliably.
    """
    conf_thr = config.EDGE_SCORE_THRESHOLD if conf_thr is None else conf_thr
    iou_thr = config.EDGE_NMS_IOU if iou_thr is None else iou_thr
    max_candidates = max_candidates or config.EDGE_MAX_CANDIDATES
    max_det = max_det or config.EDGE_MAX_DETECTIONS
    a = np.asarray(raw, dtype=np.float32)
    a = a[0] if a.ndim == 3 else a
    if a.ndim != 2:
        raise ValueError(f"unexpected model output shape {np.asarray(raw).shape}")
    if not end2end and a.shape[0] <= a.shape[1]:
        a = a.T                                   # (4+nc, N) -> (N, 4+nc)
    if end2end:
        boxes, scores, cls = a[:, :4].copy(), a[:, 4], a[:, 5].astype(int)
    else:
        cxcywh, cls_scores = a[:, :4], a[:, 4:]
        cls = cls_scores.argmax(1)
        scores = cls_scores[np.arange(len(a)), cls]
        boxes = np.empty_like(cxcywh)
        boxes[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2
        boxes[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2
        boxes[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2
        boxes[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2
    keep = scores >= conf_thr
    boxes, scores, cls = boxes[keep], scores[keep], cls[keep]
    if len(scores) == 0:
        return []
    if len(scores) > max_candidates:              # bound NMS cost on noisy frames
        top = np.argpartition(-scores, max_candidates - 1)[:max_candidates]
        boxes, scores, cls = boxes[top], scores[top], cls[top]
    if not end2end:                               # class-aware NMS via coordinate offset trick
        offset = cls[:, None].astype(np.float32) * 8192.0
        idx = nms(boxes + offset, scores, iou_thr)[:max_det]
    else:
        idx = list(np.argsort(-scores)[:max_det])
    boxes, scores, cls = boxes[idx], scores[idx], cls[idx]
    left, top = pad
    boxes = (boxes - np.array([left, top, left, top], dtype=np.float32)) / ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return [(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s), int(c))
            for b, s, c in zip(boxes, scores, cls)]


# ====================================================================== runtime
def _parse_names(meta):
    """Ultralytics stores class names in ONNX metadata as a Python-dict string: "{0: 'pothole', 1: ...}"."""
    raw = (meta or {}).get("names")
    if not raw:
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return None
    if isinstance(parsed, dict):
        return [str(parsed[k]) for k in sorted(parsed, key=lambda x: int(x))]
    return [str(x) for x in parsed] if isinstance(parsed, (list, tuple)) else None


def make_session(model_path, threads=None, providers=None):
    """ONNX Runtime session tuned for CPU/edge: full graph optimisation, bounded threads, no idle spinning."""
    import onnxruntime as ort
    threads = threads if threads is not None else config.EDGE_NUM_THREADS
    threads = threads or max(1, min(4, os.cpu_count() or 1))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = int(threads)
    so.inter_op_num_threads = 1
    so.enable_cpu_mem_arena = True
    so.enable_mem_pattern = True
    try:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # don't burn battery between frames
    except Exception:
        pass
    return ort.InferenceSession(model_path, sess_options=so, providers=providers or ["CPUExecutionProvider"])


class OnnxDetector(Detector):
    """Drop-in replacement for YoloDetector backed by ONNX Runtime (no PyTorch/Ultralytics needed)."""

    def __init__(self, model_path, name=None, scope=None, imgsz=None,
                 class_names=None, end2end=False, threads=None):
        self.model_path, self._name, self._scope = model_path, name, scope
        self.version = os.path.basename(model_path)
        self._imgsz_override, self._names_override = imgsz, class_names
        self.end2end, self.threads = end2end, threads
        self._sess, self._lock = None, threading.Lock()
        self.last_timing = {}

    # -- identity: decided from the model's own class list, so a COCO model is never presented as a hazard model
    def has_hazard_classes(self):
        return any(str(n).lower() in config.HAZARD_LABEL_MAP for n in self.names)

    @property
    def name(self):
        return self._name or ("yolo-custom-onnx" if self.has_hazard_classes() else "yolo-onnx-baseline")

    @property
    def scope(self):
        return self._scope or ("hazards (edge ONNX)" if self.has_hazard_classes() else
                               "general objects only - no hazard classes (edge ONNX)")

    # -- lifecycle
    def available(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False
        return os.path.exists(self.model_path)

    def _session(self):
        if self._sess is None:
            with self._lock:
                if self._sess is None:
                    sess = make_session(self.model_path, self.threads)
                    inp = sess.get_inputs()[0]
                    h, w = inp.shape[2], inp.shape[3]
                    static = isinstance(h, int) and isinstance(w, int)
                    self._input_name = inp.name
                    self._size = tuple(self._imgsz_override) if isinstance(self._imgsz_override, (tuple, list)) else (
                        (self._imgsz_override,) * 2 if self._imgsz_override else
                        ((h, w) if static else (config.EDGE_IMGSZ,) * 2))
                    self._names = self._names_override or _parse_names(sess.get_modelmeta().custom_metadata_map)
                    self._sess = sess
        return self._sess

    @property
    def names(self):
        self._session()
        if self._names:
            return self._names
        n_out = self._sess.get_outputs()[0].shape
        nc = (n_out[1] - 4) if len(n_out) == 3 and isinstance(n_out[1], int) and n_out[1] < (n_out[2] or 1 << 30) else None
        return list(config.EDGE_HAZARD_CLASSES) if nc == len(config.EDGE_HAZARD_CLASSES) else \
            [f"class_{i}" for i in range(nc or 0)]

    def class_names(self):
        return sorted(set(self.names))

    # -- inference
    def detect(self, img):
        sess = self._session()
        t0 = time.perf_counter()
        blob, ratio, pad = preprocess(img, self._size)
        t1 = time.perf_counter()
        out = sess.run(None, {self._input_name: blob})[0]
        t2 = time.perf_counter()
        dets = decode_output(out, ratio=ratio, pad=pad, orig_shape=img.shape, end2end=self.end2end)
        names = self.names
        result = [{"label": names[c] if c < len(names) else f"class_{c}", "confidence": s, "bbox": [x1, y1, x2, y2]}
                  for x1, y1, x2, y2, s, c in dets]
        t3 = time.perf_counter()
        self.last_timing = {"preprocess_ms": round((t1 - t0) * 1e3, 2), "inference_ms": round((t2 - t1) * 1e3, 2),
                            "postprocess_ms": round((t3 - t2) * 1e3, 2), "total_ms": round((t3 - t0) * 1e3, 2)}
        return result


def find_onnx_models(directory=None):
    return sorted(glob.glob(os.path.join(directory or config.EDGE_MODEL_DIR, "*.onnx")))


# ====================================================================== export / quantize (build-time tools)
def export_to_onnx(weights, out_path=None, imgsz=None, opset=12, simplify=True, dynamic=False):
    """Export Ultralytics .pt weights to ONNX FP32. Run this where `ultralytics` is installed (e.g. a free
    Colab/Kaggle notebook), then copy the .onnx into models/custom/ - the deployed app does not need PyTorch.
    opset 12 and a static input shape give the broadest runtime compatibility (mobile, WASM)."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("export needs `pip install ultralytics` (only on the exporting machine)") from exc
    imgsz = imgsz or config.EDGE_IMGSZ
    produced = YOLO(weights).export(format="onnx", imgsz=imgsz, opset=opset, simplify=simplify,
                                    dynamic=dynamic, half=False)
    produced = str(produced)
    if out_path and os.path.abspath(out_path) != os.path.abspath(produced):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        os.replace(produced, out_path)
        produced = out_path
    return produced


def list_images(directory):
    paths = []
    for root, _, files in os.walk(directory):
        paths += [os.path.join(root, f) for f in files if f.lower().endswith(IMAGE_EXTS)]
    return sorted(paths)


def quantize_int8_static(model_fp32, model_int8, calib_dir, imgsz=None, max_images=200, per_channel=True,
                         nodes_to_exclude=None):
    """Static INT8 (QDQ) quantization calibrated on REAL images.

    Why static, and why a real dataset: YOLO is convolution-heavy; ONNX Runtime's *dynamic* quantization only
    covers MatMul/Gemm well, so it barely helps a conv net. Static quantization needs representative activations;
    calibrating on random noise or on the wrong domain silently wrecks accuracy, so this refuses to run
    on fewer than config.EDGE_CALIBRATION_MIN_IMAGES images.
    ALWAYS follow with compare_models() and a Model Lab evaluation."""
    paths = list_images(calib_dir)
    if len(paths) < config.EDGE_CALIBRATION_MIN_IMAGES:
        raise ValueError(f"found {len(paths)} images in {calib_dir!r}; static INT8 calibration needs at least "
                         f"{config.EDGE_CALIBRATION_MIN_IMAGES} REAL, representative images (Karachi roads, day+night, "
                         "wet+dry). Collect more data rather than lowering this bar.")
    try:
        from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                              quantize_static)
    except ImportError as exc:
        raise ImportError("quantization needs `pip install onnx onnxruntime`") from exc
    import cv2
    paths = paths[:max_images]
    sess = make_session(model_fp32, threads=1)
    inp = sess.get_inputs()[0]
    size = (inp.shape[2], inp.shape[3]) if isinstance(inp.shape[2], int) else (imgsz or config.EDGE_IMGSZ,) * 2
    name, del_sess = inp.name, sess

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._it = iter(paths)

        def get_next(self):
            for p in self._it:
                img = cv2.imread(p)
                if img is not None:
                    return {name: preprocess(img, size)[0]}
            return None

    src = model_fp32
    try:  # recommended shape-inference / fusion pass before quantizing; optional
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = os.path.splitext(model_int8)[0] + ".pre.onnx"
        quant_pre_process(model_fp32, pre)
        src = pre
    except Exception:
        pass
    extra = {"ActivationSymmetric": False, "WeightSymmetric": True}
    method = getattr(CalibrationMethod, "Percentile", CalibrationMethod.MinMax)
    if method == getattr(CalibrationMethod, "Percentile", None):
        extra["CalibPercentile"] = 99.999        # clip activation outliers; MinMax is dominated by them in detectors
    os.makedirs(os.path.dirname(os.path.abspath(model_int8)), exist_ok=True)
    quantize_static(src, model_int8, _Reader(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=per_channel, calibrate_method=method,
                    nodes_to_exclude=nodes_to_exclude or [], extra_options=extra)
    if src != model_fp32 and os.path.exists(src):
        os.remove(src)
    del del_sess
    return {"int8_model": model_int8, "n_calibration_images": len(paths),
            "size_mb_fp32": round(os.path.getsize(model_fp32) / 1e6, 2),
            "size_mb_int8": round(os.path.getsize(model_int8) / 1e6, 2)}


# ====================================================================== measurement
def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def benchmark(det_or_path, runs=None, size=None, warmup=3):
    """Latency percentiles / FPS on THIS machine using a random-noise frame. Speed only - says nothing about accuracy."""
    det = det_or_path if isinstance(det_or_path, OnnxDetector) else OnnxDetector(det_or_path)
    if not det.available():
        return None
    runs = runs or config.BENCHMARK_RUNS
    h, w = size or (config.EDGE_IMGSZ, config.EDGE_IMGSZ)
    img = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    for _ in range(warmup):
        det.detect(img)
    total, infer = [], []
    for _ in range(runs):
        det.detect(img)
        total.append(det.last_timing["total_ms"])
        infer.append(det.last_timing["inference_ms"])
    total = np.array(total)
    return {"model": det.version, "n_runs": runs, "threads": det.threads or config.EDGE_NUM_THREADS or "auto",
            "mean_ms": round(float(total.mean()), 1), "p50_ms": round(float(np.percentile(total, 50)), 1),
            "p95_ms": round(float(np.percentile(total, 95)), 1), "min_ms": round(float(total.min()), 1),
            "max_ms": round(float(total.max()), 1), "inference_only_mean_ms": round(float(np.mean(infer)), 1),
            "fps": round(1000.0 / float(total.mean()), 1) if total.mean() else None,
            "size_mb": round(os.path.getsize(det.model_path) / 1e6, 2)}


def compare_models(fp32_path, int8_path, images, iou_thr=0.5, conf_thr=0.25):
    """Quantization sanity check. `images`: list of BGR arrays or a directory path. Reports speed AND how closely
    the INT8 model's detections match the FP32 model's (treating FP32 as reference). This is agreement,
    NOT accuracy against ground truth."""
    import cv2
    if isinstance(images, str):
        images = [im for im in (cv2.imread(p) for p in list_images(images)) if im is not None]
    if not images:
        raise ValueError("no images to compare on")
    ref, q = OnnxDetector(fp32_path), OnnxDetector(int8_path)
    tp = fp = fn = 0
    t_ref, t_q = [], []
    ref.detect(images[0]); q.detect(images[0])           # warm-up
    for img in images:
        a = [d for d in ref.detect(img) if d["confidence"] >= conf_thr]; t_ref.append(ref.last_timing["total_ms"])
        b = [d for d in q.detect(img) if d["confidence"] >= conf_thr]; t_q.append(q.last_timing["total_ms"])
        used = set()
        for d in sorted(b, key=lambda x: -x["confidence"]):
            m = [(_iou(d["bbox"], r["bbox"]), j) for j, r in enumerate(a) if j not in used and r["label"] == d["label"]]
            s, j = max(m, default=(0.0, -1))
            if s >= iou_thr:
                tp += 1; used.add(j)
            else:
                fp += 1
        fn += len(a) - len(used)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n_images": len(images), "fp32_mean_ms": round(float(np.mean(t_ref)), 1),
            "int8_mean_ms": round(float(np.mean(t_q)), 1),
            "speedup_x": round(float(np.mean(t_ref) / np.mean(t_q)), 2) if np.mean(t_q) else None,
            "agreement_precision": round(p, 3), "agreement_recall": round(r, 3),
            "agreement_f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "size_mb_fp32": round(os.path.getsize(fp32_path) / 1e6, 2),
            "size_mb_int8": round(os.path.getsize(int8_path) / 1e6, 2),
            "note": "Agreement with the FP32 model, not accuracy vs ground truth. Validate in Model Lab."}


# ====================================================================== CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m computer_vision.edge_onnx", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export"); e.add_argument("--weights", required=True); e.add_argument("--out")
    e.add_argument("--imgsz", type=int, default=config.EDGE_IMGSZ)
    q = sub.add_parser("quantize"); q.add_argument("--model", required=True); q.add_argument("--calib", required=True)
    q.add_argument("--out", required=True)
    c = sub.add_parser("compare"); c.add_argument("--fp32", required=True); c.add_argument("--int8", required=True)
    c.add_argument("--images", required=True)
    b = sub.add_parser("bench"); b.add_argument("--model", required=True); b.add_argument("--runs", type=int, default=30)
    i = sub.add_parser("infer"); i.add_argument("--model", required=True); i.add_argument("--image", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(export_to_onnx(a.weights, a.out, a.imgsz))
    elif a.cmd == "quantize":
        print(json.dumps(quantize_int8_static(a.model, a.out, a.calib), indent=2))
    elif a.cmd == "compare":
        print(json.dumps(compare_models(a.fp32, a.int8, a.images), indent=2))
    elif a.cmd == "bench":
        print(json.dumps(benchmark(a.model, runs=a.runs), indent=2))
    else:
        import cv2
        img = cv2.imread(a.image)
        if img is None:
            raise SystemExit(f"cannot read {a.image}")
        d = OnnxDetector(a.model)
        print(json.dumps({"detections": d.detect(img), "timing": d.last_timing}, indent=2))


if __name__ == "__main__":
    _main()
