"""Train -> export -> (optionally) quantize a YOLOv8-nano hazard detector, end to end, on FREE hardware.

Run this on Google Colab / Kaggle (free GPU) - NOT inside the Streamlit app:

    !pip install ultralytics onnx onnxruntime
    !python -m scripts.train_edge_model --data data/hazard_dataset.yaml --epochs 100 --calib data/raw

What you need first (this repository does not ship it):
    * a labelled dataset in YOLO format with classes  pothole / open_manhole / sewage_overflow
      (see data/hazard_dataset.example.yaml). Public pothole / road-damage sets exist (e.g. RDD2020, cited in
      docs/research_report.md) but contain no manholes or sewage and are not Karachi; Karachi photos taken with
      consent are what will make the model relevant here.
    * split by LOCATION, not randomly (scripts/dataset_tools.py::geographic_split) or the test score will lie.

Outputs go to models/custom/: <name>.pt (training), <name>.onnx (FP32 edge model) and, with --calib,
<name>.int8.onnx. The app picks up any .onnx in models/custom/ automatically - no code change.

After training: evaluate in Model Lab with held-out, geographically separate images and log the numbers with
experiments/tracker.py. Until then, treat the model as untested.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import config
from computer_vision import edge_onnx


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="YOLO dataset yaml")
    ap.add_argument("--name", default="hazard_n")
    ap.add_argument("--base", default="yolov8n.pt", help="starting weights (COCO-pretrained nano)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=config.EDGE_IMGSZ)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--calib", help="folder of REAL images for INT8 calibration (>= config.EDGE_CALIBRATION_MIN_IMAGES)")
    ap.add_argument("--out", default=config.EDGE_MODEL_DIR)
    a = ap.parse_args(argv)

    from ultralytics import YOLO       # imported late: only the training machine needs it
    os.makedirs(a.out, exist_ok=True)
    model = YOLO(a.base)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name, patience=20,
                # augmentations that match Karachi conditions: night, glare, rain blur, odd camera angles
                hsv_v=0.5, degrees=5.0, perspective=0.0005, mosaic=1.0, fliplr=0.5)
    best = str(model.trainer.best)
    shutil.copy(best, os.path.join(a.out, f"{a.name}.pt"))
    onnx_path = edge_onnx.export_to_onnx(best, os.path.join(a.out, f"{a.name}.onnx"), imgsz=a.imgsz)
    report = {"fp32_onnx": onnx_path, "benchmark": edge_onnx.benchmark(onnx_path)}
    if a.calib:
        int8 = os.path.join(a.out, f"{a.name}.int8.onnx")
        report["quantization"] = edge_onnx.quantize_int8_static(onnx_path, int8, a.calib, a.imgsz)
        report["compare"] = edge_onnx.compare_models(onnx_path, int8, a.calib)
        report["benchmark_int8"] = edge_onnx.benchmark(int8)
    print(json.dumps(report, indent=2))
    print("\nNext: evaluate on held-out geographic split in Model Lab. Remove the .pt from models/custom/ if you want "
          "the app to use ONNX only (ONNX files are preferred automatically when both exist).")


if __name__ == "__main__":
    main()
