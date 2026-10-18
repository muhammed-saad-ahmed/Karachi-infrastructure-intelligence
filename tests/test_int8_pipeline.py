"""INT8 calibration / verification pipeline. The tensor-level checks run on a stand-in 'INT8' model (the FP32 conv net
with 8-bit-rounded weights) so they need only onnxruntime; the real static quantization needs the `onnx` package and is
skipped without it."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import config
import tiny_onnx
from computer_vision import edge_onnx as eo
from scripts import dataset_formatter as df


def _models(tmp_path, bits=8):
    fp, q = tmp_path / "fp32.onnx", tmp_path / "int8.onnx"
    fp.write_bytes(tiny_onnx.build_conv_detector_model(seed=0))
    q.write_bytes(tiny_onnx.build_conv_detector_model(seed=0, weight_bits=bits))
    return str(fp), str(q)


# ---------------------------------------------------------------- augmentation + calibration set
def test_every_lighting_profile_changes_the_image_in_the_expected_direction():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(0)
    img = df.procedural_road_scene(rng, 128)
    lum = lambda a: float(a.mean())
    res = {p: df.apply_lighting(img, p, np.random.default_rng(1))[0] for p in df.LIGHTING_PROFILES}
    assert all(r.shape == img.shape and r.dtype == np.uint8 for r in res.values())
    assert lum(res["night_sodium"]) < lum(img) * 0.6 and lum(res["dusk"]) < lum(img)
    assert res["golden_hour"][..., 2].mean() / res["golden_hour"][..., 0].mean() > img[..., 2].mean() / img[..., 0].mean()   # warmer
    assert res["dust_haze"].std() < img.std()                               # haze lowers contrast
    assert res["monsoon_overcast"].std() < img.std()
    assert np.array_equal(res["neutral"], img)
    with pytest.raises(ValueError):
        df.apply_lighting(img, "nope", rng)


def test_augment_is_deterministic_and_validates_input():
    pytest.importorskip("cv2")
    img = df.procedural_road_scene(np.random.default_rng(0), 160)
    a, ra = df.augment(img, np.random.default_rng(3), 96)
    b, rb = df.augment(img, np.random.default_rng(3), 96)
    assert a.shape == (96, 96, 3) and np.array_equal(a, b) and ra == rb
    with pytest.raises(ValueError):
        df.augment(np.zeros((8, 8), np.uint8), np.random.default_rng(0))


def test_calibration_set_has_50_images_manifest_and_honest_flags(tmp_path):
    pytest.importorskip("cv2")
    m = df.build_calibration_set([], str(tmp_path / "c"), size=96, seed=1)
    files = [f for f in os.listdir(tmp_path / "c") if f.endswith(".jpg")]
    assert len(files) == 50 == m["n_images"] and m["synthetic"] and not m["deployment_grade"]
    assert len(m["lighting_profiles"]) >= 8
    on_disk = json.loads((tmp_path / "c" / "manifest.json").read_text())
    assert on_disk["n_images"] == 50 and "caveat" in on_disk
    with pytest.raises(ValueError):
        df.build_calibration_set([], str(tmp_path / "d"), allow_synthetic=False)


def test_few_real_sources_are_not_deployment_grade_but_many_are(tmp_path):
    cv2 = pytest.importorskip("cv2")
    src = tmp_path / "src"; src.mkdir()
    for i in range(3):
        cv2.imwrite(str(src / f"p{i}.jpg"), df.procedural_road_scene(np.random.default_rng(i), 96))
    cv2.imwrite(str(src / "dup.jpg"), cv2.imread(str(src / "p0.jpg")))          # re-encoded copy: counted as its own source (only identical pixels are merged)
    (src / "broken.jpg").write_bytes(b"not an image")
    m = df.build_calibration_set([str(src)], str(tmp_path / "c"), size=96)
    assert not m["synthetic"] and m["n_unique_sources"] >= 3 and not m["deployment_grade"]
    big = tmp_path / "big"; big.mkdir()
    for i in range(config.EDGE_CALIB_MIN_UNIQUE_SOURCES):
        cv2.imwrite(str(big / f"p{i}.png"), df.procedural_road_scene(np.random.default_rng(100 + i), 64))
    m2 = df.build_calibration_set([str(big)], str(tmp_path / "c2"), size=64)
    assert m2["n_unique_sources"] >= config.EDGE_CALIB_MIN_UNIQUE_SOURCES and m2["deployment_grade"]


# ---------------------------------------------------------------- tensor agreement + latency
def test_stand_in_int8_agrees_closely_and_passes_gates(tmp_path):
    pytest.importorskip("onnxruntime"); pytest.importorskip("cv2")
    fp, q = _models(tmp_path)
    imgs = [df.procedural_road_scene(np.random.default_rng(i), 96) for i in range(6)]
    ag = eo.tensor_agreement(fp, q, imgs)
    assert ag["cosine_worst"] > 0.99 and ag["nrmse_worst"] < 0.1 and ag["n_nonfinite_outputs"] == 0
    rep = {"agreement": ag, "latency": eo.latency_comparison(fp, q, runs=5, warmup=1),
           "calibration": {"deployment_grade": False, "caveat": "c"}}
    v = df.assert_int8_ok(rep)
    assert v["passed"] and any("c" == w for w in v["warnings"])


def test_agreement_detects_a_broken_int8_model(tmp_path):
    pytest.importorskip("onnxruntime"); pytest.importorskip("cv2")
    fp = tmp_path / "fp32.onnx"; bad = tmp_path / "bad.onnx"
    fp.write_bytes(tiny_onnx.build_conv_detector_model(seed=0))
    bad.write_bytes(tiny_onnx.build_conv_detector_model(seed=99))                 # different weights = 'wrecked' quantization
    imgs = [df.procedural_road_scene(np.random.default_rng(i), 96) for i in range(4)]
    ag = eo.tensor_agreement(str(fp), str(bad), imgs)
    rep = {"agreement": ag, "latency": {"speedup_x": 1.0}, "calibration": {"deployment_grade": True, "caveat": ""}}
    assert not df.evaluate_int8(rep)["passed"]
    with pytest.raises(AssertionError):
        df.assert_int8_ok(rep)


def test_latency_comparison_reports_both_models(tmp_path):
    pytest.importorskip("onnxruntime")
    fp, q = _models(tmp_path)
    lat = eo.latency_comparison(fp, q, runs=6, warmup=1)
    assert lat["speedup_x"] > 0 and all(k in lat for k in ("fp32", "int8"))


def test_pipeline_end_to_end_with_stand_in_quantizer(tmp_path):
    pytest.importorskip("onnxruntime"); pytest.importorskip("cv2")
    fp, q = _models(tmp_path)

    def fake_quantizer(fp32, out, calib_dir):
        assert len([f for f in os.listdir(calib_dir) if f.endswith(".jpg")]) == 50      # the real one demands >= 50 too
        with open(q, "rb") as s, open(out, "wb") as d:
            d.write(s.read())
        return {"int8_model": out}

    rep = df.run_int8_pipeline(fp, str(tmp_path / "work"), n_images=50, runs=4, agreement_images=5, quantizer=fake_quantizer)
    assert rep["verdict"]["passed"] and os.path.exists(tmp_path / "work" / "int8_report.json")
    assert rep["calibration"]["synthetic"] and not rep["calibration"]["deployment_grade"]


def test_real_quantizer_refuses_too_few_images_and_missing_onnx(tmp_path):
    pytest.importorskip("cv2")
    df.build_calibration_set([], str(tmp_path / "c"), n=5, size=64)
    with pytest.raises(ValueError):
        eo.quantize_int8_static("x.onnx", str(tmp_path / "o.onnx"), str(tmp_path / "c"))


def test_real_static_quantization_end_to_end_when_onnx_installed(tmp_path):
    pytest.importorskip("onnx"); pytest.importorskip("onnxruntime"); pytest.importorskip("cv2")
    fp = tmp_path / "fp32.onnx"
    fp.write_bytes(tiny_onnx.build_conv_detector_model(seed=0))
    rep = df.run_int8_pipeline(str(fp), str(tmp_path / "w"), n_images=50, runs=5, agreement_images=8)
    df.assert_int8_ok(rep)
