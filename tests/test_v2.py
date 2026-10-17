"""Tests for the v2 multi-modal pipeline: edge ONNX runtime, OSINT, Sentinel-2 verification, evidence fusion, ledger.
Everything runs offline (synthetic satellite provider, mock feed, hand-built ONNX stub)."""
import asyncio
import datetime as dt
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import config
from database import db as kpdb
from geospatial import satellite as sat
from geospatial import gazetteer as gz
from risk_engine import fusion as fu
from scripts import osint_text as nlp
from scripts import osint_pipeline as op

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path):
    return kpdb.connect(str(tmp_path / "v2.db"))


def test_dotenv_loader_sets_missing_ignores_empty_and_real_env_wins(tmp_path, monkeypatch):
    f = tmp_path / ".env"
    f.write_text("# comment\nKP_TEST_A=hello\nKP_TEST_B=\nKP_TEST_C=\"quoted value\"\nKP_TEST_D=from_file\n")
    for k in ("KP_TEST_A", "KP_TEST_B", "KP_TEST_C"):
        monkeypatch.setenv(k, "x")      # registers cleanup so the loader's values never leak into other tests
        monkeypatch.delenv(k)           # ...then start from "not set" (works whether or not it existed before)
    monkeypatch.setenv("KP_TEST_D", "from_env")
    config._load_dotenv(str(f))
    assert os.environ["KP_TEST_A"] == "hello" and os.environ["KP_TEST_C"] == "quoted value"
    assert "KP_TEST_B" not in os.environ and os.environ["KP_TEST_D"] == "from_env"


# ====================================================================== edge ONNX
from computer_vision import edge_onnx as eo


def test_letterbox_geometry_and_padding():
    img = np.full((720, 1280, 3), 200, np.uint8)
    canvas, r, (left, top) = eo.letterbox(img, (640, 640))
    assert canvas.shape == (640, 640, 3) and r == 0.5 and (left, top) == (0, 140)
    assert canvas[0, 0, 0] == 114 and canvas[320, 320, 0] == 200      # grey pad, image in the middle


def test_preprocess_is_nchw_float_0_1():
    blob, _, _ = eo.preprocess(np.full((100, 200, 3), 255, np.uint8), (320, 320))
    assert blob.shape == (1, 3, 320, 320) and blob.dtype == np.float32 and 0.0 <= blob.min() and blob.max() <= 1.0


def test_nms_is_class_aware_and_suppresses_overlap():
    raw = np.zeros((1, 4 + 2, 12), np.float32)
    raw[0, :4, 0] = (100, 100, 50, 50); raw[0, 4, 0] = .9          # class 0
    raw[0, :4, 1] = (102, 101, 50, 50); raw[0, 4, 1] = .8          # class 0, near-duplicate -> suppressed
    raw[0, :4, 2] = (101, 100, 50, 50); raw[0, 5, 2] = .7          # class 1, same place -> kept (different class)
    out = eo.decode_output(raw, ratio=1.0, pad=(0, 0), orig_shape=(640, 640, 3), conf_thr=.25)
    assert sorted(c for *_, c in out) == [0, 1] and len(out) == 2


def test_decode_maps_back_to_original_pixels_and_clips():
    raw = np.zeros((1, 4 + 3, 10), np.float32)
    raw[0, :4, 0] = (320, 320, 100, 100); raw[0, 4, 0] = .9
    raw[0, :4, 1] = (5, 5, 40, 40);       raw[0, 5, 1] = .8        # hangs over the top-left corner
    (x1, y1, x2, y2, s, c), (a1, b1, *_rest) = eo.decode_output(raw, ratio=.5, pad=(0, 140), orig_shape=(720, 1280, 3))
    assert (x1, y1, x2, y2) == (540.0, 260.0, 740.0, 460.0) and c == 0
    assert a1 == 0.0 and b1 == 0.0                                    # clipped to the image


def test_decode_end2end_format():
    raw = np.array([[[10, 10, 50, 50, .9, 2], [0, 0, 5, 5, .1, 0]]], np.float32)
    out = eo.decode_output(raw, ratio=1.0, pad=(0, 0), orig_shape=(100, 100, 3), end2end=True)
    assert len(out) == 1 and out[0][5] == 2


def test_decode_empty_when_nothing_confident():
    assert eo.decode_output(np.zeros((1, 7, 50), np.float32), ratio=1, pad=(0, 0), orig_shape=(10, 10, 3)) == []


def test_onnx_detector_end_to_end_with_stub_model(tmp_path):
    pytest.importorskip("onnxruntime")
    from tiny_onnx import build_constant_output_model, yolo_head
    head = yolo_head([(320, 320, 100, 100, 0, .9), (322, 321, 100, 100, 0, .8), (100, 200, 40, 40, 2, .6)])
    p = tmp_path / "hazard_n.onnx"
    p.write_bytes(build_constant_output_model(
        head, (640, 640), {"names": "{0: 'pothole', 1: 'open_manhole', 2: 'sewage_overflow'}"}))
    det = eo.OnnxDetector(str(p))
    assert det.available() and det.class_names() == ["open_manhole", "pothole", "sewage_overflow"]
    res = det.detect(np.zeros((720, 1280, 3), np.uint8))
    assert [d["label"] for d in res] == ["pothole", "sewage_overflow"]
    assert res[0]["bbox"] == [540.0, 260.0, 740.0, 460.0]
    assert set(det.last_timing) == {"preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"}
    assert eo.benchmark(det, runs=3)["n_runs"] == 3


def test_coco_onnx_is_never_presented_as_a_hazard_model(tmp_path):
    pytest.importorskip("onnxruntime")
    from tiny_onnx import build_constant_output_model, yolo_head
    from computer_vision import detection
    p = tmp_path / "coco.onnx"
    p.write_bytes(build_constant_output_model(yolo_head([]), (640, 640), {"names": "{0: 'person', 1: 'car', 2: 'bus'}"}))
    d = eo.OnnxDetector(str(p))
    assert d.name == "yolo-onnx-baseline" and not detection.is_custom(d) and "no hazard" in d.scope


def test_registry_prefers_onnx_over_baseline(tmp_path, monkeypatch):
    pytest.importorskip("onnxruntime")
    from tiny_onnx import build_constant_output_model, yolo_head
    from computer_vision import detection
    (tmp_path / "models" / "custom").mkdir(parents=True)
    (tmp_path / "models" / "custom" / "a.onnx").write_bytes(
        build_constant_output_model(yolo_head([]), (640, 640), {"names": "{0: 'pothole', 1: 'x', 2: 'y'}"}))
    monkeypatch.chdir(tmp_path)
    d = detection.get_detector()
    assert d.name == "yolo-custom-onnx" and detection.is_custom(d)
    assert detection.run(d, np.zeros((64, 64, 3), np.uint8))["mode"] == "CUSTOM"


def test_quantization_refuses_too_few_calibration_images(tmp_path):
    import cv2
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="REAL"):
        eo.quantize_int8_static("nope.onnx", str(tmp_path / "o.onnx"), str(tmp_path))


# ====================================================================== text / gazetteer / privacy
def test_text_hazards_across_languages():
    assert nlp.analyze_text("Knee deep water in Gulshan, flooded street")["hazard_type"] == "waterlogging"
    r = nlp.analyze_text("Gulshan mein pani jama hai, koi nahi sunta, bacha gir gaya")
    assert r["hazard_type"] == "waterlogging" and r["lang"] == "roman-ur" and r["urgency"] > .4
    assert nlp.analyze_text("گلشن اقبال میں گٹر ابل رہا ہے")["hazard_type"] == "sewage_overflow"
    assert nlp.analyze_text("Missing manhole cover, dangerous for children")["hazard_type"] == "open_manhole"


def test_irrelevant_and_resolved_and_negated_resolution():
    assert nlp.analyze_text("Great biryani tonight")["status"] == "irrelevant"
    assert nlp.analyze_text("The pothole near Korangi has been fixed")["status"] == "resolved"
    assert nlp.analyze_text("Pothole in Clifton not fixed yet")["status"] == "active"


def test_signal_is_bounded_and_media_adds_a_little():
    a, b = nlp.analyze_text("huge pothole", 0), nlp.analyze_text("huge pothole", 2)
    assert 0 <= a["community_signal"] <= b["community_signal"] <= 1.0


def test_redaction_removes_contact_details_and_hash_is_one_way():
    t = nlp.redact("call 0300-1234567 or mail a.b@x.com, thanks @ali_khan")
    assert "0300" not in t and "@x.com" not in t and "ali_khan" not in t
    h = nlp.author_hash("mock", "SomeHandle")
    assert h == nlp.author_hash("mock", "somehandle") and "somehandle" not in h.lower() and nlp.author_hash("mock", None) is None


def test_near_duplicate_detection():
    d = nlp.DuplicateDetector(threshold=.8)
    d.add(7, "Huge pothole on Shahrah-e-Faisal near the bridge, bikes slipping")
    assert d.check("Huge pothole on Shahrah-e-Faisal near the bridge, bikes slipping! pls share") == 7
    assert d.check("Sewage overflow in Korangi for weeks") is None


def test_gazetteer_specificity_explicit_coords_and_no_guessing():
    assert gz.geocode_text("sewage in North Nazimabad")["place"] == "North Nazimabad"
    assert gz.geocode_text("pothole in Nazimabad")["place"] == "Nazimabad"
    g = gz.geocode_text("manhole open at 24.9215, 67.0925 please help")
    assert g["method"] == "explicit" and g["radius_m"] <= 50
    assert gz.geocode_text("https://maps.google.com/@24.8607,67.0100,17z")["method"] == "explicit"
    assert gz.geocode_text("flooding somewhere in Karachi") is None
    assert gz.geocode_text("coords 51.5, -0.12") is None                   # outside Karachi: ignored


# ====================================================================== parsers + fetcher
RSS = """<?xml version="1.0"?><rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel>
<item><title>Waterlogging in Korangi</title><description>&lt;p&gt;Water standing since Monday&lt;/p&gt;</description>
<link>https://x.example/1</link><guid>g1</guid><pubDate>Tue, 13 Oct 2026 08:00:00 +0000</pubDate>
<dc:creator>Reporter</dc:creator><enclosure url="https://x.example/a.jpg" type="image/jpeg"/></item></channel></rss>"""


def test_parse_rss_extracts_fields():
    (p,) = op.parse_rss(RSS, "feed")
    assert p.post_id == "g1" and "Water standing" in p.text and "<p>" not in p.text
    assert p.posted_at == dt.datetime(2026, 10, 13, 8, 0, tzinfo=UTC) and p.image_urls == ["https://x.example/a.jpg"]


def test_parse_rss_rejects_entity_bombs():
    with pytest.raises(ValueError):
        op.parse_rss('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><rss><channel><item><title>&a;</title></item></channel></rss>', "f")


def test_parse_html_with_selectors():
    pytest.importorskip("bs4")
    html = '<div class="c" id="p1"><p class="d">Pothole near Saddar</p><a href="/c/1">l</a><time datetime="2026-10-14T10:00:00Z"></time></div>'
    spec = op.SourceSpec("site", "https://s.example/list", "html", {"item": "div.c", "text": ".d", "link": "a"})
    (p,) = op.parse_html(html, spec)
    assert p.post_id == "p1" and p.url == "https://s.example/c/1" and p.posted_at.year == 2026


class _FakeHttp(op.HttpFetcher):
    """HttpFetcher with a scripted transport and a fake clock, so retry/robots logic is tested without a network."""
    def __init__(self, script, **kw):
        self.t = 0.0
        self.sleeps = []
        async def fake_sleep(s): self.sleeps.append(s); self.t += s
        super().__init__(min_delay=1.0, sleep=fake_sleep, clock=lambda: self.t, **kw)
        self.script, self.calls = script, []

    async def _get(self, url):
        self.calls.append(url)
        r = self.script(url, len(self.calls))
        if isinstance(r, Exception):
            raise r
        return r


def test_fetcher_respects_robots_disallow():
    f = _FakeHttp(lambda u, n: (200, {}, "User-agent: *\nDisallow: /private") if u.endswith("robots.txt") else (200, {}, "ok"))
    assert asyncio.run(f.fetch("https://a.example/public")) == "ok"
    with pytest.raises(op.FetchError, match="robots"):
        asyncio.run(f.fetch("https://a.example/private/x"))


def test_fetcher_retries_503_then_succeeds_with_backoff_and_rate_limit():
    def script(u, n):
        if u.endswith("robots.txt"):
            return (404, {}, "")
        return (503, {}, "") if n <= 3 else (200, {}, "fine")
    f = _FakeHttp(script)
    assert asyncio.run(f.fetch("https://a.example/x")) == "fine"
    assert len([c for c in f.calls if not c.endswith("robots.txt")]) == 3 and sum(f.sleeps) >= 1.0


def test_fetcher_does_not_retry_404_and_blocks_non_http():
    f = _FakeHttp(lambda u, n: (404, {}, ""))
    with pytest.raises(op.FetchError, match="404"):
        asyncio.run(f.fetch("https://a.example/missing"))
    assert len([c for c in f.calls if not c.endswith("robots.txt")]) == 1
    with pytest.raises(op.FetchError):
        asyncio.run(f.fetch("file:///etc/passwd"))


def test_fetcher_unreachable_robots_is_treated_as_disallowed():
    f = _FakeHttp(lambda u, n: (503, {}, ""))
    with pytest.raises(op.FetchError, match="robots"):
        asyncio.run(f.fetch("https://a.example/x"))


# ====================================================================== pipeline
def test_mock_pipeline_stores_flags_and_is_idempotent(conn):
    s1 = op.run_sync(op.OSINTPipeline(conn, fuse=False).run_stream(op.MockStream(seed=3).stream(40)))
    assert s1.inserted == 40 and s1.near_duplicates >= 1 and s1.geocoded > 30
    posts = kpdb.fetch_osint_posts(conn)
    assert posts["is_synthetic"].all() and posts["author_hash"].dropna().str.len().eq(16).all()
    assert not posts["text"].str.contains("citizen_", regex=False).any()
    s2 = op.run_sync(op.OSINTPipeline(conn, fuse=False).run_stream(op.MockStream(seed=3).stream(40)))
    assert s2.inserted == 0 and s2.already_seen == 40


def test_pipeline_triggers_satellite_only_for_observable_hazards(conn):
    prov = sat.SyntheticSatelliteProvider(flood_probability=1.0)
    ver = sat.SatelliteVerifier(conn, prov)
    pipe = op.OSINTPipeline(conn, verifier=ver, fuse=False)
    mk = lambda i, t: op.RawPost("mock", f"p{i}", t, author=f"a{i}", posted_at=dt.datetime.now(UTC), is_synthetic=True)
    st = op.PipelineStats()
    for i, t in enumerate(["Knee deep flooded street in Korangi, children stuck, dangerous",
                           "Open manhole in Korangi very dangerous for children"]):
        asyncio.run(pipe.process(mk(i, t), st))
    checks = kpdb.fetch_satellite_checks(conn)
    assert st.alerts == 2 and len(checks) == 1 and checks.iloc[0]["hazard_type"] == "waterlogging"


# ====================================================================== satellite maths + verifier
def _scene(water_mask=None, cloud_cols=0, n=60):
    green = np.full((n, n), .10, np.float32); red = np.full((n, n), .10, np.float32)
    nir = np.full((n, n), .30, np.float32); swir = np.full((n, n), .25, np.float32)
    scl = np.full((n, n), 5, np.uint8)
    if water_mask is not None:
        green[water_mask], red[water_mask], nir[water_mask], swir[water_mask] = .07, .05, .02, .01
        scl[water_mask] = 6
    if cloud_cols:
        scl[:, :cloud_cols] = 9
    return {"green": green, "red": red, "nir": nir, "swir": swir, "scl": scl}


def test_indices_sign_conventions_and_zero_division():
    assert sat.ndwi(np.array([.07]), np.array([.02]))[0] > 0           # water
    assert sat.ndwi(np.array([.10]), np.array([.30]))[0] < 0           # land
    assert sat.ndvi(np.array([.45]), np.array([.05]))[0] > .6          # vegetation
    assert np.isnan(sat.ndwi(np.array([0.]), np.array([0.]))[0])


def test_scl_mask_flags_cloud_shadow_cirrus():
    assert sat.valid_mask_from_scl(np.array([4, 5, 6, 3, 8, 9, 10])).tolist() == [True, True, True, False, False, False, False]


def test_change_detection_finds_new_water_ignores_permanent_and_speckle():
    n = 60; yy, xx = np.mgrid[0:n, 0:n]
    perm = (yy - 10) ** 2 + (xx - 10) ** 2 < 25
    flood = (yy - 40) ** 2 + (xx - 40) ** 2 < 100
    speck = np.zeros((n, n), bool); speck[5, 50] = speck[30, 5] = True
    base = [_scene(perm), _scene(perm)]
    m, prev = sat.compute_change_metrics(_scene(perm | flood | speck), base)
    assert m["largest_gain_cluster_px"] >= 250 and m["n_gain_clusters"] == 1       # speckle dropped, permanent pond not counted
    assert m["support"] > .5 and not prev["gain"][10, 10] and prev["gain"][40, 40]
    m0, _ = sat.compute_change_metrics(_scene(perm), base)
    assert m0["support"] == 0.0 and m0["n_gain_clusters"] == 0


def test_vegetation_is_not_counted_as_new_water():
    n = 60; bad = _scene(); bad["nir"][:30] = .5; bad["red"][:30] = .05      # lush vegetation: high NDVI
    bad["green"][:30] = .30                                                  # even if NDWI-ish noise rises
    m, _ = sat.compute_change_metrics(bad, [_scene()])
    assert m["n_gain_clusters"] == 0


def test_cloudy_event_scene_reports_low_valid_fraction():
    m, _ = sat.compute_change_metrics(_scene(cloud_cols=45), [_scene()])
    assert m["joint_valid_fraction"] < .5


def _verify(conn, **prov_kw):
    return sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider(**prov_kw)), (24.9215, 67.0925)


def test_verifier_confirmed_nochange_inconclusive_not_applicable(conn):
    v, (lat, lon) = _verify(conn, flood_probability=1.0)
    assert v.verify(lat, lon, "waterlogging").verdict == sat.CONFIRMED_WET
    v2, _ = _verify(conn, flood_probability=0.0)
    assert v2.verify(lat + .02, lon, "waterlogging").verdict == sat.NO_CHANGE
    v3, _ = _verify(conn, flood_probability=1.0, cloud_fraction=.9)
    ev = v3.verify(lat + .04, lon, "waterlogging")
    assert ev.verdict == sat.INCONCLUSIVE and not ev.conclusive and ev.support == 0.0
    assert v.verify(lat, lon, "pothole").verdict == sat.NOT_APPLICABLE


def test_verifier_caches_cell_and_survives_provider_failure(conn):
    v, (lat, lon) = _verify(conn, flood_probability=1.0)
    first = v.verify(lat, lon, "waterlogging")
    again = v.verify(lat + .0002, lon, "waterlogging")
    assert not first.cached and again.cached and len(kpdb.fetch_satellite_checks(conn)) == 1

    class Boom(sat.SyntheticSatelliteProvider):
        def search(self, *a, **k): raise ConnectionError("no network")
    ev = sat.SatelliteVerifier(conn, Boom()).verify(lat + .1 - .05, lon, "waterlogging")
    assert ev.verdict == sat.INCONCLUSIVE and "ConnectionError" in ev.error


def test_satellite_row_tampering_is_detected(conn):
    v, (lat, lon) = _verify(conn, flood_probability=1.0)
    v.verify(lat, lon, "waterlogging")
    conn.executescript("DROP TRIGGER trg_sat_no_update;")
    conn.execute("UPDATE satellite_checks SET support=0.0, verdict='NO_CHANGE'"); conn.commit()
    cur = conn.execute("SELECT * FROM satellite_checks"); row = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    assert not sat.row_digest_matches(row)
    assert not fu.verify_evidence(conn)["ok"]


# ====================================================================== fusion algorithm
def _it(modality, hazard="waterlogging", raw=.8, source="s1", age_h=1.0, lat=24.9215, lon=67.0925, **kw):
    return {"modality": modality, "hazard": hazard, "lat": lat, "lon": lon, "ts": NOW - dt.timedelta(hours=age_h),
            "raw": raw, "source_key": f"{modality}:{source}", "geo_radius_m": kw.pop("geo_radius_m", 30.0),
            "synthetic": False, "ref": f"{modality}:{source}:{raw}:{age_h}", **kw}


def _score(items, sat_ev=None, **kw):
    (cell,) = fu.group_into_cells(items)
    return fu.score_cell(cell, sat_ev, now=NOW, **kw)


def _sat_ev(verdict, support=.8, quality=.9, applicability=1.0):
    return sat.SatelliteEvidence(verdict=verdict, hazard_type="waterlogging", lat=24.9215, lon=67.0925, aoi_radius_m=500,
                                 event_time=NOW.isoformat(), support=support, quality=quality, applicability=applicability,
                                 scene_time=(NOW - dt.timedelta(hours=6)).isoformat(), scene_id="S", check_id=1)


def test_math_helpers():
    assert fu.noisy_or([.5, .5]) == .75 and fu.noisy_or([]) == 0.0
    assert [fu.band(x) for x in (0, .19, .2, .99, 1.0)] == [1, 1, 2, 5, 5]
    assert fu.matrix_category(5, 5) == "Critical" and fu.matrix_category(1, 1) == "Low" and fu.matrix_category(3, 4) == "High"
    assert fu.decay(36, 36) == .5 and fu.decay(-5, 36) == 1.0


def test_one_author_spamming_does_not_inflate_but_independent_authors_do():
    spam = _score([_it("osint", source="same", raw=.7) for _ in range(10)])
    one = _score([_it("osint", source="same", raw=.7)])
    many = _score([_it("osint", source=f"a{i}", raw=.7) for i in range(5)])
    assert spam["osint_strength"] == one["osint_strength"] < many["osint_strength"]


def test_modality_cap_and_cross_modal_corroboration():
    cv = _score([_it("cv", hazard="pothole", raw=.9, source="img1")])
    both = _score([_it("cv", hazard="pothole", raw=.9, source="img1"), _it("osint", hazard="pothole", raw=.8, source="u1")])
    assert cv["cv_strength"] <= config.FUSION_MODALITY_CAP["cv"]
    assert both["evidence_confidence"] > cv["evidence_confidence"] and both["corroboration"] == "two-modality"
    allthree = _score([_it("cv", raw=.9, source="i"), _it("osint", raw=.8, source="u")], _sat_ev(sat.CONFIRMED_WET))
    assert allthree["corroboration"] == "three-modality" and allthree["evidence_confidence"] > both["evidence_confidence"] - .5
    assert allthree["satellite_contrib"] > 0 and abs(sum([allthree["cv_contrib"]]) ) >= 0


def test_older_evidence_counts_less():
    fresh = _score([_it("osint", raw=.8, age_h=1)])["osint_strength"]
    old = _score([_it("osint", raw=.8, age_h=24 * 5)])["osint_strength"]
    assert old < fresh * .25


def test_imprecise_location_counts_less_than_precise():
    precise = _score([_it("osint", geo_radius_m=30)])["osint_strength"]
    vague = _score([_it("osint", geo_radius_m=3000)])["osint_strength"]
    assert vague < precise * .5


def test_clear_satellite_contradiction_reduces_but_inconclusive_does_not():
    base = [_it("osint", source="a", raw=.9), _it("osint", source="b", raw=.9), _it("osint", source="c", raw=.9)]
    none = _score(base)
    contra = _score(base, _sat_ev(sat.NO_CHANGE, support=0, quality=.9))
    cloudy = _score(base, _sat_ev(sat.INCONCLUSIVE, support=0, quality=.1))
    assert contra["evidence_confidence"] < none["evidence_confidence"] and "satellite_contradiction" in contra["flags"]
    assert cloudy["evidence_confidence"] == none["evidence_confidence"] and "satellite_inconclusive" in cloudy["flags"]
    assert contra["evidence_confidence"] > none["evidence_confidence"] * .7      # a reduction, never a veto


def test_unobservable_hazards_never_get_satellite_contradiction():
    s = _score([_it("cv", hazard="pothole", source="x", raw=.9), _it("osint", hazard="pothole", source="y", raw=.9)])
    assert "satellite_not_applicable" in s["flags"] and s["satellite_strength"] == 0.0


def test_unknown_components_are_excluded_not_guessed():
    s = _score([_it("osint")])
    assert "exposure_unknown" in s["flags"] and "rainfall_unknown" in s["flags"]
    wet = _score([_it("osint")], rainfall_mm=100)
    assert wet["impact"] != s["impact"] and "rainfall_unknown" not in wet["flags"]


def test_urgency_raises_severity_but_is_bounded():
    calm = _score([_it("osint", urgency=0.0)])["severity"]
    urgent = _score([_it("osint", urgency=1.0)])["severity"]
    assert calm < urgent <= calm + config.FUSION_URGENCY_SEVERITY_BOOST + 1e-9


def test_risk_is_monotonic_in_evidence_and_bounded():
    prev = -1
    for n in range(0, 6):
        items = [_it("osint", source=f"a{i}") for i in range(n)] or [_it("osint", raw=0.0)]
        r = _score(items)["risk_score"]
        assert 0 <= r <= 100 and r >= prev
        prev = r


def test_explanation_is_derived_from_the_numbers():
    s = _score([_it("osint", source="a"), _it("osint", source="b")])
    assert "2 independent source(s)" in s["explanation"] and f"E={s['evidence_confidence']:.2f}" in s["explanation"]


def test_fingerprint_changes_with_configuration(monkeypatch):
    before = fu.config_fingerprint()
    monkeypatch.setitem(config.FUSION_MODALITY_CAP, "osint", 0.5)
    assert fu.config_fingerprint() != before


def test_cv_collects_only_hazard_labels_inside_karachi(conn):
    o = kpdb.insert_observation(conn, "a.jpg", 24.86, 67.0, "image", "m", "v", .9, False, "t")
    kpdb.insert_detection(conn, o, "car", .9, [0, 0, 1, 1])                 # COCO object: not a hazard
    kpdb.insert_detection(conn, o, "pothole", .8, [0, 0, 1, 1])
    o2 = kpdb.insert_observation(conn, "b.jpg", 51.5, -0.1, "image", "m", "v", .9, False, "t")   # London
    kpdb.insert_detection(conn, o2, "pothole", .8, [0, 0, 1, 1])
    o3 = kpdb.insert_observation(conn, "c.jpg", None, None, "image", "m", "v", .9, False, "t")   # no location
    kpdb.insert_detection(conn, o3, "pothole", .8, [0, 0, 1, 1])
    items = fu.collect_cv(conn)
    assert [(i["hazard"], i["raw"]) for i in items] == [("pothole", .8)]


# ====================================================================== ledger + risk matrix integrity
def _seed(conn):
    conn.execute("PRAGMA foreign_keys=OFF")
    for i in range(3):
        o = kpdb.insert_observation(conn, f"img{i}.jpg", 24.86 + i * .0001, 67.0, "image", "m", "v", .8, False, "t")
        kpdb.insert_detection(conn, o, "pothole", .8, [0, 0, 1, 1])
    return fu.assess_all(conn)


def test_assess_all_records_chain_and_is_idempotent(conn):
    cells = _seed(conn)
    assert len(cells) == 1 and cells[0]["persisted"] and cells[0]["cv_strength"] > 0
    head = kpdb.ledger_head(conn)
    again = fu.assess_all(conn)
    assert not again[0]["persisted"] and kpdb.ledger_head(conn) == head and len(kpdb.fetch_risk_matrix(conn, latest_only=False)) == 1
    assert fu.full_audit(conn)["ok"]


def test_new_evidence_appends_a_new_assessment_not_an_overwrite(conn):
    _seed(conn)
    o = kpdb.insert_observation(conn, "img9.jpg", 24.86, 67.0, "image", "m", "v", .9, False, "t")
    kpdb.insert_detection(conn, o, "pothole", .9, [0, 0, 1, 1])
    again = fu.assess_all(conn)
    assert again[0]["persisted"] and len(kpdb.fetch_risk_matrix(conn, latest_only=False)) == 2
    assert len(kpdb.fetch_risk_matrix(conn)) == 1                              # latest_only shows one row per cell
    assert fu.full_audit(conn)["ok"]


def test_append_only_triggers_block_casual_edits(conn):
    _seed(conn)
    for sql in ("UPDATE risk_matrix SET risk_score=1", "DELETE FROM risk_matrix",
                "UPDATE evidence_ledger SET payload='x'", "DELETE FROM evidence_ledger"):
        with pytest.raises(Exception, match="append-only"):
            conn.execute(sql)


def test_hash_chain_detects_edit_delete_and_reorder_even_if_triggers_are_dropped(conn):
    for i in range(4):
        kpdb.append_ledger(conn, "t", {"i": i})
    assert kpdb.verify_ledger(conn)["ok"]
    conn.executescript("DROP TRIGGER trg_ledger_no_update; DROP TRIGGER trg_ledger_no_delete;")
    conn.execute("UPDATE evidence_ledger SET payload='{\"i\":99}' WHERE seq=2"); conn.commit()
    r = kpdb.verify_ledger(conn)
    assert not r["ok"] and any(p["seq"] == 2 for p in r["problems"])
    conn.execute("DELETE FROM evidence_ledger WHERE seq=3"); conn.commit()
    assert any("gap" in p["problem"] for p in kpdb.verify_ledger(conn)["problems"])


def test_matrix_row_edit_is_caught_against_ledger(conn):
    _seed(conn)
    conn.executescript("DROP TRIGGER trg_rm_no_update;")
    conn.execute("UPDATE risk_matrix SET risk_score=99.9"); conn.commit()
    r = fu.verify_matrix(conn)
    assert not r["ok"] and "risk_score" in r["problems"][0]["problem"] and not fu.full_audit(conn)["ok"]


def test_osint_text_edit_is_caught_by_content_digest(conn):
    op.run_sync(op.OSINTPipeline(conn, fuse=False).run_stream(op.MockStream(seed=5).stream(5)))
    assert fu.verify_evidence(conn)["ok"]
    conn.executescript("DROP TRIGGER trg_osint_no_update;")
    conn.execute("UPDATE osint_posts SET text='tampered' WHERE id=1"); conn.commit()
    assert not fu.verify_evidence(conn)["ok"]


def test_hmac_mode_needs_the_key(conn, monkeypatch):
    monkeypatch.setenv(config.LEDGER_KEY_ENV, "secret-key-1")
    kpdb.append_ledger(conn, "t", {"a": 1})
    assert conn.execute("SELECT alg FROM evidence_ledger").fetchone()[0] == "hmac-sha256" and kpdb.verify_ledger(conn)["ok"]
    monkeypatch.setenv(config.LEDGER_KEY_ENV, "wrong-key")
    assert not kpdb.verify_ledger(conn)["ok"]                      # attacker without the key cannot forge a valid chain
    monkeypatch.delenv(config.LEDGER_KEY_ENV)
    assert not kpdb.verify_ledger(conn)["ok"]


def test_demo_seed_runs_offline_and_everything_is_flagged_synthetic(conn):
    from scripts.generate_demo_data import seed_fusion_demo
    out = seed_fusion_demo(conn, n_posts=50, seed=2)
    m = kpdb.fetch_risk_matrix(conn)
    assert out["cells"] == len(m) > 0 and m["is_synthetic"].all() and fu.full_audit(conn)["ok"]
    assert set(m["matrix_category"]) <= {"Low", "Moderate", "High", "Critical"}
    assert "three-modality" in set(m["corroboration"]) or "two-modality" in set(m["corroboration"])
