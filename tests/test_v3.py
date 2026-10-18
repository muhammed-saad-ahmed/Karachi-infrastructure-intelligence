"""Round-3 tests: LSMA + shadow mask, Monte-Carlo sensitivity / Metropolis calibration, semantic text analysis,
local differential privacy on report coordinates. Offline; no onnx package, no network."""
import datetime as dt
import itertools
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import config
from database import db as kpdb
from evaluation import simulation as sim
from geospatial import gazetteer as gz
from geospatial import satellite as sat
from risk_engine import fusion as fu
from scripts import osint_text as nlp
from scripts import osint_pipeline as op

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 10, 15, 12, 0, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path):
    return kpdb.connect(str(tmp_path / "v3.db"))


# ============================================================ 1. LSMA (fully constrained unmixing)
def test_fcls_recovers_known_mixtures_and_respects_constraints():
    E, names = sat.endmember_matrix()
    rng = np.random.default_rng(0)
    frac = rng.dirichlet(np.ones(E.shape[1]), size=200)
    pix = frac @ E.T
    est = sat.fcls_unmix(pix, E)
    ab = est[0] if isinstance(est, tuple) else est
    assert np.allclose(ab.sum(axis=1), 1.0, atol=1e-6) and (ab >= -1e-9).all()
    assert np.abs(ab - frac).max() < 1e-4


def test_fcls_matches_slsqp_on_noisy_pixels():
    pytest.importorskip("scipy")
    from scipy import optimize as sp
    E, _ = sat.endmember_matrix()
    rng = np.random.default_rng(1)
    pix = rng.dirichlet(np.ones(E.shape[1]), size=15) @ E.T + rng.normal(0, 0.01, (15, E.shape[0]))
    est = sat.fcls_unmix(pix, E)
    ab = est[0] if isinstance(est, tuple) else est
    m = E.shape[1]
    for p, a in zip(pix, ab):
        r = sp.minimize(lambda x: np.sum((E @ x - p) ** 2), np.ones(m) / m, method="SLSQP", bounds=[(0, 1)] * m,
                        constraints=[{"type": "eq", "fun": lambda x: x.sum() - 1}], options={"ftol": 1e-12, "maxiter": 300})
        assert np.sum((E @ a - p) ** 2) <= np.sum((E @ r.x - p) ** 2) + 1e-9


def test_endmember_validation():
    with pytest.raises(ValueError):
        sat.endmember_matrix({"a": (.1, .1, .1, .1), "b": (.2, .2, .2, .2)})          # collinear
    with pytest.raises(ValueError):
        sat.endmember_matrix({"a": (.1, .1, .1), "b": (.2, .2, .1)})                  # wrong band count


# ============================================================ 1b. shadow mask
def _scene(n=60):
    return {"green": np.full((n, n), .10, np.float32), "red": np.full((n, n), .10, np.float32),
            "nir": np.full((n, n), .30, np.float32), "swir": np.full((n, n), .25, np.float32),
            "scl": np.full((n, n), 5, np.uint8)}


def test_shadow_mask_flags_thin_shadows_not_big_water_bodies():
    s = _scene()
    # concrete (bright) with a thin dark strip: a building shadow
    for b in ("green", "red", "nir", "swir"):
        s[b][:] = .22
    for b, v in zip(("green", "red", "nir", "swir"), (.02, .02, .03, .02)):
        s[b][20:22, 5:40] = v
    # a large, genuinely water-like body elsewhere
    for b, v in zip(("green", "red", "nir", "swir"), (.07, .05, .02, .01)):
        s[b][35:55, 35:55] = v
    sh = sat.detect_building_shadows(s)["combined"]
    assert sh[20:22, 10:35].mean() > .8                 # the shadow strip is caught
    assert sh[40:50, 40:50].mean() < .1                 # the pond core is not


def test_shadows_alone_do_not_confirm_flood_but_real_flood_survives(conn):
    now = NOW
    kw = dict(building_shadows=0.9, now_fn=lambda: now)
    lat, lon = 24.9215, 67.0925
    v_off = sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider(flood_probability=0.0, **kw),
                                  now_fn=lambda: now, method="ndwi", shadow_filter=False)
    v_on = sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider(flood_probability=0.0, **kw),
                                 now_fn=lambda: now, method="lsma", shadow_filter=True)
    assert v_off.verify(lat, lon, "waterlogging", force=True).verdict == sat.CONFIRMED_WET       # the failure mode
    assert v_on.verify(lat + .03, lon, "waterlogging", force=True).verdict == sat.NO_CHANGE        # fixed
    v_flood = sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider(flood_probability=1.0, **kw),
                                    now_fn=lambda: now, method="lsma", shadow_filter=True)
    assert v_flood.verify(lat + .06, lon, "waterlogging", force=True).verdict == sat.CONFIRMED_WET


def test_metrics_report_method_and_shadow_stats():
    m, _ = sat.compute_change_metrics(_scene(), [_scene()])
    assert m["method"] == "lsma" and "shadow_frac_event" in m and "fit_rmse_event_mean" in m
    assert m["support"] == 0.0


# ============================================================ 2. Monte-Carlo sensitivity + Metropolis
def _comp(n=24, seed=3, rain=None):
    cells, evs = sim.synthetic_cells(n, seed=seed, now=NOW)
    return sim.compile_cells(cells, evs, now=NOW, rainfall_mm=rain)


def test_vectorised_engine_reproduces_real_fusion():
    for rain in (None, 40.0):
        comp = _comp(rain=rain)
        out = sim.evaluate(sim._single(sim.baseline_theta(comp)[0]), comp)
        ref = comp.baseline_fusion
        assert len(ref) == comp.n_cells
        e_ref = np.array([r["evidence_confidence"] for r in ref])
        assert np.abs(out["E"][0] - e_ref).max() < 1e-3
        assert np.abs(out["impact"][0] - np.array([r["impact"] for r in ref])).max() < 1e-3
        cats = [r["matrix_category"] for r in ref]
        assert [sim._CAT_ORDER[i] for i in out["cat"][0]] == cats


def test_sensitivity_report_shapes_ci_and_determinism():
    comp = _comp(16)
    st = sim.SimulationSettings(n_iter=400, n_ladder=150, seed=11)
    r1, r2 = sim.run_sensitivity(comp, st), sim.run_sensitivity(comp, st)
    assert r1["meta"]["n_iter"] == 400 and r1["meta"]["n_cells"] == comp.n_cells
    assert r1["cells"] == r2["cells"]                                    # seeded => identical
    for c in r1["cells"]:
        assert c["E"]["lo"] <= c["E"]["mean"] <= c["E"]["hi"] and 0 <= c["E"]["lo"] and c["E"]["hi"] <= 1
        assert abs(sum(c["category_probs"].values()) - 1) < 1e-6 and 0 <= c["stability"] <= 1
    assert len(r1["loo"]["matrix"]) == comp.n_cells
    assert {"cv", "osint"} <= set(r1["loo"]["matrix"][0])
    assert 0 <= r1["stability"]["overall_category_stability"] <= 1
    assert r1["stability"]["tolerance_sigma"] is not None or r1["stability"]["tolerance_below_ladder"]


def test_report_digest_anchored_in_ledger(conn):
    comp = _comp(8)
    rep = sim.run_sensitivity(comp, sim.SimulationSettings(n_iter=200, n_ladder=100))
    d1 = sim.report_digest(rep)
    assert d1 == sim.report_digest(rep) and len(d1) >= 32
    sim.anchor_report(conn, rep)
    ok = kpdb.verify_ledger(conn) if hasattr(kpdb, "verify_ledger") else (True,)
    assert (ok[0] if isinstance(ok, tuple) else ok)


def test_metropolis_recovers_known_gaussian():
    target_mu, target_sd = np.array([1.0, -2.0]), np.array([0.5, 1.5])
    lp = lambda x: -0.5 * (((x - target_mu) / target_sd) ** 2).sum(axis=1)
    out = sim.metropolis(lp, np.zeros(2), 4000, rng=np.random.default_rng(0), n_chains=4)
    s = out["samples"].reshape(-1, 2)
    assert np.allclose(s.mean(0), target_mu, atol=0.15) and np.allclose(s.std(0), target_sd, rtol=0.2)
    assert (out["rhat"] < 1.05).all() and (out["ess"] > 200).all()


def test_calibration_runs_and_shifts_toward_the_truth():
    comp = _comp(60, seed=5)
    rng = np.random.default_rng(2)
    theta0, _ = sim.baseline_theta(comp)
    E = sim.evaluate(sim._single(theta0), comp)["E"][0]
    labels = (rng.random(len(E)) < np.clip(E, 0.02, 0.98)).astype(int)
    cal = sim.calibrate_reliabilities(comp, labels, n_steps=800, n_chains=3)
    post = cal["posterior"]
    assert {"delta_cv", "delta_osint"} == set(post) and cal["n_labelled"] == len(labels)
    for p in post.values():
        assert p["lo"] < p["mean"] < p["hi"] and 0 < p["implied_reliability"]["mean"] < 1
    assert len(cal["rhat"]) == 2 and max(cal["rhat"]) < 1.2
    with pytest.raises(Exception):
        sim.calibrate_reliabilities(comp, labels[:-3])                       # label count must match the cells


# ============================================================ 3. semantic text
def _a(t, **k):
    return nlp.analyze_text(t, **k)


def test_spelling_variants_of_pothole_all_match():
    for t in ("sadak mein gaddha hai", "road par gadha hai bohat", "gaddhe hi gaddhe hain sadak par", "big pothole on road",
              "potholes everywhere"):
        r = _a(t)
        assert r["hazard_type"] == "pothole" and r["hazard_strength"] > 0.3, (t, r)
    assert _a("pani jama hai gali mein")["hazard_type"] == "waterlogging"
    assert _a("gutter ubal raha hai")["hazard_type"] == "sewage_overflow"


def test_negation_roman_urdu_and_english():
    for t in ("koi gaddha nahi hai", "there is no pothole here", "pani jama nahi hai", "not flooded at all"):
        r = _a(t)
        assert r["hazard_strength"] == 0.0 and r["negated"], (t, r)
    assert not _a("pani jama hai, koi nahi sunta")["negated"]            # 'nahi' belongs to the next clause
    assert _a("manhole ka dhakkan nahi hai")["hazard_type"] == "open_manhole"   # seed contains the negator: hazard


def test_intensity_scales_strength():
    base = _a("pothole on the road")["hazard_strength"]
    assert _a("very huge pothole on the road")["hazard_strength"] >= base
    assert _a("slight pothole on the road")["hazard_strength"] <= base


def test_sarcasm_detected_and_damps_urgency():
    s = _a("wah kya sadak hai, gaddhe hi gaddhe")
    assert s["sarcasm"] and s["negativity"] >= 0.6
    assert not _a("the road is nice but potholes everywhere")["sarcasm"]
    fixed = _a("thanks for fixing the pothole, it has been repaired")
    assert fixed["status"] == "resolved" and not fixed["sarcasm"]


def test_backend_reported_and_never_raises_on_garbage():
    r = _a("")
    assert r["hazard_type"] is None and r["backend"] == nlp.get_embedder().name
    for t in (None, 123, "!!!???", "a" * 5000, "گڑھا ہے سڑک پر"):
        assert isinstance(_a(t), dict)
    assert _a("گڑھا ہے سڑک پر")["hazard_type"] == "pothole"


def test_mean_pool_ignores_padding():
    h = np.array([[[1., 1.], [3., 3.], [100., 100.]]]); m = np.array([[1, 1, 0]])
    assert np.allclose(nlp.mean_pool(h, m), [[2., 2.]])


def test_seed_bank_runtime_additions_and_validation():
    b = nlp.SeedBank({"pothole": [("pothole", .8)]})
    b.add("pothole", "khaddah", .6)
    b.add("pothole", "KHADDAH", .6)                                      # duplicate ignored (case-insensitive)
    assert len(b.phrases("pothole")) == 2
    with pytest.raises(ValueError):
        b.add("pothole", "", .5)
    with pytest.raises(ValueError):
        b.add("pothole", "x", 1.5)


def test_onnx_embedder_with_stub_encoder_and_injected_tokenizer(tmp_path):
    pytest.importorskip("onnxruntime")
    import tiny_onnx
    vocab = ["<pad>", "pothole", "gaddha", "flood", "pani", "hello", "road"]
    table = np.eye(len(vocab), 8, dtype=np.float32) + 0.01
    table[2] = table[1]                                                  # 'gaddha' embeds exactly like 'pothole'
    path = tmp_path / "model.onnx"
    path.write_bytes(tiny_onnx.build_gather_encoder_model(table))

    class Enc:
        def __init__(self, ids): self.ids, self.attention_mask = ids, [1] * len(ids)

    class Tok:
        def encode_batch(self, texts):
            return [Enc([vocab.index(w) if w in vocab else 5 for w in t.lower().split()] or [0]) for t in texts]

    emb = nlp.OnnxSentenceEmbedder(str(path), tokenizer=Tok())
    v = emb.encode(["pothole", "gaddha", "flood"])
    assert emb.name == "onnx-sentence-encoder" and v.shape == (3, 8)
    assert np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-5)
    assert v[0] @ v[1] > .999 and v[0] @ v[2] < .5                       # cross-lingual match comes from the model
    r = nlp.analyze_text("gaddha", embedder=emb)
    assert r["backend"] == "onnx-sentence-encoder" and r["hazard_type"] == "pothole"


def test_broken_onnx_embedder_falls_back_to_hashed():
    class Boom(nlp.Embedder):
        name, dim = "onnx-sentence-encoder", 4
        def encode(self, texts): raise RuntimeError("boom")
    r = nlp.analyze_text("pothole on the road", embedder=Boom())
    assert r["backend"] == "hashed-char-ngram" and r["hazard_type"] == "pothole"


# ============================================================ 5. local differential privacy
def test_lambert_w_matches_scipy():
    pytest.importorskip("scipy")
    from scipy import special as sp
    xs = np.array([-0.3678, -0.3, -0.1, -1e-3, -1e-8])
    got = np.array([gz._lambert_w_minus1(x) for x in xs])
    assert np.allclose(got, sp.lambertw(xs, k=-1).real, rtol=1e-6)


def test_radius_quantiles_and_privacy_parameters():
    g = gz.GeoPrivacy.from_level(math.log(4), 100)
    assert abs(g.epsilon - math.log(4) / 100) < 1e-12 and abs(g.expected_displacement_m - 2 / g.epsilon) < 1e-9
    assert g.radius_at(0.5) < g.radius_at(0.95) < g.cap_radius_m
    # CDF of planar Laplace radius: 1 - (1 + eps r) exp(-eps r)
    for q in (.1, .5, .9, .99):
        r = g.radius_at(q)
        assert abs(1 - (1 + g.epsilon * r) * math.exp(-g.epsilon * r) - q) < 1e-6
    for bad in (0, -1, float("nan"), "x"):
        with pytest.raises(ValueError):
            gz.GeoPrivacy(bad)


def test_noise_distribution_ks_and_mean_displacement():
    g = gz.GeoPrivacy(0.01, cap_quantile=1.0)
    rng = np.random.default_rng(5)
    d = []
    for _ in range(4000):
        la, lo, info = gz.privatize_point(24.9, 67.05, g, rng=rng)
        d.append(info["displacement_m"])
    d = np.sort(d)
    cdf = 1 - (1 + g.epsilon * d) * np.exp(-g.epsilon * d)
    ks = np.max(np.maximum(np.arange(1, len(d) + 1) / len(d) - cdf, cdf - np.arange(len(d)) / len(d)))
    assert ks < 0.03 and abs(d.mean() - 2 / g.epsilon) < 8


def test_sticky_noise_is_deterministic_per_author_and_spot():
    geo = {"lat": 24.9, "lon": 67.05, "radius_m": 30, "method": "explicit"}
    a = gz.privatize_location(geo, author_hash="aaaa", secret="s")
    b = gz.privatize_location(geo, author_hash="aaaa", secret="s")
    c = gz.privatize_location(geo, author_hash="bbbb", secret="s")
    assert (a["lat"], a["lon"]) == (b["lat"], b["lon"]) and (a["lat"], a["lon"]) != (c["lat"], c["lon"])
    assert a["method"] == "explicit+ldp" and a["radius_m"] > 100


def test_privatization_scope_and_idempotence():
    cent = {"lat": 24.9, "lon": 67.05, "radius_m": 900, "method": "gazetteer"}
    assert gz.privatize_location(cent, author_hash="x") == cent                         # centroids untouched
    done = {"lat": 24.9, "lon": 67.05, "radius_m": 500, "method": "explicit+ldp"}
    assert gz.privatize_location(done, author_hash="x") == done                         # never noised twice
    syn = {"lat": 24.9, "lon": 67.05, "radius_m": 30, "method": "explicit"}
    assert gz.privatize_location(syn, author_hash="x", is_synthetic=True) == syn        # synthetic exempt by default
    assert gz.privatize_location({"lat": float("nan"), "lon": 67, "radius_m": 30, "method": "explicit"},
                                 author_hash="x") is None                               # bad coordinate dropped


def test_eps_geo_indistinguishability_density_ratio_bound():
    # planar-Laplace density ratio for two true points d apart is <= exp(eps * d) at every output point
    g = gz.GeoPrivacy(0.02, cap_quantile=1.0)
    p1, p2, d = np.array([0., 0.]), np.array([40., 0.]), 40.0
    dens = lambda z, p: math.exp(-g.epsilon * np.linalg.norm(z - p))
    for z in itertools.product(np.linspace(-300, 300, 13), repeat=2):
        z = np.array(z); assert dens(z, p1) / dens(z, p2) <= math.exp(g.epsilon * d) + 1e-9


def test_clustering_still_usable_under_noise():
    rng = np.random.default_rng(9)
    g = gz.GeoPrivacy.from_config()
    centres = [(24.86, 67.00), (24.95, 67.15)]
    pts = []
    for ci, (la, lo) in enumerate(centres):
        for _ in range(60):
            n_la, n_lo, _ = gz.privatize_point(la, lo, g, rng=rng)
            pts.append((ci, n_la, n_lo))
    for ci, (la, lo) in enumerate(centres):
        mine = np.array([(p[1], p[2]) for p in pts if p[0] == ci])
        assert abs(mine[:, 0].mean() - la) * 111_000 < 120 and abs(mine[:, 1].mean() - lo) * 100_000 < 120   # centroid survives
    a = np.array([(p[1], p[2]) for p in pts if p[0] == 0]); b = np.array([(p[1], p[2]) for p in pts if p[0] == 1])
    assert a[:, 0].max() < b[:, 0].min()                                                # hot-spots stay separable


def test_redact_coordinates():
    t = "flooded at 24.86012, 67.00123 see https://maps.example/?q=24.86012,67.00123 pls"
    r = gz.redact_coordinates(t)
    assert "24.86012" not in r and "[location]" in r
    assert gz.redact_coordinates("call 021 3456789 on 12.5 km road") == "call 021 3456789 on 12.5 km road"


def test_db_guard_privatizes_precise_coordinates_and_leaves_schema(conn, monkeypatch):
    monkeypatch.setattr(config, "PRIVACY_LDP_ENABLED", True)
    base = dict(platform="x", url="https://m.example/?q=24.86012,67.00123", author_hash="abcd" * 4, posted_at=NOW.isoformat(),
                scraped_at=NOW.isoformat(), text="pothole at 24.86012, 67.00123", lang="en", hazard_type="pothole",
                hazard_strength=.8, urgency=0, negativity=0, community_signal=.6, status="active", keywords=[],
                lat=24.86012, lon=67.00123, geo_method="explicit", geo_radius_m=30.0, content_sha256="h", is_synthetic=False)
    kpdb.insert_osint_post(conn, post_id="1", **base)
    base_syn = dict(base, is_synthetic=True)
    kpdb.insert_osint_post(conn, post_id="2", **base_syn)
    df = kpdb.fetch_osint_posts(conn).set_index("post_id")
    assert df.loc["1", "geo_method"] == "explicit+ldp" and (df.loc["1", "lat"], df.loc["1", "lon"]) != (24.86012, 67.00123)
    assert "24.86012" not in df.loc["1", "text"] and "24.86012" not in str(df.loc["1", "url"])
    assert df.loc["1", "geo_radius_m"] > 100
    assert df.loc["2", "geo_method"] == "explicit" and df.loc["2", "lat"] == 24.86012             # synthetic exempt


def test_pipeline_geocodes_from_raw_text_but_stores_no_coordinates(conn, monkeypatch):
    import asyncio
    monkeypatch.setattr(config, "PRIVACY_LDP_ENABLED", True)
    monkeypatch.setattr(config, "PRIVACY_APPLY_TO_SYNTHETIC", True)
    pipe = op.OSINTPipeline(conn, fuse=False)
    raw = op.RawPost("mock", "q1", "Open manhole very dangerous at 24.86012, 67.00123 children", author="u1",
                     posted_at=NOW, is_synthetic=True)
    asyncio.run(pipe.process(raw, op.PipelineStats()))
    row = kpdb.fetch_osint_posts(conn).iloc[0]
    assert row["geo_method"] == "explicit+ldp" and "24.86012" not in row["text"]
    assert abs(row["lat"] - 24.86012) * 111_000 < 700                                           # within the noise cap
