"""Streamlit pages for the v2 multi-modal pipeline. Kept out of app.py so the app only wires pages together.

    render_risk_matrix(conn)     fused hazard risk matrix + tamper-evidence audit
    render_osint_feed(conn)      run / inspect the OSINT pipeline (offline simulator or configured sources)
    render_satellite_check(conn) one-off Sentinel-2 NDWI/NDVI cross-verification of a coordinate
    render_edge_panel(det)       ONNX edge-runtime status + benchmark (used inside Model Lab)

Every function only needs a SQLite connection (or a detector), so each can also be called from a notebook.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from app_ui.components import (confidence_bar, empty_state, error_state, metric_tile, section_header)
from app_ui.theme import ACCENT, TEXT, TEXT_FAINT, TEXT_MUTED, risk_color
from database import db as kpdb
from geospatial import satellite as sat
from risk_engine import fusion

CATEGORIES = ["Critical", "High", "Moderate", "Low"]
SYN_NOTE = ("SYNTHETIC DEMONSTRATION DATA is present: simulated posts, simulated satellite scenes and simulated "
            "CV detections are flagged is_synthetic and are not real Karachi observations.")
SOURCES_PATH = "data/osint_sources.json"


def _transparent(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
    return fig


def make_verifier(conn, mode):
    """mode: 'off' | 'synthetic' | 'live'."""
    if mode == "synthetic":
        return sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider())
    if mode == "live":
        return sat.SatelliteVerifier(conn, sat.PlanetaryComputerProvider())
    return None


def satellite_mode_picker(key, allow_off=True):
    live_ok = sat.live_provider_available()
    options = (["off"] if allow_off else []) + ["synthetic"] + (["live"] if live_ok else [])
    labels = {"off": "Off (no satellite check)", "synthetic": "Offline simulator (synthetic scenes)",
              "live": "Live Sentinel-2 via Planetary Computer (free)"}
    mode = st.selectbox("Satellite verification", options, index=min(1, len(options) - 1) if allow_off else 0,
                        format_func=labels.get, key=key)
    if not live_ok:
        st.caption("Live Sentinel-2 needs `pip install pystac-client planetary-computer rasterio`.")
    return mode


def _hist_cell(m, l, i):
    return int(((m["likelihood_band"] == l) & (m["impact_band"] == i)).sum())


# ====================================================================== risk matrix
def render_risk_matrix(conn):
    section_header("MULTI-MODAL FUSION", "Hazard Risk Matrix",
                   "Edge CV + community reports + Sentinel-2, fused by a transparent, hash-chained algorithm.")
    top = st.columns([1, 1, 1, 1])
    use_rain = top[0].checkbox("Include rainfall scenario", False)
    rain = top[1].slider("Rainfall (mm)", 0, 150, 25, disabled=not use_rain)
    if top[2].button("Re-assess now", width="stretch"):
        fusion.assess_all(conn, rainfall_mm=rain if use_rain else None)
        st.rerun()
    if top[3].button("Run offline demo", width="stretch", type="primary"):
        from scripts.generate_demo_data import seed_fusion_demo
        with st.spinner("Simulating posts, satellite scenes and CV detections, then fusing..."):
            out = seed_fusion_demo(conn)
        st.success(f"Demo complete: {out['osint']['inserted']} synthetic posts, {out['synthetic_cv_observations']} "
                   f"synthetic CV detections, {out['cells']} hazard cells.")
        st.rerun()

    m = kpdb.fetch_risk_matrix(conn)
    if m.empty:
        empty_state("NO FUSED EVIDENCE YET",
                    "Run the offline demo, run the OSINT feed, or analyse geotagged images in AI Analysis Lab. "
                    "Cells appear here once evidence exists for a hazard at a Karachi location.")
        return
    if m["is_synthetic"].any():
        st.warning(SYN_NOTE)

    audit = fusion.full_audit(conn)
    c = st.columns(4)
    with c[0]: metric_tile("Hazard cells", f"{len(m):,}")
    with c[1]: metric_tile("High + critical", f"{int(m.matrix_category.isin(['High', 'Critical']).sum()):,}",
                           accent=risk_color("High"))
    with c[2]: metric_tile("Multi-modality", f"{int(m.corroboration.isin(['two-modality', 'three-modality']).sum()):,}")
    with c[3]: metric_tile("Ledger integrity", "VERIFIED" if audit["ok"] else "BROKEN",
                           delta=f"{audit['head_seq']} entries",
                           accent=risk_color("Low") if audit["ok"] else risk_color("Critical"))
    if not audit["ok"]:
        st.error("Integrity check FAILED - stored evidence or the risk matrix no longer matches its audit trail. "
                 "See the audit panel below.")

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown('<div class="kp-eyebrow">LIKELIHOOD x IMPACT (cells per box)</div>', unsafe_allow_html=True)
        z = np.array([[l * i for l in range(1, 6)] for i in range(5, 0, -1)])
        counts = np.array([[_hist_cell(m, l, i) for l in range(1, 6)] for i in range(5, 0, -1)])
        scale = [[0.0, risk_color("Low")], [0.2, risk_color("Low")], [0.21, risk_color("Moderate")],
                 [0.45, risk_color("Moderate")], [0.46, risk_color("High")], [0.75, risk_color("High")],
                 [0.76, risk_color("Critical")], [1.0, risk_color("Critical")]]
        fig = go.Figure(go.Heatmap(z=z, x=[1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1], colorscale=scale, zmin=1, zmax=25,
                                   showscale=False, opacity=0.55, xgap=3, ygap=3,
                                   text=np.where(counts > 0, counts.astype(str), ""), texttemplate="%{text}",
                                   textfont={"size": 20, "color": TEXT}, hoverinfo="skip"))
        fig.update_xaxes(title="Likelihood (evidence strength)", dtick=1)
        fig.update_yaxes(title="Impact", dtick=1)
        st.plotly_chart(_transparent(fig, 330), width="stretch")
        st.caption("Likelihood = fused evidence strength E, banded 1-5. Impact = weighted severity/exposure/"
                   "environment, banded 1-5. Category = L x I thresholds in config.FUSION_MATRIX_CATEGORIES.")
    with right:
        st.markdown('<div class="kp-eyebrow">WHERE</div>', unsafe_allow_html=True)
        fig = px.scatter_map(m, lat="lat", lon="lon", color="matrix_category", size="risk_score", size_max=24,
                             hover_name="hazard_type", hover_data={"risk_score": True, "corroboration": True,
                                                                   "lat": False, "lon": False, "matrix_category": False},
                             color_discrete_map={k: risk_color(k) for k in CATEGORIES}, zoom=10,
                             category_orders={"matrix_category": CATEGORIES})
        fig.update_layout(map_style="carto-darkmatter", height=330, margin=dict(l=0, r=0, t=0, b=0),
                          paper_bgcolor="rgba(0,0,0,0)", legend=dict(font_color=TEXT))
        st.plotly_chart(fig, width="stretch")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">ASSESSMENTS (HIGHEST RISK FIRST)</div>', unsafe_allow_html=True)
    show = m[["id", "hazard_type", "matrix_category", "risk_score", "evidence_confidence", "cv_strength",
              "osint_strength", "satellite_strength", "corroboration", "n_independent_sources", "matrix_cell",
              "assessed_at"]].rename(columns={"id": "assessment"})
    st.dataframe(show, width="stretch", height=260)

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">WHY THIS SCORE?</div>', unsafe_allow_html=True)
    pick = st.selectbox("Assessment", m["id"].tolist(), format_func=lambda i: _label(m, i), key="fusion_pick")
    r = m[m["id"] == pick].iloc[0]
    col = risk_color(r["matrix_category"])
    st.markdown(f'<div class="kp-card" style="border-left:3px solid {col}"><b>{r["hazard_type"].replace("_", " ").title()}'
                f'</b> &nbsp;·&nbsp; <span style="color:{col}">{r["matrix_category"].upper()}</span> &nbsp;·&nbsp; '
                f'score {r["risk_score"]} &nbsp;·&nbsp; {r["matrix_cell"]}</div>', unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Evidence by modality** (capped noisy-OR strength)")
        confidence_bar("Edge CV", float(r["cv_strength"]), color="#3DD6F5")
        confidence_bar("Community reports", float(r["osint_strength"]), color="#E8C547")
        confidence_bar("Sentinel-2", float(r["satellite_strength"]), color="#3AA76D")
        confidence_bar("Fused evidence E", float(r["evidence_confidence"]), color=col)
    with b2:
        st.markdown("**How it was derived**")
        for part in str(r["explanation"]).split(" | "):
            st.markdown(f'<div style="color:{TEXT_MUTED};font-size:0.86rem;margin-bottom:3px">• {part}</div>',
                        unsafe_allow_html=True)
        flags = json.loads(r["flags"]) if r["flags"] else []
        if flags:
            st.caption("Flags: " + ", ".join(flags))
    with st.expander("Audit trail for this assessment"):
        st.code(f"ledger seq   : {r['ledger_seq']}\nledger hash  : {r['ledger_hash']}\ninputs digest: {r['inputs_digest']}\n"
                f"algorithm    : v{r['algorithm_version']}   config fingerprint: {r['config_fingerprint']}", language="text")

    with st.expander("Integrity audit (tamper-evidence)"):
        st.markdown(f"Chain: **{'OK' if audit['chain']['ok'] else 'BROKEN'}** · matrix-vs-ledger: "
                    f"**{'OK' if audit['matrix']['ok'] else 'BROKEN'}** · evidence digests: "
                    f"**{'OK' if audit['evidence']['ok'] else 'BROKEN'}**")
        for grp in ("chain", "matrix", "evidence"):
            for p in audit[grp]["problems"]:
                st.error(f"{grp}: {p}")
        st.code(f"chain head  seq={audit['head_seq']}\nchain head hash={audit['head_hash']}", language="text")
        st.caption("Publish the head hash somewhere you do not control (a git commit message, a public post) to anchor "
                   "the chain. "
                   + ("Ledger entries are HMAC-keyed (KP_LEDGER_KEY set)." if audit["chain"]["keyed"] else
                      "No KP_LEDGER_KEY set: plain SHA-256 chain - tamper-EVIDENT against edits, but whoever can rewrite "
                      "the whole file could rebuild it. Set KP_LEDGER_KEY in .env to key it."))


def _label(m, i):
    r = m[m["id"] == i].iloc[0]
    return f"#{i} · {r['hazard_type']} · {r['matrix_category']} · {r['risk_score']}"


# ====================================================================== OSINT feed
def render_osint_feed(conn):
    from scripts import osint_pipeline as op
    section_header("OSINT PIPELINE", "Community Hazard Feed",
                   "Public civic-hazard reports -> hazard, urgency, sentiment, location -> evidence.")
    st.caption("Responsible scraping only: public pages, robots.txt honoured, per-host rate limits, author handles "
               "hashed, contact details removed, images not downloaded. Most social networks forbid scraping in their "
               "terms; use RSS feeds, official complaint portals, or sources that allow reuse.")
    has_sources = os.path.exists(SOURCES_PATH)
    mode = st.radio("Source", ["Offline simulator (synthetic)"] + (["Configured sources"] if has_sources else []),
                    horizontal=True)
    if not has_sources:
        st.caption(f"To scrape real public sources, copy data/osint_sources.example.json to {SOURCES_PATH} and edit it.")
    sat_mode = satellite_mode_picker("osint_sat_mode")
    n = st.slider("Posts to process (simulator)", 10, 300, 60, step=10) if mode.startswith("Offline") else None

    if st.button("Run pipeline now", type="primary"):
        verifier = make_verifier(conn, sat_mode)
        try:
            with st.spinner("Fetching, analysing, geocoding, cross-checking..."):
                if mode.startswith("Offline"):
                    stats = op.run_mock_demo(conn, verifier, n_posts=n)
                else:
                    stats = op.run_sync(op.OSINTPipeline(conn, verifier=verifier).run_once(op.load_sources(SOURCES_PATH)))
            c = st.columns(5)
            with c[0]: metric_tile("New posts", stats.inserted)
            with c[1]: metric_tile("Geocoded", stats.geocoded)
            with c[2]: metric_tile("Near-duplicates", stats.near_duplicates)
            with c[3]: metric_tile("Alerts", stats.alerts)
            with c[4]: metric_tile("Satellite checks", stats.satellite_checks)
            for e in stats.source_errors:
                st.warning(e)
        except Exception as exc:
            error_state("The pipeline stopped before finishing.", exc)

    posts = kpdb.fetch_osint_posts(conn, limit=500)
    if posts.empty:
        empty_state("NO POSTS YET", "Run the pipeline above. The offline simulator needs no network.")
        return
    if posts["is_synthetic"].any():
        st.warning(SYN_NOTE)
    f = st.columns(3)
    hz = f[0].multiselect("Hazard", sorted(posts["hazard_type"].dropna().unique()))
    only_active = f[1].checkbox("Active only", True)
    only_geo = f[2].checkbox("Located only", False)
    v = posts
    if hz: v = v[v["hazard_type"].isin(hz)]
    if only_active: v = v[v["status"] == "active"]
    if only_geo: v = v[v["lat"].notna()]
    st.dataframe(v[["id", "posted_at", "hazard_type", "community_signal", "urgency", "negativity", "status", "lang",
                    "geo_place", "geo_method", "geo_radius_m", "duplicate_of", "text"]], width="stretch", height=380)
    st.caption("community_signal = hazard_strength x (0.75 + 0.25 x max(negativity, urgency)) (+ small media bonus). "
               "Lexicon-based, not a trained model: sarcasm, negation and Roman-Urdu spelling variants are known weaknesses. "
               "Duplicates are stored but ignored by fusion.")
    counts = v["hazard_type"].value_counts().reset_index()
    counts.columns = ["hazard", "posts"]
    if not counts.empty:
        fig = px.bar(counts, x="hazard", y="posts")
        fig.update_traces(marker_color=ACCENT)
        st.plotly_chart(_transparent(fig, 220), width="stretch")


# ====================================================================== satellite check
def render_satellite_check(conn):
    section_header("SENTINEL-2 CROSS-VERIFICATION", "Satellite Check",
                   "Does Sentinel-2 show new water or surface change where a report says there is flooding?")
    st.caption("NDWI = (Green - NIR)/(Green + NIR) finds water; NDVI = (NIR - Red)/(NIR + Red) rules out vegetation; "
               "a pixel counts as newly wet only if it became water relative to a dry-period median composite and is "
               "not vegetated. 10 m pixels cannot see potholes, manholes or narrow street flooding - those return "
               "NOT_APPLICABLE or a limited 'no change', never proof of absence. Clouds give INCONCLUSIVE, not 'dry'.")
    c = st.columns([1, 1, 1.2])
    lat = c[0].number_input("Latitude", value=24.9215, format="%.5f", min_value=24.0, max_value=26.0)
    lon = c[1].number_input("Longitude", value=67.0925, format="%.5f", min_value=66.0, max_value=68.0)
    hazard = c[2].selectbox("Hazard", ["waterlogging", "sewage_overflow", "open_drain", "pothole"])
    mode = satellite_mode_picker("sat_mode_page", allow_off=False)
    if st.button("Verify location", type="primary"):
        verifier = make_verifier(conn, mode)
        with st.spinner("Searching scenes, reading the area of interest, computing indices..."):
            ev = verifier.verify(float(lat), float(lon), hazard, force=True)
        _render_evidence(ev)
    hist = kpdb.fetch_satellite_checks(conn, limit=50)
    if not hist.empty:
        st.markdown('<div class="kp-eyebrow" style="margin-top:0.8rem">RECENT CHECKS</div>', unsafe_allow_html=True)
        st.dataframe(hist[["id", "created_at", "hazard_type", "lat", "lon", "verdict", "support", "quality",
                           "scene_id", "provider", "is_synthetic"]], width="stretch", height=240)


def _render_evidence(ev):
    colors = {sat.CONFIRMED_WET: risk_color("High"), sat.NO_CHANGE: risk_color("Low"),
              sat.INCONCLUSIVE: risk_color("Moderate"), sat.NOT_APPLICABLE: TEXT_FAINT}
    col = colors.get(ev.verdict, TEXT_MUTED)
    st.markdown(f'<div class="kp-card" style="border-left:3px solid {col}"><b style="color:{col}">{ev.verdict}</b>'
                f'{" &nbsp;·&nbsp; cached" if ev.cached else ""}'
                f'{" &nbsp;·&nbsp; SYNTHETIC SCENES" if ev.is_synthetic else ""}<br>'
                f'<span style="color:{TEXT_MUTED};font-size:0.86rem">{" ".join(ev.notes) or "-"}</span></div>',
                unsafe_allow_html=True)
    if ev.error:
        st.warning(f"Provider error: {ev.error}")
    k = st.columns(4)
    with k[0]: metric_tile("Support", f"{ev.support:.2f}", accent=col)
    with k[1]: metric_tile("Quality", f"{ev.quality:.2f}")
    with k[2]: metric_tile("Clear-sky share", "-" if ev.valid_fraction is None else f"{ev.valid_fraction:.0%}")
    with k[3]: metric_tile("Scene", ev.scene_id or "-")
    pv = ev.preview
    if pv:
        g = st.columns(3)
        for col_, key, title in ((g[0], "ndwi_base", "NDWI baseline (dry median)"), (g[1], "ndwi_event", "NDWI event scene"),
                                 (g[2], "gain", "Newly wet pixels")):
            arr = pv[key].astype(float) if key == "gain" else pv[key]
            fig = px.imshow(arr, color_continuous_scale="Blues" if key != "gain" else "Reds",
                            zmin=-0.6 if key != "gain" else 0, zmax=0.6 if key != "gain" else 1, aspect="equal")
            fig.update_layout(coloraxis_showscale=False, title=dict(text=title, font=dict(size=12)),
                              margin=dict(l=0, r=0, t=30, b=0), height=240, paper_bgcolor="rgba(0,0,0,0)", font_color=TEXT)
            fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
            col_.plotly_chart(fig, width="stretch")
    with st.expander("All measured values"):
        st.json(ev.to_dict())


# ====================================================================== edge runtime panel (Model Lab)
def render_edge_panel(det):
    from computer_vision import edge_onnx as eo
    st.markdown('<div class="kp-eyebrow" style="margin-top:0.8rem">EDGE RUNTIME — ONNX (NO SERVER, NO PYTORCH)</div>',
                unsafe_allow_html=True)
    st.caption("Export the trained YOLOv8-nano to ONNX once, then run it with onnxruntime on CPU - laptop, Raspberry Pi, "
               "Android (ONNX Runtime Mobile) or the browser (onnxruntime-web). Training/export/INT8 quantization "
               "happen on free Colab/Kaggle: `python -m scripts.train_edge_model`. No trained hazard weights ship with this repo.")
    models = eo.find_onnx_models()
    if not models:
        empty_state("NO ONNX MODEL IN models/custom/",
                    "Drop a trained <name>.onnx (or <name>.int8.onnx) there; it is picked up automatically and "
                    "takes priority over .pt files and the COCO baseline.")
        return
    pick = st.selectbox("ONNX model", models, key="edge_model_pick")
    runs = st.slider("Benchmark runs", 5, 60, 20, key="edge_runs")
    if st.button("Benchmark this ONNX model", key="edge_bench"):
        b = eo.benchmark(pick, runs=runs)
        if b is None:
            st.warning("onnxruntime is not installed.")
        else:
            c = st.columns(5)
            with c[0]: metric_tile("Mean", f"{b['mean_ms']} ms")
            with c[1]: metric_tile("p95", f"{b['p95_ms']} ms")
            with c[2]: metric_tile("Throughput", f"{b['fps']} FPS")
            with c[3]: metric_tile("Model size", f"{b['size_mb']} MB")
            with c[4]: metric_tile("Threads", str(b["threads"]))
            st.caption("Measured on THIS machine with a noise image (speed only, not accuracy). Re-run on the target device.")
