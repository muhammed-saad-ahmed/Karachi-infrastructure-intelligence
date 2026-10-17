import os, json, tempfile, datetime
import pandas as pd, plotly.express as px, streamlit as st

import config
from risk_engine.scoring import risk_score, road_health
from risk_engine.traffic import density_level
from computer_vision import preprocessing as pp, detection as dt
from computer_vision.tracking import track
from evaluation.metrics import evaluate_with_conditions, STANDARD_CONDITIONS
from scripts.generate_demo_data import make
from prediction.hotspots import recurring_hotspots, MIN_OBSERVATIONS_FOR_PREDICTION
from database.db import (connect as db_connect, insert_observation, insert_detection, fetch_observations,
                         fetch_detections, fetch_risk_matrix)
from risk_engine import fusion
from app_ui.fusion_pages import (render_risk_matrix, render_osint_feed, render_satellite_check,
                                  render_edge_panel)
from app_ui.theme import inject_global_css, risk_color, TEXT, TEXT_MUTED, TEXT_FAINT, ACCENT, VERSION
from app_ui.components import (metric_tile, confidence_bar, section_header, hero, radial_gauge,
                                pipeline_diagram, empty_state, error_state, footer, privacy_badge,
                                incident_dossier, nav_group, go_to, draw_detections, grouped_detections,
                                detection_group_row, class_color_hex)

st.set_page_config("Karachi Pulse", layout="wide", initial_sidebar_state="expanded")
inject_global_css()

NAV_GROUPS = {
    "OVERVIEW": ["Command Center"],
    "ANALYZE": ["AI Analysis Lab"],
    "INTELLIGENCE": ["Intelligence Map", "Incidents", "Road Health", "Flood Lab", "Traffic Lab"],
    "FUSION": ["Risk Matrix", "OSINT Feed", "Satellite Check"],
    "DATA": ["Analytics", "Data Explorer"],
    "RESEARCH": ["Model Lab", "Research", "About"],
}
ALL_PAGES = [p for group in NAV_GROUPS.values() for p in group]
SYN_NOTICE = "SYNTHETIC DEMONSTRATION DATA — generated for interface testing, not real Karachi observations."
LEVELS = ["Critical", "High", "Moderate", "Low"]

if "active_page" not in st.session_state:
    st.session_state.active_page = ALL_PAGES[0]


DEFAULT_BASELINE = "yolov8s.pt"
BASELINE_CHOICES = {
    "yolov8s.pt": "Balanced — small (default; ~22MB one-time download; documented higher COCO accuracy)",
    "yolov8n.pt": "Fast — nano (smallest, fastest, least accurate)",
}

@st.cache_resource
def get_cached_detector(weights=DEFAULT_BASELINE):
    return dt.get_detector(weights)

@st.cache_resource
def get_db():
    return db_connect(config.DB_PATH)


def refresh_fusion():
    """Re-fuses all evidence after new geotagged detections are saved. Never lets a fusion problem break an upload."""
    try:
        fusion.assess_all(get_db())
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Risk matrix not refreshed: {type(exc).__name__}: {exc}")


@st.cache_data
def demo():
    p = "data/demo/synthetic_incidents.csv"
    if not os.path.exists(p):
        os.makedirs("data/demo", exist_ok=True)
        make().to_csv(p, index=False)
    return pd.read_csv(p)


def scored(df, rain=0.0):
    recs = df.to_dict("records")
    res = [risk_score(x, rainfall_mm=rain) for x in recs]
    return df.assign(risk=[r["score"] for r in res], level=[r["level"] for r in res],
                      reasons=["; ".join(r["reasons"]) or "no dominant factor" for r in res],
                      risk_confidence=[r["risk_confidence"] for r in res])


def _parse_coord(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def dark_map(fig, height):
    fig.update_layout(map_style="carto-darkmatter", height=height, margin=dict(l=0, r=0, t=0, b=0),
                       paper_bgcolor="rgba(0,0,0,0)", legend=dict(font_color="#F5F7FA"))
    return fig


det = get_cached_detector(st.session_state.get("baseline_weights", DEFAULT_BASELINE))

# ---------------------------------------------------------------- sidebar shell
with st.sidebar:
    st.markdown('<div class="kp-logo-row"><div class="kp-logo-mark"></div>'
                '<div class="kp-logo">KARACHI PULSE</div></div>'
                '<div class="kp-logo-sub">AI-POWERED URBAN RISK INTELLIGENCE</div>', unsafe_allow_html=True)
    for group_title, items in NAV_GROUPS.items():
        nav_group(group_title, items, st.session_state.active_page)

    det_ok = det.available()
    st.markdown(
        f'<div class="kp-status"><div class="kp-status-title">SYSTEM STATUS</div>'
        f'<div class="kp-status-row"><span class="kp-dot" style="background:{"#3AA76D" if det_ok else TEXT_FAINT}"></span>'
        f'Vision model: {det.name}{" (baseline)" if det.name == "yolo-baseline" else ""}</div>'
        f'<div class="kp-status-row"><span class="kp-dot" style="background:{TEXT_FAINT}"></span>Data mode: synthetic demo</div>'
        f'<div class="kp-status-row"><span class="kp-dot" style="background:#3AA76D"></span>'
        f'Processing v{config.PROCESSING_VERSION}</div></div>', unsafe_allow_html=True)

page = st.session_state.active_page

# ================================================================ COMMAND CENTER
if page == "Command Center":
    hr = datetime.datetime.now().hour
    greet = "GOOD MORNING" if hr < 12 else "GOOD AFTERNOON" if hr < 18 else "GOOD EVENING"
    st.markdown(f'<div class="kp-eyebrow">{greet}</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="margin-top:0">Karachi Pulse</h1>', unsafe_allow_html=True)
    st.markdown('<div class="kp-sub">Urban intelligence overview — seeing the city through data.</div>',
                unsafe_allow_html=True)
    st.write("")
    hero()

    df = scored(demo())
    st.warning(SYN_NOTICE)
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_tile("Observations", f"{int(df.n_observations.sum()):,}")
    with c2: metric_tile("Active incidents", f"{len(df):,}")
    with c3: metric_tile("High + critical risk", f"{(df.level.isin(['High', 'Critical'])).sum():,}",
                          accent=risk_color("High"))
    with c4:
        _rm = fetch_risk_matrix(get_db())
        metric_tile("Fused hazard cells", f"{len(_rm):,}",
                    delta="from Risk Matrix" if len(_rm) else "run the demo on Risk Matrix")

    st.write("")
    cL, cR = st.columns([2, 1])
    with cL:
        section_header("CITY INTELLIGENCE", "Karachi Intelligence Map")
        fig = px.scatter_map(df, lat="lat", lon="lon", color="level",
                              color_discrete_map={l: risk_color(l) for l in LEVELS})
        st.plotly_chart(dark_map(fig, 380), width="stretch")
        st.caption("Full filtering and incident detail live on the Intelligence Map page.")
    with cR:
        section_header("QUICK ACTIONS", "")
        if st.button("ANALYZE URBAN SCENE", width="stretch", type="primary"):
            go_to("AI Analysis Lab")
        if st.button("EXPLORE KARACHI", width="stretch"):
            go_to("Intelligence Map")
        if st.button("OPEN RISK MATRIX", width="stretch"):
            go_to("Risk Matrix")
        st.markdown('<div class="kp-card" style="margin-top:0.6rem">'
                    '<div style="font-weight:700;margin-bottom:0.3rem">Research question</div>'
                    f'<div style="color:{TEXT_MUTED};font-size:0.86rem;line-height:1.5">Can low-cost smartphone '
                    'imagery, combined with environmental, temporal, traffic and geospatial data, help identify, '
                    'verify, prioritize and predict urban infrastructure risks in a resource-constrained megacity?'
                    '</div></div>', unsafe_allow_html=True)
    footer()

# ================================================================ AI ANALYSIS LAB
elif page == "AI Analysis Lab":
    section_header("ANALYZE", "AI Analysis Lab", "Turn a road image or short video into structured urban intelligence.")
    privacy_badge()
    st.caption("Detections from this page are saved to the local SQLite database "
               f"(`{config.DB_PATH}`) as real, non-synthetic observations — view them in Data Explorer. "
               "Hazard detections with coordinates also feed the fused Risk Matrix. Nothing leaves your machine.")
    st.write("")
    files = st.file_uploader("Drop an image or video, or select from device",
                              type=["jpg", "jpeg", "png", "mp4", "avi", "mov"], accept_multiple_files=True)
    blur = st.checkbox("Blur faces before analysis", True)
    if blur and not pp.face_blur_available():
        st.info("Face blur is unavailable on this OpenCV build (v5+ moved CascadeClassifier to opencv-contrib). "
                 "Analysis will continue without blurring — see Research page for details.")

    active_det = det
    if dt.is_custom(det):
        st.caption("Custom Karachi-trained model detected (" + det.name + ") — this is used automatically; the baseline selector below is hidden.")
    else:
        chosen = st.selectbox("Baseline detector", list(BASELINE_CHOICES), index=list(BASELINE_CHOICES).index(
                               st.session_state.get("baseline_weights", DEFAULT_BASELINE)),
                               format_func=lambda w: BASELINE_CHOICES[w], key="baseline_weights")
        active_det = get_cached_detector(chosen)
        st.markdown(f'<div class="kp-card kp-card-tight" style="font-size:0.82rem;color:{TEXT_MUTED}">'
                    f'<b style="color:{TEXT}">About this detector:</b> a general-purpose COCO model '
                    '(people, vehicles, everyday objects) — it is <b>not</b> trained on Karachi road hazards, '
                    'so potholes or waterlogging will not be detected, and occasional odd COCO-class labels '
                    'on unfamiliar objects are a known limitation of this baseline, not a bug. '
                    'See Model Lab and Research for details.</div>', unsafe_allow_html=True)
    c = st.columns(2)
    lat = c[0].text_input("Latitude (optional)"); lon = c[1].text_input("Longitude (optional)")

    if not files:
        empty_state("NO SCENE LOADED YET",
                     "Upload a road photo or a short clip to begin building the intelligence layer. "
                     "Nothing is analyzed until you provide an input.")
    else:
        try:
            pp.validate_count(len(files))
        except pp.InvalidInput as e:
            error_state("The number of files exceeds what this prototype can safely process at once.", e)
            st.stop()

        stage_box = st.empty()
        STAGES = ["INPUT RECEIVED", "VISION MODEL", "OBJECT DETECTION", "TEMPORAL ANALYSIS",
                  "RISK ENGINE", "GEOSPATIAL CONTEXT", "FINALIZING"]

        def show_stage(i):
            with stage_box.container():
                pipeline_diagram(STAGES, active_index=i)

        rows = []
        for i, f in enumerate(files):
            show_stage(0)
            try:
                data = f.getvalue()
                is_video = f.name.lower().endswith(("mp4", "avi", "mov"))
                pp.validate_upload(f.name, len(data), "video" if is_video else "image")
                show_stage(1)
                if is_video:
                    with tempfile.NamedTemporaryFile(suffix="." + f.name.rsplit(".", 1)[1]) as t:
                        t.write(data); t.flush(); frames = pp.sample_frames(t.name)
                    show_stage(2)
                    runs = [dt.run(active_det, x) for x in frames]
                    show_stage(3)
                    tr = track([r["detections"] for r in runs])
                    show_stage(4); show_stage(5); show_stage(6)
                    cL, cR = st.columns([1, 1])
                    with cL:
                        st.image(frames[0][:, :, ::-1], caption=f"{f.name} — first sampled frame", width="stretch")
                    with cR:
                        st.markdown(f'<div class="kp-card"><b>{len(frames)}</b> frames sampled &nbsp;·&nbsp; '
                                    f'<b>{len(tr)}</b> tracked object(s) — repeated frames merged into single incidents.</div>',
                                    unsafe_allow_html=True)
                        track_groups = grouped_detections(
                            [{"label": t_["label"], "confidence": sum(t_["confs"]) / len(t_["confs"])} for t_ in tr])
                        for g in track_groups:
                            detection_group_row(g)
                    rows += [{"file": f.name, "label": t_["label"], "frames": t_["last_frame"] - t_["first_frame"] + 1,
                              "mean_conf": round(sum(t_["confs"]) / len(t_["confs"]), 3)} for t_ in tr]
                    if tr:
                        db = get_db()
                        obs_id = insert_observation(db, source_id=f.name, lat=_parse_coord(lat), lon=_parse_coord(lon),
                                                     evidence_type="video", model=runs[0]["model"], model_version=runs[0]["version"],
                                                     confidence=sum(sum(t_["confs"]) / len(t_["confs"]) for t_ in tr) / len(tr),
                                                     is_synthetic=False, processing_version=config.PROCESSING_VERSION)
                        for t_ in tr:
                            insert_detection(db, obs_id, t_["label"], sum(t_["confs"]) / len(t_["confs"]), None)
                        if _parse_coord(lat) is not None and _parse_coord(lon) is not None:
                            refresh_fusion()
                    continue

                img = pp.decode_image(data)
                show_stage(2)
                view_img = pp.blur_faces(img) if blur else img
                r = dt.run(active_det, view_img)
                show_stage(3); show_stage(4); show_stage(5); show_stage(6)

                cL, cR = st.columns([1, 1])
                with cL:
                    view = st.radio("View", ["Original", "Detected", "Evidence"], horizontal=True, key=f"view_{i}_{f.name}")
                    disp = view_img.copy()
                    if view == "Detected":
                        disp = draw_detections(disp, r["detections"])
                    st.image(disp[:, :, ::-1], caption=f.name, width="stretch")
                    if view == "Evidence":
                        st.caption("Evidence view shows provenance below — no additional overlay is fabricated.")
                with cR:
                    st.markdown('<div class="kp-card">', unsafe_allow_html=True)
                    if r["status"] == dt.INSUFFICIENT:
                        reason = ("This baseline only detects general COCO objects (people, vehicles, everyday "
                                  "items) — hazard detection needs a custom model.") if active_det.name == "yolo-baseline" \
                                  else "No detection cleared the confidence threshold."
                        st.markdown(f'<div style="font-weight:700">No reliable visual evidence detected.</div>'
                                    f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin-top:4px">{reason}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="font-weight:700">{len(r["detections"])} object(s) detected</div>',
                                    unsafe_allow_html=True)
                        groups = grouped_detections(r["detections"])
                        for g in groups:
                            detection_group_row(g)
                        if len(r["detections"]) > len(groups):
                            with st.expander(f"All {len(r['detections'])} raw detections"):
                                for d in sorted(r["detections"], key=lambda x: -x["confidence"]):
                                    confidence_bar(d["label"].replace("_", " ").title(), d["confidence"],
                                                   color=class_color_hex(d["label"]))
                    st.markdown('</div>', unsafe_allow_html=True)
                    with st.expander("Data provenance"):
                        st.json({"model": r["model"], "version": r["version"], "scope": r["scope"],
                                 "input_resolution": r["input_resolution"], "inference_ms": r["inference_ms"],
                                 "mode": r["mode"], "processing_version": config.PROCESSING_VERSION,
                                 "data_type": "real (user upload)", "lat": lat or None, "lon": lon or None})
                rows += [{"file": f.name, **d} for d in r["detections"]]
                db = get_db()
                obs_id = insert_observation(db, source_id=f.name, lat=_parse_coord(lat), lon=_parse_coord(lon),
                                             evidence_type="image", model=r["model"], model_version=r["version"],
                                             confidence=(max((d["confidence"] for d in r["detections"]), default=0.0)),
                                             is_synthetic=False, processing_version=config.PROCESSING_VERSION)
                for d in r["detections"]:
                    insert_detection(db, obs_id, d["label"], d["confidence"], d["bbox"])
                if r["detections"] and _parse_coord(lat) is not None and _parse_coord(lon) is not None:
                    refresh_fusion()
            except pp.InvalidInput as e:
                error_state(f"{f.name} could not be read as a valid image or video.", e)
            except Exception as e:
                error_state(f"Something prevented {f.name} from being processed.", e)

        if rows:
            st.write("")
            dfres = pd.DataFrame(rows)
            st.dataframe(dfres, width="stretch")
            cc = st.columns(2)
            cc[0].download_button("Download CSV", dfres.to_csv(index=False), "karachi_pulse_report.csv", width="stretch")
            cc[1].download_button("Download JSON", json.dumps(rows, default=str), "karachi_pulse_report.json", width="stretch")
    footer()

# ================================================================ INTELLIGENCE MAP
elif page == "Intelligence Map":
    section_header("CITY INTELLIGENCE", "Karachi Intelligence Map")
    st.warning(SYN_NOTICE)
    df = scored(demo())

    f1, f2, f3 = st.columns([2, 2, 1])
    types = f1.multiselect("Category", sorted(df.type.unique()), sorted(df.type.unique()))
    levels = f2.multiselect("Risk", LEVELS, LEVELS)
    minconf = f3.slider("Min. evidence", 0.0, 1.0, 0.0)
    d = df[df.type.isin(types) & df.level.isin(levels) & (df.evidence_confidence >= minconf)].reset_index(drop=True)

    cMap, cPanel = st.columns([2, 1])
    sel_idx = None
    with cMap:
        if d.empty:
            empty_state("NO OBSERVATIONS MATCH THESE FILTERS", "Widen a filter to see incidents on the map.")
        else:
            fig = px.scatter_map(d, lat="lat", lon="lon", color="level", hover_name="code",
                                  hover_data={"type": True, "risk": True, "lat": False, "lon": False, "level": False},
                                  color_discrete_map={l: risk_color(l) for l in LEVELS}, zoom=10)
            event = st.plotly_chart(dark_map(fig, 560), width="stretch", on_select="rerun",
                                     selection_mode="points", key="map_select")
            if event and event.get("selection", {}).get("points"):
                sel_idx = event["selection"]["points"][0]["point_index"]
    with cPanel:
        if not d.empty and sel_idx is not None and sel_idx < len(d):
            row = d.iloc[sel_idx].to_dict()
            incident_dossier(row["code"], row, risk_score(row))
        else:
            empty_state("SELECT A POINT", "Click any point on the map to open its intelligence dossier.")
    footer()

# ================================================================ INCIDENTS
elif page == "Incidents":
    section_header("INCIDENT INTELLIGENCE", "Incidents")
    st.warning(SYN_NOTICE)
    df = scored(demo()).sort_values("risk", ascending=False).reset_index(drop=True)

    cL, cR = st.columns([1, 1.2])
    with cL:
        sel = st.selectbox("Incident code", df.code)
        st.dataframe(df[["code", "type", "level", "risk", "n_observations"]], width="stretch", height=440)
    with cR:
        row = df[df.code == sel].iloc[0].to_dict()
        incident_dossier(sel, row, risk_score(row))
    footer()

# ================================================================ ROAD HEALTH
elif page == "Road Health":
    section_header("ENGINEERING INSTRUMENT", "Road Health Index",
                    "Not an official municipal index — an experimental, transparent composite score.")
    weights = {k: st.sidebar.slider(k.replace("_", " ").title(), 0.0, 1.0, v, key=f"w_{k}")
               for k, v in config.ROAD_HEALTH_WEIGHTS.items()}
    comp_labels = {"surface": "Surface", "drainage": "Drainage", "cleanliness": "Cleanliness",
                   "lighting": "Lighting", "traffic_exposure": "Traffic", "safety_evidence": "Safety",
                   "recurrence": "Recurrence"}
    cL, cR = st.columns([1, 1.3])
    with cL:
        comp = {k: st.slider(v, 0, 100, 60, key=f"c_{k}") for k, v in comp_labels.items()}
    score = road_health(comp, weights)
    with cR:
        radial_gauge(score, "ROAD HEALTH", color=risk_color("Low" if score >= 50 else "High"))
        bar = px.bar(x=list(comp_labels.values()), y=[comp[k] for k in comp_labels], height=260)
        bar.update_traces(marker_color=ACCENT)
        bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#F5F7FA", margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 100])
        st.plotly_chart(bar, width="stretch")
    st.caption("Adjust weights in the sidebar to test how the composite score responds — this is the point of a transparent index.")
    with st.expander("How is this calculated?"):
        st.write("ROAD HEALTH = weighted average of the components above, using the weights in the sidebar "
                 "(defaults from config.ROAD_HEALTH_WEIGHTS). Higher component values mean a healthier road; "
                 "no component or weight is inferred from an image — these are manual research inputs.")
    footer()

# ================================================================ FLOOD LAB
elif page == "Flood Lab":
    section_header("MODEL ESTIMATE — NOT A FLOOD FORECAST", "Flood & Waterlogging Lab")
    st.warning(SYN_NOTICE + " Rainfall response uses configured sensitivity weights, unvalidated against real flood data.")
    rain = st.select_slider("Rainfall scenario (mm)", options=[10, 25, 50, 75, 100], value=25)
    d = demo(); d = d[d.type == "waterlogging"].reset_index(drop=True)
    cur = scored(d, rain)

    cL, cR = st.columns([1, 1.4])
    with cL:
        radial_gauge(round(cur.risk.mean(), 1), "ESTIMATED WATERLOGGING RISK", color=risk_color("Moderate"))
        metric_tile("Affected zones (risk ≥ Moderate)", int((cur.level.isin(["Moderate", "High", "Critical"])).sum()),
                    accent=risk_color("Moderate"))
    with cR:
        curve = pd.DataFrame([{"rain_mm": mm, "mean_risk": scored(d, mm).risk.mean()} for mm in [0, 10, 25, 50, 75, 100]])
        fig = px.line(curve, x="rain_mm", y="mean_risk", markers=True, height=260)
        fig.update_traces(line_color=ACCENT)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
        st.caption("Mean estimated risk across all waterlogging zones, recomputed live for each rainfall level.")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">WATERLOGGING RISK MAP AT THIS SCENARIO</div>', unsafe_allow_html=True)
    if cur.empty:
        empty_state("NO WATERLOGGING ZONES IN THE DEMO SET", "Nothing to map at this scenario.")
    else:
        mfig = px.scatter_map(cur, lat="lat", lon="lon", color="level", hover_name="code",
                               color_discrete_map={l: risk_color(l) for l in LEVELS}, zoom=10)
        st.plotly_chart(dark_map(mfig, 340), width="stretch")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">AFFECTED ZONES BY RAINFALL SCENARIO</div>', unsafe_allow_html=True)
    comp = pd.DataFrame([{"rain_mm": mm, "affected_zones": int((scored(d, mm).level.isin(["Moderate", "High", "Critical"])).sum())}
                         for mm in [0, 10, 25, 50, 75, 100]])
    bfig = px.bar(comp, x="rain_mm", y="affected_zones", height=240)
    bfig.update_traces(marker_color=risk_color("Moderate"))
    bfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(bfig, width="stretch")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">AFFECTED ZONES AT THIS SCENARIO</div>', unsafe_allow_html=True)
    affected = cur[cur.level.isin(["Moderate", "High", "Critical"])][["code", "level", "risk"]].sort_values("risk", ascending=False)
    if affected.empty:
        st.caption("No zones reach Moderate risk or above at this rainfall level.")
    else:
        st.dataframe(affected, width="stretch", height=200)

    if not d.empty:
        st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">PER-ZONE RISK TRAJECTORY</div>', unsafe_allow_html=True)
        zone = st.selectbox("Zone", d.code, key="flood_zone_select")
        row = d[d.code == zone].iloc[0].to_dict()
        traj = pd.DataFrame([{"rain_mm": r, "risk": risk_score(row, rainfall_mm=r)["score"]} for r in [0, 10, 25, 50, 75, 100]])
        tfig = px.line(traj, x="rain_mm", y="risk", markers=True, height=220)
        tfig.update_traces(line_color=risk_color("High"))
        tfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(tfig, width="stretch")
        st.caption(f"How {zone}'s own risk score changes with rainfall, using its actual recorded severity, "
                   "evidence confidence and traffic exposure — only the environmental component varies here.")

    st.caption("Contributing factors: detected standing water, rainfall (this scenario), garbage accumulation and "
               "historical recurrence — combined through the same transparent risk engine used elsewhere in the app.")
    footer()

# ================================================================ TRAFFIC LAB
elif page == "Traffic Lab":
    section_header("TRAFFIC INTELLIGENCE", "Traffic Lab",
                    "Estimate vehicle density from a road photo or short clip using the active vision model.")
    st.caption("Vehicle classes come straight from the COCO baseline (car, truck, bus, motorcycle, bicycle, train). "
               "Density bands below are configured in config.py, not calibrated against measured real-world traffic — "
               "treat them as an experimental estimate, not a traffic-engineering figure.")
    tfiles = st.file_uploader("Upload a road photo or short clip", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
                               accept_multiple_files=True, key="traffic_upload")

    if not tfiles:
        empty_state("NO SCENE LOADED", "Upload a road photo or short video to estimate vehicle density here. "
                     "Nothing is computed until you provide an input.")
    else:
        for f in tfiles:
            try:
                data = f.getvalue()
                is_video = f.name.lower().endswith(("mp4", "avi", "mov"))
                pp.validate_upload(f.name, len(data), "video" if is_video else "image")
                if is_video:
                    with tempfile.NamedTemporaryFile(suffix="." + f.name.rsplit(".", 1)[1]) as t:
                        t.write(data); t.flush(); frames = pp.sample_frames(t.name)
                    counts, last_vehicles = [], []
                    for fr in frames:
                        res = dt.run(det, fr)
                        veh = [dd for dd in res["detections"] if dd["label"] in config.VEHICLE_CLASSES]
                        counts.append(len(veh)); last_vehicles = veh
                    cL, cR = st.columns([1, 1])
                    with cL:
                        st.image(draw_detections(frames[-1], last_vehicles)[:, :, ::-1],
                                 caption=f"{f.name} — last sampled frame", width="stretch")
                    with cR:
                        avg = sum(counts) / len(counts) if counts else 0
                        metric_tile("Avg. vehicles / frame", round(avg, 1))
                        metric_tile("Estimated density", density_level(round(avg)), accent=ACCENT)
                    trend = pd.DataFrame({"frame": list(range(len(counts))), "vehicles": counts})
                    trfig = px.line(trend, x="frame", y="vehicles", markers=True, height=220)
                    trfig.update_traces(line_color=ACCENT)
                    trfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                                         margin=dict(l=10, r=10, t=10, b=10))
                    st.plotly_chart(trfig, width="stretch")
                    st.caption("Real per-frame vehicle counts measured from this clip — not a calibrated traffic-flow rate.")
                else:
                    img = pp.decode_image(data)
                    res = dt.run(det, img)
                    veh = [dd for dd in res["detections"] if dd["label"] in config.VEHICLE_CLASSES]
                    cL, cR = st.columns([1, 1])
                    with cL:
                        st.image(draw_detections(img, veh)[:, :, ::-1], caption=f.name, width="stretch")
                    with cR:
                        st.markdown('<div class="kp-card">', unsafe_allow_html=True)
                        if not veh:
                            st.markdown(f'<div style="font-weight:700">No vehicles detected.</div>'
                                        f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin-top:4px">'
                                        'Either the scene has none in the detectable classes, or none cleared the '
                                        'confidence threshold.</div>', unsafe_allow_html=True)
                        else:
                            for g in grouped_detections(veh):
                                detection_group_row(g)
                            st.markdown(f'<div style="margin-top:0.6rem;font-size:0.85rem;color:{TEXT_MUTED}">'
                                        f'Total vehicles: <b style="color:{TEXT}">{len(veh)}</b> &nbsp;·&nbsp; '
                                        f'Estimated density: <b style="color:{ACCENT}">{density_level(len(veh))}</b></div>',
                                        unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            except pp.InvalidInput as e:
                error_state(f"{f.name} could not be read as a valid image or video.", e)
            except Exception as e:
                error_state(f"Something prevented {f.name} from being processed.", e)

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">WHAT THIS DOESN\'T COVER YET</div>', unsafe_allow_html=True)
    pipeline_diagram(["GROUND-TRUTH CALIBRATION", "MULTI-SESSION HISTORY", "PEAK-PERIOD ANALYSIS", "ROAD-SEGMENT COMPARISON"])
    st.caption("These need traffic data collected over many sessions and times of day — a single upload can't "
               "provide that, so this page doesn't claim to show it.")
    footer()

# ================================================================ MODEL LAB
elif page == "Analytics":
    section_header("ANALYTICS", "Analytics", "Distribution and trend analysis over the demo dataset, plus any real observations you've logged.")
    st.warning(SYN_NOTICE + " Real observations from AI Analysis Lab (if any) are charted separately and labelled.")
    df = scored(demo())

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="kp-eyebrow">HAZARD TYPE DISTRIBUTION (SYNTHETIC)</div>', unsafe_allow_html=True)
        counts = df["type"].value_counts().reset_index()
        counts.columns = ["type", "count"]
        fig1 = px.bar(counts, x="type", y="count", height=260)
        fig1.update_traces(marker_color=ACCENT)
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig1, width="stretch")
    with c2:
        st.markdown('<div class="kp-eyebrow">RISK LEVEL DISTRIBUTION (SYNTHETIC)</div>', unsafe_allow_html=True)
        lvl_counts = df["level"].value_counts().reindex(LEVELS).fillna(0).reset_index()
        lvl_counts.columns = ["level", "count"]
        fig2 = px.bar(lvl_counts, x="level", y="count", height=260)
        fig2.update_traces(marker_color=[risk_color(l) for l in lvl_counts["level"]])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                            margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig2, width="stretch")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">OBSERVATIONS OVER TIME (SYNTHETIC)</div>', unsafe_allow_html=True)
    df["date"] = pd.to_datetime(df["date"])
    trend = df.groupby(df["date"].dt.to_period("M")).size().reset_index(name="count")
    trend["date"] = trend["date"].astype(str)
    fig3 = px.line(trend, x="date", y="count", markers=True, height=220)
    fig3.update_traces(line_color=ACCENT)
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                        margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig3, width="stretch")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">RECURRING HOTSPOTS</div>', unsafe_allow_html=True)
    st.caption(f"Flags locations with {MIN_OBSERVATIONS_FOR_PREDICTION}+ recorded observations as recurring — "
               "a simple recurrence rule, not a trained forecasting model. Returns 'unavailable' honestly when "
               "there isn't enough history, rather than guessing.")
    hotspots = recurring_hotspots(df)
    if hotspots is None:
        empty_state("PREDICTION UNAVAILABLE", "Additional historical observations are required before any location can be called a recurring hotspot.")
    else:
        st.dataframe(hotspots[["code", "type", "n_observations", "level", "risk"]], width="stretch", height=220)

    real_obs = fetch_observations(get_db())
    if not real_obs.empty:
        st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">REAL OBSERVATIONS LOGGED THIS SESSION</div>', unsafe_allow_html=True)
        st.caption("From AI Analysis Lab uploads — not synthetic, not merged with the demo set above.")
        real_counts = real_obs["evidence_type"].value_counts().reset_index()
        real_counts.columns = ["evidence_type", "count"]
        st.dataframe(real_counts, width="stretch")
    footer()

# ================================================================ DATA EXPLORER
elif page == "Data Explorer":
    section_header("DATA EXPLORER", "Data Explorer", "Browse the underlying records behind every other page.")

    st.markdown('<div class="kp-eyebrow">SYNTHETIC DEMO DATASET</div>', unsafe_allow_html=True)
    st.warning(SYN_NOTICE)
    df = scored(demo())
    c1, c2, c3 = st.columns(3)
    f_type = c1.multiselect("Type", sorted(df["type"].unique()), sorted(df["type"].unique()), key="de_type")
    f_level = c2.multiselect("Level", LEVELS, LEVELS, key="de_level")
    f_search = c3.text_input("Search code", key="de_search")
    filtered = df[df["type"].isin(f_type) & df["level"].isin(f_level)]
    if f_search:
        filtered = filtered[filtered["code"].str.contains(f_search, case=False, na=False)]
    st.dataframe(filtered, width="stretch", height=320)
    st.download_button("Download filtered synthetic data (CSV)", filtered.to_csv(index=False),
                        "karachi_pulse_synthetic_export.csv")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.8rem">REAL OBSERVATIONS (FROM AI ANALYSIS LAB)</div>', unsafe_allow_html=True)
    db = get_db()
    real_obs = fetch_observations(db)
    real_dets = fetch_detections(db)
    if real_obs.empty:
        empty_state("NO REAL OBSERVATIONS YET", "Analyze an image or video in AI Analysis Lab — each run is saved here automatically, as real (non-synthetic) data.")
    else:
        st.dataframe(real_obs, width="stretch", height=240)
        st.download_button("Download real observations (CSV)", real_obs.to_csv(index=False),
                            "karachi_pulse_real_observations.csv")
        with st.expander(f"Detections behind these observations ({len(real_dets)})"):
            st.dataframe(real_dets, width="stretch")
    footer()

# ================================================================ FUSION PAGES
elif page == "Risk Matrix":
    render_risk_matrix(get_db())
    footer()

elif page == "OSINT Feed":
    render_osint_feed(get_db())
    footer()

elif page == "Satellite Check":
    render_satellite_check(get_db())
    footer()

elif page == "Model Lab":
    section_header("AI RESEARCH ENVIRONMENT", "Model Lab")
    c1, c2, c3 = st.columns(3)
    with c1: metric_tile("Model", det.name)
    with c2: metric_tile("Weights", det.version)
    with c3: metric_tile("Scope", det.scope)
    st.write("")
    if dt.is_custom(det):
        st.info("A custom model was found in models/custom/. Its class list and benchmark below reflect that model.")

    names = dt.class_names(det)
    with st.expander(f"Classes this model can recognize ({len(names)})" if names else "Classes this model can recognize"):
        st.write(", ".join(names) if names else "No model is loaded, so there is no class list to show.")

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.6rem">INFERENCE BENCHMARK — MEASURED ON YOUR MACHINE</div>', unsafe_allow_html=True)
    st.caption("Runs real inference on a synthetic random-noise image to time raw speed only — this is not an "
               "accuracy measurement, and results vary by machine (CPU/GPU).")
    if st.button("Run inference benchmark"):
        if not det.available():
            st.warning("No model is loaded — install `ultralytics` or add a custom model to benchmark.")
        else:
            with st.spinner(f"Running {config.BENCHMARK_RUNS} inference passes..."):
                bench = dt.benchmark(det)
            b1, b2, b3, b4 = st.columns(4)
            with b1: metric_tile("Mean latency", f"{bench['mean_ms']} ms")
            with b2: metric_tile("Median latency", f"{bench['median_ms']} ms")
            with b3: metric_tile("Min / Max", f"{bench['min_ms']} / {bench['max_ms']} ms")
            with b4: metric_tile("Throughput", f"{bench['fps']} FPS")

    render_edge_panel(det)

    st.markdown('<div class="kp-eyebrow" style="margin-top:0.8rem">EVALUATE AGAINST YOUR OWN LABELLED IMAGES</div>', unsafe_allow_html=True)
    st.caption("Upload images plus a ground-truth JSON and this computes REAL precision/recall/F1 — not a placeholder. "
               "Two formats both work. Plain (no error-analysis breakdown): "
               "{\"filename.jpg\": [{\"label\": \"car\", \"bbox\": [x1,y1,x2,y2]}]}. "
               "With conditions (enables the error-analysis breakdown below): "
               "{\"filename.jpg\": {\"annotations\": [{\"label\": \"car\", \"bbox\": [x1,y1,x2,y2]}], "
               "\"conditions\": [\"rain\", \"blur\"]}}.")
    st.caption("Standard condition tags: " + ", ".join(f"`{c}`" for c in STANDARD_CONDITIONS) + " — or use your own.")
    example_gt = json.dumps({
        "example_plain.jpg": [{"label": "car", "bbox": [50, 60, 200, 180]}],
        "example_with_conditions.jpg": {"annotations": [{"label": "person", "bbox": [10, 10, 80, 220]}],
                                         "conditions": ["low_light", "blur"]},
    }, indent=2)
    st.download_button("Download example ground-truth template", example_gt, "ground_truth_template.json")
    ec1, ec2 = st.columns(2)
    eval_imgs = ec1.file_uploader("Labelled images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="eval_imgs")
    eval_gt_file = ec2.file_uploader("Ground-truth JSON", type=["json"], key="eval_gt")

    if eval_imgs and eval_gt_file:
        try:
            gt = json.loads(eval_gt_file.getvalue().decode("utf-8"))
            images = {im.name: pp.decode_image(im.getvalue()) for im in eval_imgs}
            result = evaluate_with_conditions(det, images, gt)
            if result is None:
                st.warning("No filenames in your images match keys in the ground-truth JSON — check the names line up exactly.")
            else:
                o = result["overall"]
                r1, r2, r3, r4 = st.columns(4)
                with r1: metric_tile("Precision", f"{o['precision']:.0%}")
                with r2: metric_tile("Recall", f"{o['recall']:.0%}")
                with r3: metric_tile("F1", f"{o['f1']:.0%}")
                with r4: metric_tile("Mean inference", f"{result['mean_inference_ms']} ms")
                st.caption(f"Computed live on {result['n_images']} matched image(s) at IoU ≥ 0.5 — these are real "
                           "numbers from your uploaded data, not fabricated placeholders.")
                st.dataframe(pd.DataFrame(result["per_image"]), width="stretch")

                st.markdown('<div class="kp-eyebrow" style="margin-top:0.8rem">ERROR ANALYSIS — BY CONDITION</div>', unsafe_allow_html=True)
                cb = result["condition_breakdown"]
                tagged = {k: v for k, v in cb.items() if k != "unspecified"}
                if not tagged:
                    st.caption("No images were tagged with conditions — add a `\"conditions\": [...]` list per image "
                               "(see template above) to break performance down by lighting, rain, blur, occlusion, etc.")
                else:
                    cdf = pd.DataFrame([{"condition": k, "f1": v["f1"], "precision": v["precision"],
                                          "recall": v["recall"], "n_images": v["n_images"]}
                                        for k, v in tagged.items()]).sort_values("f1")
                    cfig = px.bar(cdf, x="f1", y="condition", orientation="h", height=max(180, 46 * len(cdf)),
                                  hover_data=["precision", "recall", "n_images"])
                    cfig.update_traces(marker_color=[risk_color("Critical") if v < 0.5 else risk_color("Moderate") if v < 0.8 else risk_color("Low") for v in cdf["f1"]])
                    cfig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#F5F7FA",
                                        margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 1])
                    st.plotly_chart(cfig, width="stretch")
                    st.caption("Lower F1 for a condition (red/orange) means the detector genuinely struggles there — "
                               "a real weak spot measured from your data, not a guess. Each bar's n_images is how "
                               "many of your uploaded images carried that tag; small n means low statistical confidence.")
        except (json.JSONDecodeError, pp.InvalidInput, KeyError) as e:
            error_state("The ground-truth file or images could not be read — check the JSON format matches the template.", e)
    else:
        st.caption("mAP, confusion matrix, and per-class breakdown still require a larger labelled set than a quick "
                   "upload provides — those remain NOT EVALUATED. Error analysis above becomes available the moment "
                   "you upload condition-tagged images.")
        for label in ["mAP", "Confusion matrix", "Class performance"]:
            st.markdown(f'<div class="kp-card kp-card-tight" style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span>{label}</span><span class="kp-pill">NOT EVALUATED</span></div>', unsafe_allow_html=True)
    footer()

# ================================================================ RESEARCH
elif page == "Research":
    section_header("DIGITAL RESEARCH PAPER", "Research")
    st.caption("Full write-up with sourced motivation, related work, and verified citations: "
               "`docs/research_report.md` in the project folder.")
    st.markdown('<div class="kp-card"><b>Research question</b><br>'
                f'<span style="color:{TEXT_MUTED}">Can low-cost smartphone imagery, combined with environmental, '
                'temporal, traffic, weather and geospatial data, identify, verify, prioritize and predict urban '
                'infrastructure risks in a resource-constrained megacity?</span></div>', unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="kp-eyebrow">METHODOLOGY PIPELINE</div>', unsafe_allow_html=True)
    pipeline_diagram(["CAMERA", "COMPUTER VISION", "TEMPORAL TRACKING", "GEOLOCATION",
                      "EVIDENCE FUSION", "RISK ENGINE", "URBAN INTELLIGENCE"])
    st.write("")

    methodology = open("docs/methodology.md").read()
    with st.expander("Data pipeline"):
        st.markdown(methodology)
    with st.expander("Computer vision"):
        st.write("Image and video pipelines with confidence filtering, IoU tracking and a swappable model registry (computer_vision/).")
    with st.expander("Risk engine"):
        st.write("A weighted, configurable composite (risk_engine/scoring.py) that separates AI confidence, evidence confidence and risk score, and reports uncertainty explicitly.")
    with st.expander("Geospatial model"):
        st.write("Haversine-distance clustering merges same-type observations into single incidents (geospatial/clustering.py).")
    with st.expander("Evaluation"):
        st.write("Precision/Recall/F1/IoU-matching and MAE/RMSE utilities (evaluation/metrics.py), unit-tested; require a labelled dataset to produce real numbers.")
    with st.expander("Edge runtime (ONNX)"):
        st.write("YOLOv8-nano exported to ONNX and run with onnxruntime on CPU - no PyTorch, no server (computer_vision/edge_onnx.py). "
                 "INT8 quantization needs real calibration images and must be validated; no trained hazard weights ship with the repo.")
    with st.expander("OSINT pipeline"):
        st.write("Async, polite (robots.txt, rate limits, backoff) collection of public civic-hazard reports; lexicon-based English/"
                 "Roman-Urdu/Urdu analysis; offline gazetteer geocoding; author hashing; near-duplicate suppression (scripts/osint_pipeline.py).")
    with st.expander("Satellite verification"):
        st.write("Sentinel-2 L2A via Microsoft Planetary Computer: NDWI/NDVI change against a dry-period baseline, SCL cloud masking, "
                 "explicit INCONCLUSIVE / NOT_APPLICABLE verdicts (geospatial/satellite.py).")
    with st.expander("Evidence fusion & tamper-evidence"):
        st.write("Capped, time-decayed noisy-OR across independent sources and modalities -> 5x5 likelihood x impact matrix, recorded in a "
                 "hash-chained append-only ledger (risk_engine/fusion.py, database/db.py). Tamper-EVIDENT, not tamper-proof.")
    with st.expander("Limitations"):
        st.markdown("**Limitations." + methodology.split("**Limitations.**")[-1])
    with st.expander("Future work"):
        st.write("Custom Karachi-trained detector, real annotated dataset with geographic train/test split, calibrated traffic and flood modules, and a completed research report (docs/research_report.md).")
    footer()

# ================================================================ ABOUT
else:
    section_header("ABOUT", "Karachi Pulse")
    st.markdown('<div class="kp-card">'
                '<p>An independent research prototype exploring low-cost AI for urban infrastructure intelligence.</p>'
                '<p>Developer: Muhammad Saad.</p>'
                f'<p style="color:{TEXT_MUTED}">Not affiliated with any government, municipal, or academic institution. '
                'Risk scores are experimental and are not an official safety rating.</p></div>', unsafe_allow_html=True)
    footer()
