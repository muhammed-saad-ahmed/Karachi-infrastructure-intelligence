"""SYNTHETIC DEMONSTRATION DATA. Random points near Karachi's bounding box; NOT real observations."""
import numpy as np, pandas as pd
def make(n=150, seed=42):
    r = np.random.default_rng(seed)
    types = ["pothole", "waterlogging", "garbage", "road_damage", "open_drain"]
    df = pd.DataFrame({"code": [f"SYN-{i:05d}" for i in range(n)], "type": r.choice(types, n),
        "lat": r.uniform(24.82, 25.03, n), "lon": r.uniform(66.95, 67.20, n),
        "severity": r.uniform(0.2, 1, n).round(2), "n_observations": r.integers(1, 7, n),
        "evidence_confidence": r.uniform(0.3, 0.95, n).round(2), "traffic_exposure": r.uniform(0, 1, n).round(2),
        "date": pd.Timestamp("2025-01-01") + pd.to_timedelta(r.integers(0, 365, n), "D"),
        "is_synthetic": True})
    return df

def seed_fusion_demo(conn, n_posts=80, seed=11, satellite_flood_probability=0.7):
    """OFFLINE DEMO of the whole multi-modal pipeline. Everything it writes is flagged is_synthetic=1:
        * simulated community posts (MockStream)            -> osint_posts
        * simulated Sentinel-2 scenes (synthetic provider)  -> satellite_checks
        * simulated edge-CV detections at a few places      -> observations + detections
    then fuses them into the risk matrix. Real evidence is never touched or mixed in implicitly: it simply sits
    in the same tables with is_synthetic=0, and the UI/fusion flag any cell that contains synthetic items."""
    from database import db as kpdb
    from geospatial import satellite as sat
    from geospatial.gazetteer import geocode_text
    from risk_engine import fusion
    from scripts import osint_pipeline as op
    rng = np.random.default_rng(seed)
    verifier = sat.SatelliteVerifier(conn, sat.SyntheticSatelliteProvider(flood_probability=satellite_flood_probability, seed=seed))
    stats = op.run_mock_demo(conn, verifier, n_posts=n_posts, seed=seed, fuse=False)
    n_cv = 0
    for place, label in [("Korangi", "pothole"), ("Saddar", "open_manhole"), ("Lyari", "sewage_overflow"),
                         ("Clifton", "pothole"), ("Gulshan-e-Iqbal", "sewage_overflow"), ("Malir", "open_manhole")]:
        g = geocode_text(place)
        for k in range(int(rng.integers(1, 4))):       # 1-3 independent uploads, ~30 m jitter
            lat, lon = g["lat"] + rng.normal(0, 0.0002), g["lon"] + rng.normal(0, 0.0002)
            oid = kpdb.insert_observation(conn, f"SYNTHETIC-{place}-{k}.jpg", float(lat), float(lon), "image",
                                          "synthetic-demo", "n/a", float(rng.uniform(0.55, 0.9)), True, "demo")
            kpdb.insert_detection(conn, oid, label, float(rng.uniform(0.55, 0.9)), [0, 0, 10, 10])
            n_cv += 1
    cells = fusion.assess_all(conn)
    return {"osint": stats.as_dict(), "synthetic_cv_observations": n_cv, "cells": len(cells)}


if __name__ == "__main__":
    make().to_csv("data/demo/synthetic_incidents.csv", index=False); print("written")
