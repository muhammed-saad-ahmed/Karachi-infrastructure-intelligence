"""Sentinel-2 satellite cross-verification for OSINT hazard alerts.

When a community report claims urban flooding / waterlogging at a Karachi coordinate, this module
asks a free satellite archive (Microsoft Planetary Computer's Sentinel-2 L2A STAC API, no API key
needed for reads) whether the ground there actually changed:

    1. Take a small square area of interest (default 1 km x 1 km, 100 x 100 pixels at 10 m).
    2. Fetch the clear-sky scene closest to the report  ->  "event" image.
    3. Fetch up to N older dry-period scenes            ->  "baseline" median composite.
    4. Mask clouds / shadows with the Scene Classification Layer (SCL).
    5. Compute NDWI (water), MNDWI (water, urban-robust), NDVI (vegetation).
    6. Count pixels that are NEWLY wet: NDWI rose above the water threshold AND by a minimum delta,
       while NDVI says "not vegetation". Speckle is removed by a minimum connected-patch size.
    7. Turn that into a transparent 0..1 `support` score and one of four verdicts:
         CONFIRMED_WET   clear imagery shows a meaningful new wet area
         NO_CHANGE       clear imagery shows no meaningful new wet area
         INCONCLUSIVE    clouds / no scene / no baseline / provider error  (NOT evidence against)
         NOT_APPLICABLE  hazard is far smaller than a 10 m pixel (pothole, manhole, ...)

Indices (all from surface reflectance):
    NDWI  = (Green - NIR)  / (Green + NIR)    McFeeters (1996), Int. J. Remote Sensing 17(7)
    MNDWI = (Green - SWIR) / (Green + SWIR)   Xu (2006),        Int. J. Remote Sensing 27(14)
    NDVI  = (NIR - Red)    / (NIR + Red)      Tucker (1979),    Remote Sensing of Environment 8(2)
Sentinel-2 mission: Drusch et al. (2012), Remote Sensing of Environment 120.

Honest limitations (also repeated in README/docs):
    * 10 m pixels cannot see a pothole, an open manhole or a single flooded street. This module can
      only corroborate area-scale waterlogging (hectares), which is why hazards get an `applicability`
      factor and unobservable ones return NOT_APPLICABLE instead of a false "no change".
    * Floods come with clouds. Optical imagery is often blocked exactly when it is needed; that yields
      INCONCLUSIVE, which fusion treats as "no information", never as contradiction. (Sentinel-1 SAR
      sees through cloud and is the natural next step - not implemented.)
    * Revisit is every few days and processing adds latency, so a report may predate the next clear scene.
    * Thresholds in config.SAT_* are expert-set and NOT calibrated against a labelled Karachi flood.

Architecture: all decisions live in pure NumPy functions + `SatelliteVerifier`, which talks to a
`SceneProvider`. `PlanetaryComputerProvider` is the live implementation (needs pystac-client,
planetary-computer, rasterio). `SyntheticSatelliteProvider` is a clearly-labelled offline simulator
used for tests and the offline demo - its output is never real imagery.

Coordinate convention: this module uses STAC order bbox = (lon_min, lat_min, lon_max, lat_max).
`config.KARACHI_BBOX` is (lat_min, lon_min, lat_max, lon_max); do not mix them.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import importlib.util
import logging
import math
import warnings

import numpy as np

import config
from database import db as kpdb
from geospatial.clustering import haversine_m

log = logging.getLogger(__name__)
UTC = dt.timezone.utc

CONFIRMED_WET = "CONFIRMED_WET"
NO_CHANGE = "NO_CHANGE"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_APPLICABLE = "NOT_APPLICABLE"
CONCLUSIVE_VERDICTS = (CONFIRMED_WET, NO_CHANGE)

# Sentinel-2 L2A Scene Classification Layer classes
SCL_WATER = 6
# 0 no data, 1 saturated/defective, 2 dark/shadow features, 3 cloud shadow,
# 8 cloud (medium prob.), 9 cloud (high prob.), 10 thin cirrus, 11 snow/ice
SCL_INVALID = (0, 1, 2, 3, 8, 9, 10, 11)

_OFFSET_CHANGE_DATE = dt.datetime(2022, 1, 25, tzinfo=UTC)  # processing baseline 04.00: BOA offset -1000 DN


# ====================================================================== pure remote-sensing maths
def normalized_difference(a, b):
    """(a - b) / (a + b), NaN where the denominator is 0 or either input is NaN."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (a - b) / (a + b)
    return np.where(np.isfinite(out), out, np.nan).astype(np.float32)


def ndwi(green, nir):
    """McFeeters NDWI: positive over open water, negative over land/vegetation."""
    return normalized_difference(green, nir)


def mndwi(green, swir):
    """Xu's modified NDWI: suppresses built-up land better than NDWI in cities."""
    return normalized_difference(green, swir)


def ndvi(nir, red):
    """Tucker NDVI: high over healthy vegetation, <= ~0.1 over water, bare soil and asphalt."""
    return normalized_difference(nir, red)


def dn_to_reflectance(dn, scale=1e-4, offset=0.0, nodata=0):
    """Sentinel-2 digital numbers -> surface reflectance, NaN at nodata, clipped to >= 0."""
    dn = np.asarray(dn)
    out = dn.astype(np.float32) * np.float32(scale) + np.float32(offset)
    out = np.clip(out, 0.0, None)
    out[dn == nodata] = np.nan
    return out


def valid_mask_from_scl(scl):
    """True where a pixel is clear enough to trust (not cloud, shadow, cirrus, snow, defective)."""
    return ~np.isin(np.asarray(scl), SCL_INVALID)


def aoi_bbox(lat, lon, radius_m):
    """Square AOI around (lat, lon) as (lon_min, lat_min, lon_max, lat_max)."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * max(math.cos(math.radians(lat)), 1e-6))
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def cell_key(hazard_type, lat, lon, step=None):
    """Cache key: hazard + ~550 m grid cell. Two reports in the same cell share one satellite check."""
    step = step or config.SAT_CELL_DEG
    return f"{hazard_type}|{round(lat / step)}|{round(lon / step)}"


def _clip01(x):
    return float(max(0.0, min(1.0, x)))


def support_score(gain_frac_filtered, largest_cluster_px):
    """Transparent satellite support in [0,1]:
         area_score  = clip(newly-wet fraction of AOI / SAT_GAIN_FULL_FRAC)
         patch_score = clip(largest connected newly-wet patch (px) / SAT_CLUSTER_FULL_PX)
         support     = 0.6 * area_score + 0.4 * patch_score
    The patch term rewards one coherent flooded area over scattered pixels (noise looks scattered)."""
    area = _clip01(gain_frac_filtered / config.SAT_GAIN_FULL_FRAC)
    patch = _clip01(largest_cluster_px / config.SAT_CLUSTER_FULL_PX)
    return 0.6 * area + 0.4 * patch, area, patch


def baseline_depth_factor(n_scenes):
    """A median of several dry scenes is a sturdier reference than a single scene."""
    return {0: 0.0, 1: 0.8, 2: 0.9}.get(int(n_scenes), 1.0)


def _masked_index(scene, which, extra_invalid=None):
    """Index array with NaN wherever the pixel is not trustworthy (SCL cloud/shadow classes, plus an optional
    extra mask such as the structural building-shadow mask). Returns (index, valid)."""
    valid = valid_mask_from_scl(scene["scl"])
    if extra_invalid is not None:
        valid = valid & ~np.asarray(extra_invalid, dtype=bool)
    if which == "ndwi":
        idx = ndwi(scene["green"], scene["nir"])
    elif which == "ndvi":
        idx = ndvi(scene["nir"], scene["red"])
    elif which == "mndwi":
        idx = mndwi(scene["green"], scene["swir"])
    else:
        raise ValueError(which)
    return np.where(valid, idx, np.nan).astype(np.float32), valid


def _nanmedian_stack(arrays):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixel columns are expected under cloud
        return np.nanmedian(np.stack(arrays), axis=0).astype(np.float32)


def _r(x, nd=4):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), nd)


# ====================================================================== sub-pixel unmixing (LSMA)
BAND_NAMES = ("green", "red", "nir", "swir")      # B03, B04, B08, B11


def endmember_matrix(spec=None):
    """-> (E, names). E has shape (n_bands, n_endmembers); column j is the spectrum of names[j].
    Raises ValueError for spectra that are negative, non-finite, wrong length or linearly dependent
    (unmixing would then be ill-posed)."""
    spec = config.SAT_ENDMEMBERS if spec is None else spec
    names = tuple(spec)
    if len(names) < 2:
        raise ValueError("need at least two endmembers")
    try:
        E = np.array([spec[n] for n in names], dtype=np.float64).T
    except (TypeError, ValueError) as exc:
        raise ValueError(f"endmember spectra must be equal-length numeric tuples: {exc}") from exc
    if E.ndim != 2 or E.shape[0] != len(BAND_NAMES):
        raise ValueError(f"each endmember needs {len(BAND_NAMES)} values (bands {BAND_NAMES})")
    if not np.isfinite(E).all() or (E < 0).any():
        raise ValueError("endmember reflectances must be finite and non-negative")
    if np.linalg.matrix_rank(E) < E.shape[1]:
        raise ValueError("endmember spectra are linearly dependent; abundances would not be unique")
    return E, names


def fcls_unmix(pixels, endmembers):
    """Fully Constrained Least Squares linear spectral unmixing:  min ||x - E a||^2  s.t.  a >= 0, sum(a) = 1.

    pixels      (n, n_bands) reflectance; rows containing NaN come back as NaN.
    endmembers  (n_bands, m) spectra as columns.
    Returns (abundances (n, m), rmse (n,)) with rmse = sqrt(mean squared residual per band).

    Exact and fully vectorised: the optimum of this convex problem has some support set S (the endmembers with
    non-zero abundance). For each of the 2^m - 1 candidate supports we solve the sum-to-one equality-constrained
    least-squares problem in closed form (Lagrange multiplier), keep the candidates whose solution is
    non-negative, and take the one with the smallest residual. For the m = 4 used here that is 15 small matrix
    products for the whole image - no per-pixel loops, no iterative solver, and no tolerance to tune.
    (Heinz & Chang 2001 give the iterative NNLS-based FCLS; this enumeration reaches the same optimum.)
    """
    X = np.atleast_2d(np.asarray(pixels, dtype=np.float64))
    E = np.asarray(endmembers, dtype=np.float64)
    if X.ndim != 2 or E.ndim != 2 or E.shape[0] != X.shape[1]:
        raise ValueError(f"pixels {X.shape} and endmembers {E.shape} are not compatible")
    n, nb = X.shape
    m = E.shape[1]
    if not 1 <= m <= 12:
        raise ValueError("exhaustive FCLS supports 1..12 endmembers")
    A = np.full((n, m), np.nan)
    rmse = np.full(n, np.nan)
    finite = np.isfinite(X).all(axis=1)
    if not finite.any():
        return A, rmse
    Xf = X[finite]
    best_res = np.full(len(Xf), np.inf)
    best_A = np.zeros((len(Xf), m))
    for mask in range(1, 1 << m):
        idx = [j for j in range(m) if (mask >> j) & 1]
        Es = E[:, idx]
        Ginv = np.linalg.pinv(Es.T @ Es)
        g1 = Ginv @ np.ones(len(idx))
        denom = float(g1.sum())
        if abs(denom) < 1e-12:
            continue
        a_u = Xf @ (Es @ Ginv)                                   # unconstrained LS on this support
        a = a_u + ((1.0 - a_u.sum(axis=1)) / denom)[:, None] * g1[None, :]   # project onto sum(a) = 1
        feasible = (a >= -1e-12).all(axis=1)
        if not feasible.any():
            continue
        res = ((Xf - a @ Es.T) ** 2).sum(axis=1)
        better = np.nonzero(feasible & (res < best_res - 1e-15))[0]
        if better.size:
            best_res[better] = res[better]
            best_A[better] = 0.0
            best_A[np.ix_(better, idx)] = np.clip(a[better], 0.0, None)
    A[finite] = best_A
    rmse[finite] = np.sqrt(best_res / nb)
    return A, rmse


def unmix_scene(scene, valid=None, endmembers=None):
    """Per-pixel fractional abundances for one scene dict.
    -> ({endmember name: float32 (H, W) abundance in [0, 1], NaN where invalid}, rmse (H, W))."""
    E, names = endmember_matrix(endmembers)
    bands = [np.asarray(scene[b], dtype=np.float64) for b in BAND_NAMES]
    h, w = bands[0].shape
    X = np.stack(bands, axis=-1).reshape(-1, len(BAND_NAMES))
    if valid is not None:
        X = X.copy()
        X[~np.asarray(valid, dtype=bool).reshape(-1)] = np.nan
    A, rmse = fcls_unmix(X, E)
    return ({n: A[:, i].reshape(h, w).astype(np.float32) for i, n in enumerate(names)},
            rmse.reshape(h, w).astype(np.float32))


# ====================================================================== building-shadow filter
def detect_building_shadows(scene, *, dark_max=None, window=None, min_std=None, bright=None, min_body_px=None):
    """Finds pixels that are dark because a building or tree blocks the sun - the classic false 'water' in dense cities.

    Water and shadow are both dark, so brightness alone cannot separate them. What differs is STRUCTURE:
    a flood is a smooth, compact body; a building shadow is a thin strip lying right next to something very bright.
    A pixel is flagged `structural` only if ALL of these hold:
        1. dark          mean reflectance over the available bands <= dark_max
        2. thin          it is NOT inside a dark body at least `min_body_px` wide (morphological opening)
        3. high variance the local brightness std in a `window` x `window` neighbourhood >= min_std
        4. next to light a pixel >= `bright` brightness (roof / concrete) lies within `min_body_px` pixels
    Sentinel-2 SCL classes 2 (dark area) and 3 (cloud shadow) are reported separately as `scl`.
    Returns dict of boolean (H, W) masks: scl, structural, combined, dark, plus float32 'local_std'.

    Limits: a genuinely flooded street narrower than min_body_px next to bright buildings is also masked (a 10 m
    pixel cannot resolve it reliably anyway). Thresholds are expert-set, not validated on labelled Karachi scenes.
    Topographic (terrain) shadow would need a DEM and the sun angles from the scene metadata - not used here.
    """
    import cv2
    dark_max = config.SAT_SHADOW_DARK_MAX if dark_max is None else dark_max
    window = int(config.SAT_SHADOW_WINDOW if window is None else window) | 1
    min_std = config.SAT_SHADOW_MIN_STD if min_std is None else min_std
    bright = config.SAT_SHADOW_BRIGHT if bright is None else bright
    body = int(config.SAT_SHADOW_MIN_BODY_PX if min_body_px is None else min_body_px)
    names = [b for b in BAND_NAMES if b in scene]
    if len(names) < 3:
        raise ValueError("shadow detection needs at least three of green/red/nir/swir")
    stack = np.stack([np.asarray(scene[b], dtype=np.float32) for b in names])
    finite = np.isfinite(stack).all(axis=0)
    shape = finite.shape
    scl = np.asarray(scene.get("scl", np.zeros(shape, np.uint8)))
    scl_shadow = np.isin(scl, (2, 3))
    if not finite.any():
        z = np.zeros(shape, bool)
        return {"scl": scl_shadow, "structural": z, "combined": scl_shadow.copy(), "dark": z,
                "local_std": np.zeros(shape, np.float32)}
    bright_map = stack.mean(axis=0)
    fill = float(np.median(bright_map[finite]))
    B = np.where(finite, bright_map, fill).astype(np.float32)
    mean = cv2.boxFilter(B, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    mean2 = cv2.boxFilter(B * B, -1, (window, window), borderType=cv2.BORDER_REFLECT)
    local_std = np.sqrt(np.maximum(mean2 - mean * mean, 0.0)).astype(np.float32)
    dark = (B <= dark_max) & finite
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (body, body))
    bulk = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, kern).astype(bool)
    reach = 2 * body + 1
    near_bright = cv2.dilate(((B >= bright) & finite).astype(np.uint8), np.ones((reach, reach), np.uint8)).astype(bool)
    structural = dark & ~bulk & (local_std >= min_std) & near_bright
    return {"scl": scl_shadow, "structural": structural, "combined": structural | scl_shadow,
            "dark": dark, "local_std": local_std}


def _clear_mask(scene, shadow_filter):
    """(clear mask, structural-shadow mask or None) for one scene."""
    valid = valid_mask_from_scl(scene["scl"])
    if not shadow_filter:
        return valid, None
    sh = detect_building_shadows(scene)["structural"]
    return valid & ~sh, sh


# ====================================================================== change metrics
def _ndwi_gain(event, baselines, ev_struct, base_structs, thr, min_delta, max_ndvi_water):
    """Index-threshold 'newly wet' mask (the original rule). Returns a dict of arrays."""
    ndwi_ev, ev_valid = _masked_index(event, "ndwi", ev_struct)
    ndvi_ev, _ = _masked_index(event, "ndvi", ev_struct)
    ndwi_base = _nanmedian_stack([_masked_index(b, "ndwi", s)[0] for b, s in zip(baselines, base_structs)])
    ndvi_base = _nanmedian_stack([_masked_index(b, "ndvi", s)[0] for b, s in zip(baselines, base_structs)])
    joint = ev_valid & np.isfinite(ndwi_ev) & np.isfinite(ndwi_base)
    gain = (joint & (ndwi_ev > thr) & (ndwi_base <= thr) & ((ndwi_ev - ndwi_base) >= min_delta)
            & np.isfinite(ndvi_ev) & (ndvi_ev <= max_ndvi_water))
    return {"ndwi_ev": ndwi_ev, "ndvi_ev": ndvi_ev, "ndwi_base": ndwi_base, "ndvi_base": ndvi_base,
            "ev_valid": ev_valid, "joint": joint, "gain": gain}


def _filter_speckle(gain, min_cluster_px):
    """Drops connected patches smaller than min_cluster_px. -> (kept mask, kept areas array)."""
    import cv2
    n_c, labels, stats, _ = cv2.connectedComponentsWithStats(gain.astype(np.uint8), connectivity=8)
    if n_c <= 1:
        return np.zeros_like(gain, dtype=bool), np.array([], dtype=int)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_ids = np.nonzero(areas >= min_cluster_px)[0] + 1
    return np.isin(labels, keep_ids), areas[areas >= min_cluster_px]


def compute_change_metrics(event, baselines, *, thr=None, min_delta=None, max_ndvi_water=None,
                           min_cluster_px=None, method=None, shadow_filter=None, endmembers=None):
    """Compares one event scene against the median of the baseline scenes.

    event / each baseline: dict with float32 reflectance arrays 'green','red','nir' (+ 'swir') and a uint8 'scl'
    array, all the same shape (a scene dict as returned by SceneProvider.read()).

    method="lsma" (default, needs 'swir'): a pixel is NEWLY WET if, after fully-constrained linear unmixing,
        its WATER abundance is >= SAT_LSMA_MIN_WATER and rose by >= SAT_LSMA_MIN_DELTA over the baseline median,
        the unmixing fits well (rmse <= SAT_LSMA_MAX_RMSE, i.e. the pixel really is a mix of the modelled
        surfaces), and it is clear in the event scene and in at least one baseline scene.
        Sub-pixel water therefore counts in proportion to its abundance, and concrete/asphalt/vegetation
        mixtures that merely lower NDWI are not mistaken for water.
    method="ndwi": the original rule - NDWI > thr, baseline NDWI <= thr, rise >= min_delta, NDVI <= max_ndvi_water.
    shadow_filter (default config.SAT_SHADOW_FILTER): pixels flagged by detect_building_shadows() are treated as
        not observable (in the event AND in every baseline scene) before anything is decided.
    Returns (metrics: dict of plain floats, preview: dict of arrays for plotting).
    """
    thr = config.SAT_NDWI_THRESHOLD if thr is None else thr
    min_delta = config.SAT_MIN_DELTA_NDWI if min_delta is None else min_delta
    max_ndvi_water = config.SAT_MAX_NDVI_WATER if max_ndvi_water is None else max_ndvi_water
    min_cluster_px = config.SAT_MIN_CLUSTER_PX if min_cluster_px is None else min_cluster_px
    shadow_filter = config.SAT_SHADOW_FILTER if shadow_filter is None else bool(shadow_filter)
    method = str(method or config.SAT_METHOD).lower()
    if method not in ("lsma", "ndwi"):
        raise ValueError(f"unknown satellite method {method!r}; use 'lsma' or 'ndwi'")
    if not baselines:
        raise ValueError("at least one baseline scene is required")
    has_swir = "swir" in event and all("swir" in b for b in baselines)
    method_used = "lsma" if (method == "lsma" and has_swir) else "ndwi"

    ev_clear, ev_struct = _clear_mask(event, shadow_filter)
    base_masks = [_clear_mask(b, shadow_filter) for b in baselines]      # (clear, structural shadow) per baseline
    base_structs = [st for _, st in base_masks]
    core = _ndwi_gain(event, baselines, ev_struct, base_structs, thr, min_delta, max_ndvi_water)
    ndwi_ev, ndwi_base, ndvi_ev, ndvi_base = core["ndwi_ev"], core["ndwi_base"], core["ndvi_ev"], core["ndvi_base"]
    ev_valid, joint = core["ev_valid"], core["joint"]
    mndwi_ev = _masked_index(event, "mndwi", ev_struct)[0] if has_swir else None
    mndwi_base = (_nanmedian_stack([_masked_index(b, "mndwi", s)[0] for b, s in zip(baselines, base_structs)])
                  if has_swir else None)

    n_px = int(ndwi_ev.size)
    n_joint = int(joint.sum())
    m = {"n_pixels": n_px, "n_valid_joint": n_joint, "method": method_used, "shadow_filter": shadow_filter,
         "valid_fraction_event": _r(float(ev_valid.mean())),
         "joint_valid_fraction": _r(n_joint / n_px if n_px else 0.0),
         "n_baseline_scenes": len(baselines)}
    if method == "lsma" and not has_swir:
        m["method_note"] = "no SWIR band available: fell back to the NDWI rule"
    if shadow_filter:
        sh_ev = detect_building_shadows(event)
        m["shadow_frac_event"] = _r(float(sh_ev["combined"].mean()))
        m["structural_shadow_frac_event"] = _r(float(sh_ev["structural"].mean()))
        naive = _ndwi_gain(event, baselines, None, [None] * len(baselines), thr, min_delta, max_ndvi_water)
        naive_kept, _ = _filter_speckle(naive["gain"], min_cluster_px)
        # how many pixels the unfiltered NDWI rule would have called "newly wet" that sit in building shadow
        m["shadow_false_positive_px"] = int((naive_kept & sh_ev["structural"]).sum())
        m["naive_ndwi_gain_frac"] = _r(float(naive_kept.sum()) / max(int(naive["joint"].sum()), 1))
    else:
        sh_ev = None

    gain = np.zeros_like(joint)
    w_ev = None
    if n_joint:
        delta = ndwi_ev - ndwi_base
        if method_used == "lsma":
            ab_ev, rmse_ev = unmix_scene(event, ev_valid, endmembers)
            ab_bs = [unmix_scene(b, clear, endmembers) for b, (clear, _) in zip(baselines, base_masks)]
            w_ev = ab_ev["water"]
            w_base = _nanmedian_stack([a["water"] for a, _ in ab_bs])
            fit_ok = np.isfinite(rmse_ev) & (rmse_ev <= config.SAT_LSMA_MAX_RMSE)
            gain = (joint & fit_ok & np.isfinite(w_ev) & np.isfinite(w_base)
                    & (w_ev >= config.SAT_LSMA_MIN_WATER) & ((w_ev - w_base) >= config.SAT_LSMA_MIN_DELTA))
            weight = np.clip(np.nan_to_num(w_ev - w_base), 0.0, 1.0)
            for nm, key in (("water", "water"), ("vegetation", "veg"), ("impervious", "imperv"), ("shade", "shade")):
                if nm in ab_ev:
                    m[f"{key}_abund_event_mean"] = _r(np.nanmean(ab_ev[nm][joint]))
            m["water_abund_base_mean"] = _r(np.nanmean(w_base[joint & np.isfinite(w_base)])) if np.isfinite(w_base[joint]).any() else None
            m["fit_rmse_event_mean"] = _r(np.nanmean(rmse_ev[joint]))
            m["poor_fit_fraction"] = _r(float((~fit_ok[joint]).mean()))
        else:
            gain = core["gain"]
            weight = np.ones(gain.shape, np.float32)
        kept_mask, kept_areas = _filter_speckle(gain, min_cluster_px)
        gain_filtered = int(kept_mask.sum())
        largest = int(kept_areas.max()) if kept_areas.size else 0
        weighted_area = float(weight[kept_mask].sum())          # water-equivalent pixels (sub-pixel aware)
        m.update({
            "ndwi_event_mean": _r(np.nanmean(ndwi_ev[joint])), "ndwi_base_mean": _r(np.nanmean(ndwi_base[joint])),
            "delta_ndwi_mean": _r(np.nanmean(delta[joint])), "delta_ndwi_p95": _r(np.nanpercentile(delta[joint], 95)),
            "ndvi_event_mean": _r(np.nanmean(ndvi_ev[joint & np.isfinite(ndvi_ev)])) if np.isfinite(ndvi_ev[joint]).any() else None,
            "ndvi_base_mean": _r(np.nanmean(ndvi_base[joint & np.isfinite(ndvi_base)])) if np.isfinite(ndvi_base[joint]).any() else None,
            "water_frac_event": _r(float((ndwi_ev[joint] > thr).mean())),
            "water_frac_base": _r(float((ndwi_base[joint] > thr).mean())),
            "water_gain_frac": _r(float(gain.sum()) / n_joint),
            "water_gain_frac_filtered": _r(gain_filtered / n_joint),
            "water_equiv_gain_frac": _r(weighted_area / n_joint),
            "n_gain_clusters": int(kept_areas.size), "largest_gain_cluster_px": largest,
            "scl_water_frac_event": _r(float((np.asarray(event["scl"])[joint] == SCL_WATER).mean())),
        })
        if m["ndvi_event_mean"] is not None and m["ndvi_base_mean"] is not None:
            m["delta_ndvi_mean"] = _r(m["ndvi_event_mean"] - m["ndvi_base_mean"])
        if has_swir:
            ok = joint & np.isfinite(mndwi_ev) & np.isfinite(mndwi_base)
            if ok.any():
                m["mndwi_event_mean"] = _r(np.nanmean(mndwi_ev[ok]))
                m["mndwi_base_mean"] = _r(np.nanmean(mndwi_base[ok]))
        support, area_s, patch_s = support_score(weighted_area / n_joint, largest)
        m.update({"area_score": _r(area_s), "patch_score": _r(patch_s), "support": _r(support)})
        gain = kept_mask                                        # what the verdict was actually based on
    else:
        m.update({"water_gain_frac": 0.0, "water_gain_frac_filtered": 0.0, "n_gain_clusters": 0,
                  "largest_gain_cluster_px": 0, "support": 0.0})

    preview = {"ndwi_event": ndwi_ev, "ndwi_base": ndwi_base, "gain": gain, "valid": joint}
    if w_ev is not None:
        preview["water_abundance_event"] = w_ev
    if sh_ev is not None:
        preview["shadow"] = sh_ev["combined"]
    return m, preview


# ====================================================================== evidence record
@dataclasses.dataclass
class SatelliteEvidence:
    verdict: str
    hazard_type: str
    lat: float
    lon: float
    aoi_radius_m: float
    event_time: str
    support: float = 0.0          # 0..1, how strongly imagery shows a new wet area
    quality: float = 0.0          # 0..1, how trustworthy the reading is (clear-sky fraction x baseline depth)
    applicability: float = 0.0    # 0..1, how observable this hazard is at 10 m
    scene_time: str | None = None
    scene_id: str | None = None
    baseline_scene_ids: list = dataclasses.field(default_factory=list)
    cloud_pct: float | None = None
    valid_fraction: float | None = None
    metrics: dict = dataclasses.field(default_factory=dict)
    provider: str = ""
    is_synthetic: bool = False
    error: str | None = None
    notes: list = dataclasses.field(default_factory=list)
    check_id: int | None = None
    cached: bool = False
    result_sha256: str = ""
    preview: dict | None = dataclasses.field(default=None, repr=False, compare=False)

    @property
    def conclusive(self):
        return self.verdict in CONCLUSIVE_VERDICTS

    def digest(self):
        """Hash of everything that defines the result; stored so later edits are detectable."""
        return kpdb.sha256_hex(kpdb.canonical_json({
            "verdict": self.verdict, "support": round(self.support, 6), "quality": round(self.quality, 6),
            "scene_id": self.scene_id, "baseline": self.baseline_scene_ids, "metrics": self.metrics}))

    def to_dict(self):
        d = dataclasses.asdict(self)
        d.pop("preview", None)
        return d


def evidence_from_row(row):
    """Rebuilds a SatelliteEvidence from a satellite_checks row (dict)."""
    import json
    metrics = json.loads(row.get("metrics_json") or "{}")
    notes = metrics.pop("notes", [])       # stored beside the metrics, but not part of the hashed result
    base = row.get("baseline_scene_ids")
    return SatelliteEvidence(
        verdict=row["verdict"], hazard_type=row["hazard_type"], lat=row["lat"], lon=row["lon"],
        aoi_radius_m=row["aoi_radius_m"], event_time=row["event_time"], support=row["support"] or 0.0,
        quality=row["quality"] or 0.0, applicability=row["applicability"] or 0.0,
        scene_time=row["scene_time"], scene_id=row["scene_id"],
        baseline_scene_ids=json.loads(base) if base else [], cloud_pct=row["cloud_pct"],
        valid_fraction=row["valid_fraction"], metrics=metrics, provider=row["provider"] or "",
        is_synthetic=bool(row["is_synthetic"]), error=row["error"], notes=notes, check_id=row["id"],
        result_sha256=row["result_sha256"] or "")


def row_digest_matches(row):
    """True if a stored satellite_checks row still hashes to the digest recorded when it was written."""
    return evidence_from_row(row).digest() == row.get("result_sha256")


def find_evidence(conn, lat, lon, hazard_type, *, max_distance_m=None, since_iso=None):
    """Best stored satellite evidence for a hazard cell (used by risk_engine.fusion).
    Prefers a conclusive check with the highest quality, then the newest. Falls back to the newest
    INCONCLUSIVE check so fusion can report 'imagery was tried but blocked'. None if nothing is near."""
    q = "SELECT * FROM satellite_checks WHERE hazard_type=?"
    args = [hazard_type]
    if since_iso:
        q += " AND created_at>=?"
        args.append(since_iso)
    with kpdb.db_lock:
        cur = conn.execute(q + " ORDER BY id DESC LIMIT 300", args)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    near = []
    for r in rows:
        reach = max(r["aoi_radius_m"] or 0.0, max_distance_m or 0.0)
        if haversine_m((lat, lon), (r["lat"], r["lon"])) <= reach:
            near.append(r)
    if not near:
        return None
    conclusive = [r for r in near if r["verdict"] in CONCLUSIVE_VERDICTS]
    pool = conclusive or near
    best = max(pool, key=lambda r: ((r["quality"] or 0.0), r["scene_time"] or "", r["id"]))
    return evidence_from_row(best)


# ====================================================================== providers
@dataclasses.dataclass
class SceneInfo:
    id: str
    time: dt.datetime                 # timezone-aware UTC acquisition time
    cloud_cover: float | None = None  # scene-level percent, if known
    extra: dict = dataclasses.field(default_factory=dict)


class SceneProvider:
    """Interface the verifier needs. read() returns reflectance arrays + SCL on a fixed size_px grid."""
    name = "base"
    is_synthetic = False

    def search(self, bbox, start, end, max_cloud):  # -> list[SceneInfo]
        raise NotImplementedError

    def read(self, scene, bbox, size_px):  # -> dict(green, red, nir, swir, scl)
        raise NotImplementedError


def live_provider_available():
    """True if the optional packages for the live Planetary Computer provider are importable."""
    return all(importlib.util.find_spec(m) is not None for m in ("pystac_client", "planetary_computer", "rasterio"))


class PlanetaryComputerProvider(SceneProvider):
    """Live Sentinel-2 L2A from Microsoft Planetary Computer (free, no key for reads).
    Needs: pip install pystac-client planetary-computer rasterio.
    Reads only the AOI window of each Cloud-Optimized GeoTIFF over HTTP, so a check transfers
    a few hundred KB, not a 1 GB scene."""
    name = "planetary-computer/sentinel-2-l2a"
    is_synthetic = False
    _BANDS = {"B03": "green", "B04": "red", "B08": "nir", "B11": "swir"}

    def __init__(self, stac_url=None, collection=None):
        self.stac_url = stac_url or config.SAT_STAC_URL
        self.collection = collection or config.SAT_COLLECTION
        self._client = None

    def _get_client(self):
        if self._client is None:
            import planetary_computer
            import pystac_client
            self._client = pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        return self._client

    def search(self, bbox, start, end, max_cloud):
        search = self._get_client().search(
            collections=[self.collection], bbox=list(bbox),
            datetime=f"{start.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}/{end.astimezone(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            query={"eo:cloud_cover": {"lt": float(max_cloud)}})
        getter = getattr(search, "item_collection", None) or search.get_all_items
        scenes = []
        for item in getter():
            t = item.datetime or dt.datetime.fromisoformat(item.properties["datetime"].replace("Z", "+00:00"))
            if t.tzinfo is None:
                t = t.replace(tzinfo=UTC)
            scenes.append(SceneInfo(id=item.id, time=t.astimezone(UTC),
                                    cloud_cover=item.properties.get("eo:cloud_cover"), extra={"item": item}))
        return scenes

    @staticmethod
    def _scale_offset(asset, when):
        rb = (getattr(asset, "extra_fields", None) or {}).get("raster:bands")
        if isinstance(rb, list) and rb and "scale" in rb[0]:
            return float(rb[0]["scale"]), float(rb[0].get("offset", 0.0))
        return 1e-4, (-0.1 if when >= _OFFSET_CHANGE_DATE else 0.0)

    @staticmethod
    def _read_window(href, bbox, size_px, nearest):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.warp import transform_bounds
        from rasterio.windows import from_bounds
        env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
                   GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_TIMEOUT="30")
        with rasterio.Env(**env), rasterio.open(href) as ds:
            left, bottom, right, top = transform_bounds("EPSG:4326", ds.crs, *bbox, densify_pts=21)
            win = from_bounds(left, bottom, right, top, transform=ds.transform)
            return ds.read(1, window=win, out_shape=(size_px, size_px), boundless=True, fill_value=0,
                           resampling=Resampling.nearest if nearest else Resampling.bilinear)

    def read(self, scene, bbox, size_px):
        import planetary_computer
        item = planetary_computer.sign(scene.extra["item"])  # refresh SAS token if the item is old
        out = {}
        for key, name in self._BANDS.items():
            asset = item.assets[key]
            scale, offset = self._scale_offset(asset, scene.time)
            dn = self._read_window(asset.href, bbox, size_px, nearest=False)
            out[name] = dn_to_reflectance(dn, scale, offset)
        out["scl"] = self._read_window(item.assets["SCL"].href, bbox, size_px, nearest=True).astype(np.uint8)
        return out


class SyntheticSatelliteProvider(SceneProvider):
    """OFFLINE SIMULATOR - not real imagery. Generates deterministic, physically-plausible band
    reflectances (land, vegetation patches, permanent water, optional flood blobs, optional cloud
    band) so the whole verification pipeline can run in tests and in the offline demo.
    Everything it produces is flagged is_synthetic=True end to end.

    flood_probability  share of cells that 'flood' in scenes at/after `wet_after`
    cloud_fraction     share of the AOI covered by cloud in those 'event' scenes (0 = clear)
    wet_after          scenes at/after this time show the flood (default: 15 days before now)
    building_shadows   0..1 density of simulated tall buildings. Each casts a thin, dark, blue-shifted shadow on
                       the concrete beside it; the shadow falls on a different side in 'event' scenes than in
                       baseline scenes (different sun geometry), and 70% of shadow pixels are mislabelled as
                       water by the simulated SCL, as Sen2Cor often does. Stylised stress test for the
                       shadow filter - real shadows are messier.
    """
    name = "synthetic-offline-simulator"
    is_synthetic = True

    def __init__(self, flood_probability=0.6, cloud_fraction=0.0, seed=7, wet_after=None,
                 cadence_days=5, latency_days=2, now_fn=None, building_shadows=0.0):
        self.building_shadows = float(building_shadows)
        self.flood_probability = float(flood_probability)
        self.cloud_fraction = float(cloud_fraction)
        self.seed = int(seed)
        self.cadence_days = int(cadence_days)
        self.latency_days = int(latency_days)
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self.wet_after = wet_after or (self._now() - dt.timedelta(days=15))

    def _rng(self, *parts):
        h = hashlib.sha256("|".join(str(p) for p in (self.seed,) + parts).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def search(self, bbox, start, end, max_cloud):
        last = min(end, self._now() - dt.timedelta(days=self.latency_days))
        day = start.astimezone(UTC).date()
        out = []
        while day <= last.astimezone(UTC).date():
            if day.toordinal() % self.cadence_days == self.seed % self.cadence_days:
                t = dt.datetime(day.year, day.month, day.day, 6, 20, tzinfo=UTC)
                cloud = float(self._rng("cloud", day.toordinal()).integers(0, 30))
                if cloud < max_cloud and start <= t <= last:
                    out.append(SceneInfo(id=f"SYN-{day:%Y%m%d}", time=t, cloud_cover=cloud))
            day += dt.timedelta(days=1)
        return out

    def _add_buildings(self, cell, noise, is_event, bands, scl, water, n):
        """Stylised dense-urban block layout: bright roofs, concrete strips on both sides, one side shaded."""
        green, red, nir, swir = bands
        roof = (0.20, 0.22, 0.25, 0.30)                      # bright concrete
        tilt = (0.35, 0.30, 0.15, 0.10)                      # shade keeps blue-ish sky light, loses NIR/SWIR
        count = int(round(self.building_shadows * 24))
        taken = np.zeros((n, n), bool)
        for _ in range(count):
            h, w = int(cell.integers(7, 13)), int(cell.integers(7, 13))
            if n < h + 16 or n < w + 16:
                continue
            y0, x0 = int(cell.integers(2, n - h - 8)), int(cell.integers(5, n - w - 8))
            if taken[max(0, y0 - 1):y0 + h + 1, max(0, x0 - 4):x0 + w + 8].any():
                continue
            taken[y0:y0 + h, x0 - 4:x0 + w + 7] = True
            sl = (slice(y0, y0 + h),)
            for arr, v, t in zip((green, red, nir, swir), roof, tilt):
                arr[y0:y0 + h, x0 - 3:x0 + w + 3] = v                    # roof + concrete strips either side
                shade_cols = slice(x0 + w, x0 + w + 3) if is_event else slice(x0 - 3, x0)
                arr[y0:y0 + h, shade_cols] = v * t                       # the strip that is in shadow
            shade_cols = slice(x0 + w, x0 + w + 3) if is_event else slice(x0 - 3, x0)
            block = np.zeros((n, n), bool)
            block[y0:y0 + h, shade_cols] = True
            mislabel = block & (noise.random((n, n)) < 0.70)
            scl[mislabel] = SCL_WATER
            scl[block & ~mislabel] = 5
            scl[y0:y0 + h, x0:x0 + w] = 5
            water[y0:y0 + h, x0 - 3:x0 + w + 3] = False

    def read(self, scene, bbox, size_px):
        import cv2
        n = int(size_px)
        clat, clon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
        cell = self._rng("cell", round(clat, 3), round(clon, 3))   # shared by every scene of this cell
        noise = self._rng("scene", round(clat, 3), round(clon, 3), scene.id)
        yy, xx = np.mgrid[0:n, 0:n]

        coarse = noise.random((max(2, n // 10), max(2, n // 10))).astype(np.float32)
        veg = (cv2.resize(coarse, (n, n), interpolation=cv2.INTER_CUBIC) > 0.70)

        def disk(cy, cx, r):
            return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2

        permanent = np.zeros((n, n), bool)
        if cell.random() < 0.15:                                   # a pond/river present in every scene
            permanent |= disk(cell.integers(n // 5, n - n // 5), cell.integers(n // 5, n - n // 5), n * 0.08)
        is_flood_cell = cell.random() < self.flood_probability
        severity = 0.4 + 0.6 * cell.random()
        centers = [(cell.integers(n // 6, n - n // 6), cell.integers(n // 6, n - n // 6),
                    n * (0.06 + 0.04 * cell.random()) * severity) for _ in range(3)]
        is_event = scene.time >= self.wet_after
        flood = np.zeros((n, n), bool)
        if is_flood_cell and is_event:
            for cy, cx, r in centers:
                flood |= disk(cy, cx, r)
        water = permanent | flood

        green = 0.10 + 0.02 * veg
        red = 0.12 - 0.07 * veg
        nir = 0.28 + 0.12 * veg
        swir = 0.25 - 0.06 * veg
        for arr, w in ((green, 0.07), (red, 0.05), (nir, 0.02), (swir, 0.01)):
            arr[water] = w
        scl = np.where(veg, 4, 5).astype(np.uint8)
        if self.building_shadows > 0:
            self._add_buildings(cell, noise, is_event, (green, red, nir, swir), scl, water, n)
        scl[water] = SCL_WATER
        if is_event and self.cloud_fraction > 0:
            cloud = xx < int(self.cloud_fraction * n)
            for arr in (green, red, nir, swir):
                arr[cloud] = 0.55
            scl[cloud] = 9
        out = {}
        for name, arr in (("green", green), ("red", red), ("nir", nir), ("swir", swir)):
            out[name] = np.clip(arr + noise.normal(0, 0.006, arr.shape), 0.001, None).astype(np.float32)
        out["scl"] = scl
        return out


# ====================================================================== verifier
def _as_utc(x):
    if x is None:
        return dt.datetime.now(UTC)
    if isinstance(x, str):
        x = dt.datetime.fromisoformat(x.replace("Z", "+00:00"))
    if x.tzinfo is None:
        x = x.replace(tzinfo=UTC)
    return x.astimezone(UTC)


class SatelliteVerifier:
    """Orchestrates one cross-verification: cache -> scene search -> read -> indices -> verdict -> store.
    Safe to call from worker threads (the asyncio pipeline uses asyncio.to_thread)."""

    def __init__(self, conn, provider=None, *, aoi_radius_m=None, now_fn=None, method=None, shadow_filter=None):
        self.method = method                    # None -> config.SAT_METHOD
        self.shadow_filter = shadow_filter      # None -> config.SAT_SHADOW_FILTER
        self.conn = conn
        self.provider = provider or PlanetaryComputerProvider()
        self.aoi_radius_m = float(aoi_radius_m or config.SAT_AOI_RADIUS_M)
        self._now = now_fn or (lambda: dt.datetime.now(UTC))

    # ------------------------------------------------------------------ cache
    def _cached(self, key):
        ttl = max(config.SAT_CACHE_TTL_HOURS, config.SAT_INCONCLUSIVE_RETRY_HOURS)
        with kpdb.db_lock:
            row = kpdb.latest_satellite_check(self.conn, key, ttl)
        if row is None:
            return None
        age_h = (self._now() - _as_utc(row["created_at"])).total_seconds() / 3600.0
        if row["verdict"] in CONCLUSIVE_VERDICTS and age_h <= config.SAT_CACHE_TTL_HOURS:
            return row
        if row["verdict"] == INCONCLUSIVE and age_h <= config.SAT_INCONCLUSIVE_RETRY_HOURS:
            return row
        return None

    # ------------------------------------------------------------------ main entry
    def verify(self, lat, lon, hazard_type, event_time=None, trigger_post_id=None, force=False, aoi_radius_m=None):
        """aoi_radius_m: half-width of the window to inspect. Pass the report's location uncertainty (clipped to
        [SAT_AOI_RADIUS_M, SAT_AOI_MAX_RADIUS_M]) when the report only names a neighbourhood."""
        event = _as_utc(event_time) if event_time else self._now()
        applicability = float(config.SAT_APPLICABILITY.get(hazard_type, 0.0))
        radius = float(min(max(aoi_radius_m or self.aoi_radius_m, config.SAT_AOI_RADIUS_M), config.SAT_AOI_MAX_RADIUS_M))
        base_kw = dict(hazard_type=hazard_type, lat=float(lat), lon=float(lon), aoi_radius_m=radius,
                       event_time=event.isoformat(timespec="seconds"), applicability=applicability,
                       provider=self.provider.name, is_synthetic=self.provider.is_synthetic)
        if applicability <= 0.0:
            return SatelliteEvidence(verdict=NOT_APPLICABLE, notes=[
                f"'{hazard_type}' is smaller than a 10 m Sentinel-2 pixel; satellite cannot confirm or refute it."],
                **base_kw)

        key = cell_key(hazard_type, lat, lon)
        if not force:
            row = self._cached(key)
            if row is not None:
                ev = evidence_from_row(row)
                ev.cached = True
                return ev

        try:
            ev = self._measure(base_kw, lat, lon, event, radius)
        except Exception as exc:  # network, rasterio, STAC schema change ... never crash the pipeline
            log.warning("satellite check failed for %s at %.5f,%.5f: %s", hazard_type, lat, lon, exc)
            ev = SatelliteEvidence(verdict=INCONCLUSIVE, error=f"{type(exc).__name__}: {exc}",
                                   notes=["Satellite provider error - treated as no information."], **base_kw)
        ev.result_sha256 = ev.digest()
        ev.check_id = kpdb.insert_satellite_check(
            self.conn, trigger_post_id=trigger_post_id, hazard_type=hazard_type, lat=float(lat), lon=float(lon),
            aoi_radius_m=radius, event_time=ev.event_time, cell_key=key, verdict=ev.verdict,
            support=ev.support, quality=ev.quality, applicability=ev.applicability, scene_time=ev.scene_time,
            scene_id=ev.scene_id, baseline_scene_ids=ev.baseline_scene_ids, cloud_pct=ev.cloud_pct,
            valid_fraction=ev.valid_fraction, metrics_json=kpdb.canonical_json({**ev.metrics, "notes": ev.notes}),
            provider=ev.provider, is_synthetic=ev.is_synthetic, result_sha256=ev.result_sha256, error=ev.error)
        return ev

    def recheck_pending(self, max_age_days=5, limit=20):
        """Re-runs INCONCLUSIVE checks (a newer clear scene may exist now). The cache window
        (SAT_INCONCLUSIVE_RETRY_HOURS) stops this from hammering the free API."""
        cutoff = (self._now() - dt.timedelta(days=max_age_days)).isoformat(timespec="seconds")
        with kpdb.db_lock:
            cur = self.conn.execute(
                "SELECT hazard_type, lat, lon, event_time, trigger_post_id FROM satellite_checks "
                "WHERE verdict=? AND created_at>=? GROUP BY cell_key ORDER BY MAX(id) DESC LIMIT ?",
                (INCONCLUSIVE, cutoff, int(limit)))
            rows = cur.fetchall()
        return [self.verify(lat, lon, hz, event_time=ev_t, trigger_post_id=tp)
                for hz, lat, lon, ev_t, tp in rows]

    # ------------------------------------------------------------------ measurement
    def _measure(self, base_kw, lat, lon, event, radius):
        bbox = aoi_bbox(lat, lon, radius)
        size = max(8, int(round(2 * radius / config.SAT_PIXEL_M)))
        notes = []

        ev_start = event - dt.timedelta(days=config.SAT_EVENT_LOOKBACK_DAYS)
        ev_end = event + dt.timedelta(days=config.SAT_EVENT_LOOKAHEAD_DAYS)
        scenes = self.provider.search(bbox, ev_start, min(ev_end, self._now()), config.SAT_MAX_CLOUD_PCT)
        if not scenes:
            return SatelliteEvidence(verdict=INCONCLUSIVE, notes=[
                f"No Sentinel-2 scene under {config.SAT_MAX_CLOUD_PCT:.0f}% cloud within "
                f"{config.SAT_EVENT_LOOKBACK_DAYS} days of the report yet."], **base_kw)

        b_far, b_near = config.SAT_BASELINE_DAYS
        base_scenes = self.provider.search(bbox, event - dt.timedelta(days=b_far), event - dt.timedelta(days=b_near),
                                           config.SAT_MAX_CLOUD_PCT)
        base_scenes = sorted(base_scenes, key=lambda s: s.cloud_cover if s.cloud_cover is not None else 50.0
                             )[:config.SAT_BASELINE_MAX_SCENES]
        if not base_scenes:
            return SatelliteEvidence(verdict=INCONCLUSIVE, notes=[
                "No clear baseline scenes in the dry reference window; cannot measure change."], **base_kw)

        baselines, used_base = [], []
        for s in base_scenes:
            try:
                baselines.append(self.provider.read(s, bbox, size))
                used_base.append(s.id)
            except Exception as exc:
                notes.append(f"baseline scene {s.id} unreadable: {type(exc).__name__}")
        if not baselines:
            return SatelliteEvidence(verdict=INCONCLUSIVE, notes=notes + ["No baseline scene could be read."],
                                     **base_kw)

        ranked = sorted(scenes, key=lambda s: abs((s.time - event).total_seconds()) / 86400.0
                        + 0.05 * (s.cloud_cover or 0.0))
        best = None  # (valid_fraction, scene, metrics, preview)
        for s in ranked[:3]:
            try:
                arrays = self.provider.read(s, bbox, size)
            except Exception as exc:
                notes.append(f"event scene {s.id} unreadable: {type(exc).__name__}")
                continue
            metrics, preview = compute_change_metrics(arrays, baselines, method=self.method,
                                                      shadow_filter=self.shadow_filter)
            vf = metrics["joint_valid_fraction"] or 0.0
            if best is None or vf > best[0]:
                best = (vf, s, metrics, preview)
            if vf >= config.SAT_MIN_VALID_FRACTION:
                break
        if best is None:
            return SatelliteEvidence(verdict=INCONCLUSIVE, notes=notes + ["No event scene could be read."], **base_kw)

        vf, scene, metrics, preview = best
        gap_days = (scene.time - event).total_seconds() / 86400.0
        metrics["scene_offset_days"] = _r(gap_days, 2)
        quality = _clip01(vf * baseline_depth_factor(len(baselines)))
        common = dict(scene_time=scene.time.isoformat(timespec="seconds"), scene_id=scene.id,
                      baseline_scene_ids=used_base, cloud_pct=scene.cloud_cover, valid_fraction=vf,
                      metrics=metrics, quality=quality, preview=preview)
        if vf < config.SAT_MIN_VALID_FRACTION:
            notes.append(f"Only {vf:.0%} of the AOI is clear sky (needs {config.SAT_MIN_VALID_FRACTION:.0%}); "
                         "clouds/shadows block the view. Not evidence either way.")
            return SatelliteEvidence(verdict=INCONCLUSIVE, support=0.0, notes=notes, **common, **base_kw)
        if gap_days < -config.SAT_EVENT_LOOKBACK_DAYS:  # defensive; search window already enforces this
            notes.append("Nearest clear scene is too old to verify this report.")
            return SatelliteEvidence(verdict=INCONCLUSIVE, support=0.0, notes=notes, **common, **base_kw)

        support = float(metrics["support"])
        verdict = CONFIRMED_WET if support >= config.SAT_CONFIRM_THRESHOLD else NO_CHANGE
        if gap_days < 0:
            notes.append(f"Scene was acquired {abs(gap_days):.1f} days BEFORE the report; flooding that "
                         "started after it would not be visible.")
        if verdict == NO_CHANGE:
            notes.append("A 10 m pixel cannot resolve narrow street-level flooding; 'no change' means no "
                         "hectare-scale new water, not that the street is dry.")
        return SatelliteEvidence(verdict=verdict, support=support, notes=notes, **common, **base_kw)
