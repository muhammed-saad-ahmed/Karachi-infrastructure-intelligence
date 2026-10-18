"""Offline geocoder for Karachi civic reports: explicit coordinates, map links, or known neighbourhood names.

No external geocoding API (they are rate-limited, and many forbid storing results). Instead:

    1. EXPLICIT   "24.9215, 67.0925", "@24.92,67.09" (map links), "?q=24.92,67.09"  -> radius 30 m
    2. GAZETTEER  neighbourhood name found in the text -> approximate centroid + an uncertainty radius

HONESTY: gazetteer coordinates are APPROXIMATE neighbourhood centroids typed from general knowledge, good to
roughly a kilometre; the radius is the uncertainty, and fusion down-weights imprecise locations
(config.FUSION_GEO_PRECISION). Verify/replace them with OpenStreetMap-derived centroids for serious use -
this table is meant to be edited. A report that only says "Karachi" gets NO location and is never fused.

LOCATION PRIVACY (bottom of this file)
    A precise coordinate in a community report (typed by the author or taken from a map link) is a pointer to a
    person's position. Before such a coordinate is stored or used, `privatize_location` adds planar-Laplace noise:
    the mechanism of geo-indistinguishability (Andres, Bordenabe, Chatzikokolakis, Palamidessi, CCS 2013), the
    metric-space form of local differential privacy. Gazetteer centroids are NOT perturbed: they are the same
    public point for everybody in a neighbourhood, so noise would cost accuracy and protect nobody.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import random
import re

import numpy as np

import config
from geospatial.clustering import valid_coord

# name -> (lat, lon, uncertainty_radius_m, aliases...)
_RAW = {
    "Saddar":            (24.8607, 67.0100, 1500, "saddar", "sadar", "صدر"),
    "Clifton":           (24.8138, 67.0300, 2500, "clifton", "کلفٹن"),
    "DHA":               (24.8100, 67.0650, 4000, "dha", "defence housing", "defence", "defense"),
    "Teen Talwar":       (24.8210, 67.0310, 600,  "teen talwar", "teen talwaar"),
    "Sea View":          (24.7830, 67.0500, 1500, "sea view", "seaview"),
    "Korangi":           (24.8300, 67.1200, 3500, "korangi", "کورنگی"),
    "Landhi":            (24.8450, 67.2000, 3500, "landhi", "landhi"),
    "Malir":             (24.8930, 67.1900, 3500, "malir", "ملیر"),
    "Shah Faisal":       (24.8850, 67.1550, 2000, "shah faisal"),
    "Airport":           (24.9065, 67.1608, 1500, "jinnah airport", "karachi airport"),
    "Gulshan-e-Iqbal":   (24.9215, 67.0925, 3000, "gulshan-e-iqbal", "gulshan e iqbal", "gulshan-i-iqbal",
                          "gulshan iqbal", "gulshan", "گلشن اقبال", "گلشن"),
    "Nipa":              (24.9190, 67.0970, 800,  "nipa", "nipa chowrangi"),
    "Gulistan-e-Johar":  (24.9145, 67.1330, 3000, "gulistan-e-johar", "gulistan e johar", "gulistan-e-jauhar",
                          "johar", "jauhar"),
    "Scheme 33":         (24.9700, 67.1400, 3500, "scheme 33", "scheme-33"),
    "Liaquatabad":       (24.9100, 67.0450, 1500, "liaquatabad", "lyaquatabad", "لیاقت آباد"),
    "Nazimabad":         (24.9100, 67.0300, 1500, "nazimabad"),
    "North Nazimabad":   (24.9420, 67.0400, 2000, "north nazimabad"),
    "Federal B Area":    (24.9300, 67.0750, 2000, "federal b area", "f.b area", "fb area"),
    "Gulberg":           (24.9300, 67.0600, 1500, "gulberg"),
    "New Karachi":       (24.9900, 67.0600, 3500, "new karachi", "north karachi"),
    "Orangi Town":       (24.9500, 66.9930, 3500, "orangi", "orangi town", "اورنگی"),
    "Baldia":            (24.9100, 66.9600, 3000, "baldia", "baldia town"),
    "SITE":              (24.9000, 66.9800, 3000, "site area", "s.i.t.e", "sindh industrial"),
    "Surjani":           (25.0300, 67.0500, 3500, "surjani", "surjani town"),
    "Lyari":             (24.8650, 67.0000, 1500, "lyari", "لیاری"),
    "Keamari":           (24.8300, 66.9800, 3000, "keamari", "kemari"),
    "Hawksbay":          (24.8480, 66.8900, 3000, "hawksbay", "hawks bay"),
    "Garden":            (24.8700, 67.0230, 1000, "garden east", "garden west", "karachi zoo"),
    "PECHS":             (24.8700, 67.0600, 1500, "pechs", "p.e.c.h.s", "tariq road"),
    "Bahadurabad":       (24.8800, 67.0680, 1000, "bahadurabad", "bahadur abad"),
    "Sharea Faisal":     (24.8700, 67.0900, 1500, "shahrah-e-faisal", "shara-e-faisal", "sharea faisal",
                          "shahrah e faisal", "sharah e faisal"),
    "Mazar-e-Quaid":     (24.8750, 67.0430, 700,  "mazar-e-quaid", "mazar e quaid", "quaid mazar"),
    "Manzoor Colony":    (24.8470, 67.0770, 1500, "manzoor colony", "mahmoodabad", "mehmoodabad"),
}


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_.]+", " ", s.lower()).strip()


_ALIAS = []   # (alias_normalised, place, lat, lon, radius, is_latin)
for _place, (_lat, _lon, _rad, *_aliases) in _RAW.items():
    for _a in {_norm(x) for x in _aliases} | {_norm(_place)}:
        _ALIAS.append((_a, _place, _lat, _lon, _rad, _a.isascii()))
# longest alias first so "north nazimabad" wins over "nazimabad", "gulshan e iqbal" over "gulshan"
_ALIAS.sort(key=lambda t: -len(t[0]))

_COORD_RES = (
    re.compile(r"@(-?\d{1,3}\.\d+),\s*(-?\d{1,3}\.\d+)"),                          # maps.google.com/@lat,lon,zoom
    re.compile(r"[?&](?:q|ll|query|destination)=(-?\d{1,3}\.\d+)(?:,|%2C)\s*(-?\d{1,3}\.\d+)"),
    re.compile(r"(?<![\d.])(2[45]\.\d{3,})\s*[,;/ ]\s*(6[67]\.\d{3,})(?![\d])"),   # plain "24.9215, 67.0925"
)


def extract_explicit_coords(text: str):
    """First coordinate pair in `text` that lies inside Karachi's bounding box, else None."""
    for rx in _COORD_RES:
        for m in rx.finditer(text or ""):
            lat, lon = float(m.group(1)), float(m.group(2))
            if valid_coord(lat, lon, config.KARACHI_BBOX):
                return lat, lon
    return None


def geocode_text(text: str):
    """-> {"lat","lon","radius_m","method","place"} or None. Never guesses from the word 'Karachi' alone."""
    if not text:
        return None
    xy = extract_explicit_coords(text)
    if xy:
        return {"lat": xy[0], "lon": xy[1], "radius_m": 30.0, "method": "explicit", "place": None}
    low = _norm(text)
    best = None
    for alias, place, lat, lon, rad, latin in _ALIAS:
        if latin:
            m = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", low)
        else:
            m = re.search(re.escape(alias), low)
        # most specific name wins: longest alias ("north nazimabad" beats "nazimabad"), then tighter radius
        if m and (best is None or (len(alias), -rad, -m.start()) > (len(best[3]), -best[4], -best[5])):
            best = (place, lat, lon, alias, rad, m.start())
    if best is None:
        return None
    place, lat, lon, _alias, rad, _pos = best
    return {"lat": lat, "lon": lon, "radius_m": float(rad), "method": "gazetteer", "place": place}


def known_places():
    return sorted(_RAW)


# ====================================================================== location privacy (geo-indistinguishability)
M_PER_DEG_LAT = 111_320.0
_SYSTEM_RNG = random.SystemRandom()           # OS entropy; noise for privacy must not be predictable
_MIN_X = -1.0 / math.e


def _lambert_w_minus1(x):
    """Lower real branch W_{-1}(x) of the Lambert W function for x in [-1/e, 0), by Halley iteration.
    Verified against scipy.special.lambertw in the tests (no SciPy needed at runtime)."""
    if not (_MIN_X <= x < 0.0):
        raise ValueError("W_-1 is real only on [-1/e, 0)")
    if x == _MIN_X:
        return -1.0
    if x < -0.25:                                                  # series around the branch point
        p = -math.sqrt(2.0 * (math.e * x + 1.0))
        w = -1.0 + p - p * p / 3.0 + 11.0 / 72.0 * p ** 3
    else:                                                          # asymptotic expansion towards 0-
        l1 = math.log(-x)
        l2 = math.log(-l1)
        w = l1 - l2 + l2 / l1
    for _ in range(50):
        ew = math.exp(w)
        f = w * ew - x
        if f == 0.0:
            break
        denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0)
        if denom == 0.0:
            break
        w_new = w - f / denom
        if abs(w_new - w) <= 1e-15 * max(1.0, abs(w)):
            w = w_new
            break
        w = w_new
    return w


def planar_laplace_radius(epsilon, p):
    """Inverse CDF of the planar-Laplace radius: C(r) = 1 - (1 + eps r) exp(-eps r),
    r = -(1/eps) (W_-1((p - 1)/e) + 1).  epsilon in 1/metre, p in [0, 1)."""
    if not epsilon > 0:
        raise ValueError("epsilon must be > 0 (per metre)")
    if not 0.0 <= p < 1.0:
        raise ValueError("p must be in [0, 1)")
    return -(_lambert_w_minus1((p - 1.0) / math.e) + 1.0) / epsilon


class GeoPrivacy:
    """Parameters of the mechanism. `epsilon` is per metre: any two true locations d metres apart yield output
    distributions whose density ratio is at most exp(epsilon * d).

        expected displacement   2 / epsilon
        radius containing a share q of the noise     radius_at(q)
    The noise radius is hard-capped at radius_at(cap_quantile): the output is then (epsilon, delta)-geo-
    indistinguishable with delta <= (exp(epsilon*d) + 1) * (1 - cap_quantile) for points d metres apart. The
    cap exists so that a stored point is never absurdly far away; set cap_quantile=1.0 for the pure mechanism."""

    def __init__(self, epsilon, cap_quantile=None):
        if not (isinstance(epsilon, (int, float)) and math.isfinite(epsilon) and epsilon > 0):
            raise ValueError("epsilon must be a positive finite number (per metre)")
        self.epsilon = float(epsilon)
        q = config.PRIVACY_CAP_QUANTILE if cap_quantile is None else float(cap_quantile)
        if not 0.5 <= q <= 1.0:
            raise ValueError("cap_quantile must be in [0.5, 1]")
        self.cap_quantile = q
        self.cap_radius_m = None if q >= 1.0 else self.radius_at(q)

    @classmethod
    def from_level(cls, level=None, radius_m=None, cap_quantile=None):
        """Andres et al.'s parametrisation: privacy level l within radius r  ->  epsilon = l / r."""
        level = config.PRIVACY_LEVEL if level is None else level
        radius_m = config.PRIVACY_RADIUS_M if radius_m is None else radius_m
        if not (level > 0 and radius_m > 0):
            raise ValueError("level and radius_m must be > 0")
        return cls(level / radius_m, cap_quantile)

    @classmethod
    def from_config(cls):
        return cls.from_level()

    def radius_at(self, q):
        return planar_laplace_radius(self.epsilon, min(max(q, 0.0), 1.0 - 1e-12))

    @property
    def expected_displacement_m(self):
        return 2.0 / self.epsilon

    def describe(self):
        return {"epsilon_per_m": self.epsilon, "expected_displacement_m": round(self.expected_displacement_m, 1),
                "r50_m": round(self.radius_at(0.5), 1), "r95_m": round(self.radius_at(0.95), 1),
                "cap_radius_m": None if self.cap_radius_m is None else round(self.cap_radius_m, 1),
                "cap_quantile": self.cap_quantile}

    def __repr__(self):
        return f"GeoPrivacy(epsilon={self.epsilon:.5g}/m, cap_quantile={self.cap_quantile})"


def sticky_rng(secret, context):
    """Deterministic generator keyed by HMAC(secret, context). Used so the SAME author reporting from the SAME ~55 m
    spot always gets the SAME noise: otherwise k reports would let an observer average the noise away and learn the
    true position (privacy loss composes to k * epsilon). Without `secret` the stream is unpredictable."""
    key = secret.encode() if isinstance(secret, str) else bytes(secret)
    digest = hmac.new(key, str(context).encode(), hashlib.sha256).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "little")))


def privatize_point(lat, lon, privacy, *, rng=None, bbox=None):
    """Planar-Laplace noise on one coordinate. rng: object with .random() in [0,1) (default: OS entropy).
    The result is remapped into `bbox` (default Karachi) - post-processing that does not weaken the guarantee.
    -> (lat, lon, info) where info has displacement_m, clamped."""
    lat, lon = float(lat), float(lon)
    if not valid_coord(lat, lon):
        raise ValueError(f"invalid coordinate {lat!r}, {lon!r}")
    rng = rng or _SYSTEM_RNG
    p = min(rng.random(), 1.0 - 2.0 ** -53)
    r = planar_laplace_radius(privacy.epsilon, p)
    if privacy.cap_radius_m is not None:
        r = min(r, privacy.cap_radius_m)
    theta = 2.0 * math.pi * rng.random()
    dlat = r * math.sin(theta) / M_PER_DEG_LAT
    dlon = r * math.cos(theta) / (M_PER_DEG_LAT * max(math.cos(math.radians(lat)), 1e-6))
    new_lat, new_lon = lat + dlat, lon + dlon
    bbox = config.KARACHI_BBOX if bbox is None else bbox
    clamped = False
    if bbox is not None:
        c_lat, c_lon = min(max(new_lat, bbox[0]), bbox[2]), min(max(new_lon, bbox[1]), bbox[3])
        clamped = (c_lat, c_lon) != (new_lat, new_lon)
        new_lat, new_lon = c_lat, c_lon
    return new_lat, new_lon, {"displacement_m": r, "clamped": clamped}


def uncertainty_radius_m(privacy, original_radius_m=30.0, q=0.95):
    """Location uncertainty to record after privatisation. Two independently perturbed reports of the same true
    spot differ by about sqrt(2) times the single-report noise, so the radius within which fusion should look for
    matching reports is sqrt(2) * r_q combined (root-sum-square) with the original radius."""
    return math.hypot(float(original_radius_m or 0.0), math.sqrt(2.0) * privacy.radius_at(q))


def privatize_location(geo, *, author_hash=None, is_synthetic=False, privacy=None, rng=None, secret=None):
    """Applies the mechanism to a geocode dict {"lat","lon","radius_m","method","place"} and returns a NEW dict.
    Only precise methods (config.PRIVACY_PRECISE_METHODS) are touched; gazetteer centroids and already-private
    results (method ends with '+ldp') pass through unchanged. The returned radius_m is the enlarged uncertainty so
    that risk_engine.fusion down-weights the report and still groups it with its neighbours.
    Never raises on bad input: an unusable coordinate is DROPPED (returns None) rather than stored unprotected."""
    if not geo:
        return geo
    if geo.get("method") not in config.PRIVACY_PRECISE_METHODS:
        return geo
    if not config.PRIVACY_LDP_ENABLED and privacy is None:
        return geo
    if is_synthetic and not config.PRIVACY_APPLY_TO_SYNTHETIC and privacy is None:
        return geo
    try:
        priv = privacy or GeoPrivacy.from_config()
        if rng is None:
            step = config.PRIVACY_STICKY_GRID_DEG
            ctx = f"{author_hash or 'anon'}|{round(geo['lat'] / step)}|{round(geo['lon'] / step)}"
            # an anonymous report gets fresh OS noise; an identified (pseudonymous) author gets sticky noise
            rng = sticky_rng(secret or config.OSINT_PRIVACY_SALT, ctx) if author_hash else None
        lat, lon, info = privatize_point(geo["lat"], geo["lon"], priv, rng=rng)
    except Exception:
        return None
    out = dict(geo)
    out.update(lat=lat, lon=lon, radius_m=uncertainty_radius_m(priv, geo.get("radius_m") or 30.0),
               method=f"{geo['method']}+ldp", privacy_epsilon=priv.epsilon)
    return out


_ANY_COORD_RE = re.compile(r"(?<![\d.])(-?\d{1,3}\.\d{3,})\s*[,;/ ]\s*(-?\d{1,3}\.\d{3,})(?![\d])")


def redact_coordinates(text):
    """Replaces every latitude/longitude pair (>= 3 decimals) in free text or a URL with '[location]'. The precise point
    must not survive in the stored text after it has been privatised in the lat/lon columns."""
    def sub(m):
        try:
            return "[location]" if valid_coord(float(m.group(1)), float(m.group(2))) else m.group(0)
        except ValueError:
            return m.group(0)
    return _ANY_COORD_RE.sub(sub, text or "")
