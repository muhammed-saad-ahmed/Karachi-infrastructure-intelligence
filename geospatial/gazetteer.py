"""Offline geocoder for Karachi civic reports: explicit coordinates, map links, or known neighbourhood names.

No external geocoding API (they are rate-limited, and many forbid storing results). Instead:

    1. EXPLICIT   "24.9215, 67.0925", "@24.92,67.09" (map links), "?q=24.92,67.09"  -> radius 30 m
    2. GAZETTEER  neighbourhood name found in the text -> approximate centroid + an uncertainty radius

HONESTY: gazetteer coordinates are APPROXIMATE neighbourhood centroids typed from general knowledge, good to
roughly a kilometre; the radius is the uncertainty, and fusion down-weights imprecise locations
(config.FUSION_GEO_PRECISION). Verify/replace them with OpenStreetMap-derived centroids for serious use -
this table is meant to be edited. A report that only says "Karachi" gets NO location and is never fused.
"""
from __future__ import annotations

import re

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
