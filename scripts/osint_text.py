"""Transparent, lexicon-based text analysis for Karachi civic-hazard reports (English, Roman Urdu, Urdu script).

Deliberately NOT a neural model: there is no labelled Karachi hazard-text corpus to validate one on, and a
lexicon can be read, audited and corrected by the person it is explaining itself to. Every score is
traceable to the matched terms (`keywords`), and every weight lives in this file.

Outputs of analyze_text():
    hazard_type      one of config.HAZARD_LABEL_MAP values, or None (post is irrelevant)
    hazard_strength  0..1  noisy-OR over the weights of matched hazard terms
    urgency          0..1  injury/child/accident cues
    negativity       0..1  complaint / frustration cues (lexicon sentiment, NOT ML)
    community_signal 0..1  hazard_strength x (0.75 + 0.25 x max(negativity, urgency)) (+ media bonus)
    status           'active' | 'resolved' | 'irrelevant'
    lang             'en' | 'ur' | 'roman-ur' | 'mixed'

Known weaknesses (real, not hypothetical): sarcasm, negation beyond a 3-token window, spelling variants of
Roman Urdu, ambiguous words ("gaddha" = pit/hole, also an insult), and posts that quote an old incident.
The fusion layer limits the damage by counting each author once and capping how much text can ever contribute.
"""
from __future__ import annotations

import hashlib
import re
from collections import deque

import config

# ---------------------------------------------------------------- lexicons: (term, weight)
# Latin-script terms match on word boundaries; Urdu-script terms match as substrings.
HAZARD_LEXICON = {
    "pothole": [("pothole", .80), ("potholes", .80), ("road crater", .70), ("caved in", .55), ("cave in", .55),
                ("sinkhole", .60), ("gaddha", .45), ("gaddhe", .45), ("gaddhay", .45), ("gadhe", .30),
                ("گڑھا", .55), ("گڑھے", .55)],
    "open_manhole": [("open manhole", .90), ("manhole", .65), ("missing manhole", .90), ("manhole cover", .70),
                     ("uncovered manhole", .90), ("khula manhole", .90), ("manhole ka dhakkan", .85),
                     ("dhakkan nahi", .70), ("dhakan nahi", .70), ("dhakkan ghaib", .80),
                     ("مین ہول", .70), ("ڈھکن", .55)],
    "sewage_overflow": [("sewage", .70), ("sewerage", .70), ("sewer", .60), ("overflowing sewer", .90),
                        ("gutter overflow", .90), ("gutter", .55), ("gutar", .55), ("gutter ubal", .90),
                        ("ubal raha", .55), ("ubalti", .55), ("gandagi ka pani", .75), ("sewerage ka pani", .90),
                        ("گٹر", .65), ("سیوریج", .70), ("ابل", .35)],
    "waterlogging": [("waterlogging", .90), ("water logging", .90), ("waterlogged", .85), ("flooded", .70),
                     ("flooding", .65), ("flood", .50), ("standing water", .75), ("rainwater", .50),
                     ("rain water", .50), ("inundated", .80), ("knee deep", .70), ("pani jama", .85),
                     ("paani jama", .85), ("pani khara", .75), ("paani khara", .75), ("barsat ka pani", .65),
                     ("sailab", .55), ("پانی جمع", .85), ("بارش کا پانی", .60), ("سیلاب", .55)],
    "open_drain": [("open drain", .85), ("open nala", .85), ("nullah", .55), ("nala overflow", .85),
                   ("nala", .45), ("nalay", .45), ("drain overflow", .75), ("نالہ", .50), ("نالے", .50)],
    "garbage": [("garbage", .60), ("trash", .50), ("rubbish", .50), ("dumped waste", .60), ("kachra", .60),
                ("kooda", .50), ("کچرا", .60), ("کوڑا", .50)],
    "road_damage": [("broken road", .75), ("damaged road", .75), ("road damaged", .75), ("cracked road", .70),
                    ("road collapsed", .85), ("sadak toot", .70), ("tooti hui sadak", .75), ("ٹوٹی سڑک", .75)],
}

URGENCY_LEXICON = [("child", .60), ("children", .60), ("kid", .50), ("kids", .50), ("baby", .55),
                   ("accident", .55), ("injured", .70), ("injury", .60), ("died", .90), ("death", .85),
                   ("dead", .85), ("drowned", .95), ("fell into", .75), ("fell", .45), ("electrocut", .85),
                   ("life threatening", .85), ("emergency", .60), ("urgent", .50), ("dangerous", .45),
                   ("bacha", .60), ("bachay", .60), ("bachcha", .60), ("bache", .60), ("hadsa", .60),
                   ("haadsa", .60), ("zakhmi", .70), ("mar gaya", .90), ("foran", .45), ("khatarnak", .45),
                   ("بچہ", .60), ("بچے", .60), ("حادثہ", .60), ("زخمی", .70), ("خطرناک", .45), ("فوری", .45)]

NEGATIVITY_LEXICON = [("terrible", .50), ("horrible", .55), ("worst", .60), ("disgusting", .55), ("fed up", .60),
                      ("nobody cares", .70), ("no one cares", .70), ("ignored", .55), ("helpless", .55),
                      ("complaint", .40), ("complained", .45), ("still", .20), ("again", .20), ("weeks", .35),
                      ("months", .45), ("unbearable", .60), ("pareshan", .50), ("tang", .40), ("masla", .35),
                      ("masail", .40), ("koi nahi sunta", .70), ("kab tak", .50), ("bura haal", .55),
                      ("zimmedar", .45), ("پریشان", .50), ("مسئلہ", .35), ("کب تک", .50)]

RESOLVED_LEXICON = ["fixed", "repaired", "cleared", "resolved", "cleaned", "covered now", "has been covered",
                    "theek ho gaya", "thik ho gaya", "saaf ho gaya", "hal ho gaya", "kaam mukammal",
                    "ٹھیک ہو گیا", "صاف ہو گیا", "حل ہو گیا"]
NEGATORS = {"not", "never", "no", "nahi", "nahin", "nhi", "na", "yet", "isn't", "wasn't", "hasn't", "still", "abhi"}

ROMAN_URDU_MARKERS = {"hai", "hain", "nahi", "nahin", "ka", "ki", "ke", "ko", "mein", "main", "bohat", "bahut",
                      "ho", "gaya", "gayi", "raha", "rahi", "pani", "paani", "aur", "yeh", "ye", "woh", "kya",
                      "abhi", "sab", "koi", "se", "par", "pe", "tak", "bhi", "hi", "jama", "khula", "sadak"}

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?92|0)?[-\s]?(?:3\d{2}|21)[-\s]?\d{7}(?!\d)")
_MENTION_RE = re.compile(r"(?<![\w])@\w{2,}")
_URDU_RE = re.compile(r"[؀-ۿ]")


# ---------------------------------------------------------------- privacy
def redact(text: str) -> str:
    """Removes emails, phone numbers and @mentions from stored text (raw handles are never kept)."""
    text = _EMAIL_RE.sub("[email]", text or "")
    text = _PHONE_RE.sub("[phone]", text)
    return _MENTION_RE.sub("[user]", text)


_salt_warned = False


def author_hash(platform: str, handle) -> str | None:
    """Salted one-way pseudonym so 'same author' can be detected without storing who the author is."""
    global _salt_warned
    if not handle:
        return None
    if config.OSINT_PRIVACY_SALT == "change-me-in-dotenv" and not _salt_warned:
        import logging
        logging.getLogger(__name__).warning("KP_PRIVACY_SALT is not set; author pseudonyms use the default salt "
                                            "and can be brute-forced. Set it in .env.")
        _salt_warned = True
    return hashlib.sha256(f"{config.OSINT_PRIVACY_SALT}|{platform}|{str(handle).strip().lower()}".encode()).hexdigest()[:16]


# ---------------------------------------------------------------- matching helpers
def _norm(text: str) -> str:
    t = _URL_RE.sub(" ", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s؀-ۿ']", " ", t)).strip()


def _find(low: str, term: str) -> int:
    """Index of `term` in normalised text (word-bounded for Latin script), or -1."""
    if term.isascii():
        m = re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", low)
        return m.start() if m else -1
    return low.find(term)


def _matches(low: str, lexicon):
    hits = [(t, w) for t, w in lexicon if _find(low, t) >= 0]
    drop = {t for t, _ in hits if any(t != u and t in u for u, _ in hits)}   # keep the longer phrase only
    return [(t, w) for t, w in hits if t not in drop]


def _noisy_or(ws):
    p = 1.0
    for w in ws:
        p *= 1.0 - w
    return 1.0 - p


def detect_language(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    urdu_ratio = sum(1 for c in letters if _URDU_RE.match(c)) / len(letters)
    toks = set(_norm(text).split())
    roman_hits = len(toks & ROMAN_URDU_MARKERS)
    if urdu_ratio > 0.6:
        return "ur"
    if urdu_ratio > 0.15:
        return "mixed"
    return "roman-ur" if roman_hits >= 2 else "en"


def _is_resolved(low: str) -> bool:
    for term in RESOLVED_LEXICON:
        i = _find(low, term)
        if i < 0:
            continue
        before = low[:i].split()[-3:]
        if not (set(before) & NEGATORS):
            return True
    return False


# ---------------------------------------------------------------- main entry
def analyze_text(text: str, n_images: int = 0) -> dict:
    low = _norm(text or "")
    per_hazard = {}
    for hazard, lex in HAZARD_LEXICON.items():
        hits = _matches(low, lex)
        if hits:
            per_hazard[hazard] = (_noisy_or([w for _, w in hits]), [t for t, _ in hits])
    urg_hits, neg_hits = _matches(low, URGENCY_LEXICON), _matches(low, NEGATIVITY_LEXICON)
    urgency, negativity = _noisy_or([w for _, w in urg_hits]), _noisy_or([w for _, w in neg_hits])
    lang = detect_language(text or "")

    if not per_hazard:
        return {"hazard_type": None, "hazard_strength": 0.0, "urgency": round(urgency, 3),
                "negativity": round(negativity, 3), "community_signal": 0.0, "status": "irrelevant",
                "keywords": [], "lang": lang}
    # Specific beats generic: 'sewage' + 'flood' in one sentence is a sewage overflow, 'nala' + 'pani jama' a drain flood.
    hazard, (strength, terms) = max(per_hazard.items(), key=lambda kv: (kv[1][0], len(kv[1][1])))
    media = 0.10 * strength if n_images > 0 else 0.0
    signal = min(1.0, strength * (0.75 + 0.25 * max(negativity, urgency)) + media)
    return {"hazard_type": hazard, "hazard_strength": round(strength, 3), "urgency": round(urgency, 3),
            "negativity": round(negativity, 3), "community_signal": round(signal, 3),
            "status": "resolved" if _is_resolved(low) else "active",
            "keywords": terms + [t for t, _ in urg_hits][:3], "lang": lang}


# ---------------------------------------------------------------- near-duplicate detection
def shingles(text: str, k: int = 3) -> frozenset:
    toks = _norm(text).split()
    if len(toks) < k:
        return frozenset([" ".join(toks)]) if toks else frozenset()
    return frozenset(" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1))


def jaccard(a, b) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


class DuplicateDetector:
    """Remembers the last N posts and flags a new one that is a copy-paste (Jaccard over 3-word shingles).
    Copies still get stored (provenance), but fusion ignores them so a viral forward cannot fake corroboration."""

    def __init__(self, memory=None, threshold=None):
        self.threshold = config.OSINT_DEDUP_JACCARD if threshold is None else threshold
        self._recent = deque(maxlen=memory or config.OSINT_DEDUP_MEMORY)   # (db_id, shingles)

    def seed(self, rows):
        for db_id, text in rows:
            self._recent.append((db_id, shingles(text)))

    def check(self, text):
        """-> id of the earlier near-identical post, or None."""
        s = shingles(text)
        for db_id, other in self._recent:
            if jaccard(s, other) >= self.threshold:
                return db_id
        return None

    def add(self, db_id, text):
        self._recent.append((db_id, shingles(text)))
