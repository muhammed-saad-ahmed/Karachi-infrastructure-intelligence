"""Asynchronous OSINT pipeline: public civic-hazard posts -> structured, geolocated, de-duplicated evidence.

    sources (RSS / HTML / mock stream)
        -> polite async fetch (robots.txt, per-host rate limit, retries + backoff, circuit breaker)
        -> parse (RSS/Atom or CSS-selector HTML)  -> RawPost
        -> privacy redaction + author pseudonym -> hazard / urgency / sentiment (scripts/osint_text.py)
        -> geocode (explicit coords or Karachi gazetteer) -> near-duplicate check
        -> SQLite `osint_posts`
        -> if the signal is strong and the hazard is satellite-observable: Sentinel-2 cross-check
           (geospatial/satellite.py), then (optionally) refresh the fused risk matrix (risk_engine/fusion.py)

Run inside Streamlit (a script re-run must not block): use `run_sync(...)` or `start_background(...)`.
Run from a shell:
    python -m scripts.osint_pipeline --mock 40                      # offline simulator, no network
    python -m scripts.osint_pipeline --sources data/osint_sources.json --once
    python -m scripts.osint_pipeline --sources data/osint_sources.json --interval 600

Responsible scraping (this is a hard requirement, not a suggestion)
-------------------------------------------------------------------
* Only public pages you are allowed to read. No logins, no bypassing paywalls/bot walls/CAPTCHAs.
* robots.txt is honoured (including Crawl-delay); per-host minimum delay >= config.OSINT_MIN_DELAY_S.
* Most social platforms forbid scraping in their Terms of Service and offer official APIs with their own
  limits. Do not point this at them. Prefer RSS feeds, official open-data/complaint portals, and sources that
  explicitly permit reuse. Check each source's terms yourself.
* Personal data is minimised: author handles are salted-hashed, emails/phones/@mentions are removed from
  stored text, images are NOT downloaded (URLs only). Reports are about places, not people.
* The default User-Agent identifies this project; set a contact URL/email in config.OSINT_USER_AGENT.
* The mock stream is synthetic and every row it produces is flagged is_synthetic=1.

Pure Python 3.10+; only optional extras are `beautifulsoup4` (+`lxml`) for HTML sources and `playwright`
for JavaScript-rendered pages.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import email.utils
import hashlib
import json
import logging
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
import xml.etree.ElementTree as ET
from typing import AsyncIterator, Awaitable, Callable, Iterable

import config
from database import db as kpdb
from geospatial.gazetteer import geocode_text, known_places, privatize_location
from scripts import osint_text as nlp

log = logging.getLogger(__name__)
UTC = dt.timezone.utc


# ====================================================================== data classes
@dataclasses.dataclass
class RawPost:
    platform: str
    post_id: str
    text: str
    url: str | None = None
    author: str | None = None              # raw handle: hashed immediately, never stored
    posted_at: dt.datetime | None = None
    image_urls: list = dataclasses.field(default_factory=list)
    lat: float | None = None               # only if the source itself provides coordinates
    lon: float | None = None
    is_synthetic: bool = False


@dataclasses.dataclass
class SourceSpec:
    """kind='rss' needs only a url. kind='html' needs CSS selectors (relative to each item):
        {"item": "article.post", "text": ".body", "link": "a.permalink", "time": "time",
         "image": "img", "author": ".author"}
    kind='html_js' is the same but rendered with Playwright (JavaScript pages)."""
    name: str
    url: str
    kind: str = "rss"
    selectors: dict = dataclasses.field(default_factory=dict)
    max_items: int = 50


@dataclasses.dataclass
class PipelineStats:
    scraped: int = 0
    inserted: int = 0
    already_seen: int = 0
    irrelevant: int = 0
    resolved: int = 0
    near_duplicates: int = 0
    geocoded: int = 0
    alerts: int = 0
    satellite_checks: int = 0
    fused_cells: int = 0
    source_errors: list = dataclasses.field(default_factory=list)

    def merge(self, other: "PipelineStats"):
        for f in dataclasses.fields(self):
            a, b = getattr(self, f.name), getattr(other, f.name)
            setattr(self, f.name, a + b)

    def as_dict(self):
        return dataclasses.asdict(self)


def load_sources(path: str) -> list[SourceSpec]:
    with open(path, encoding="utf-8") as f:
        return [SourceSpec(**s) for s in json.load(f)]


# ====================================================================== polite async HTTP fetcher
class FetchError(RuntimeError):
    pass


class HttpFetcher:
    """Async, stdlib-only (urllib in worker threads, so no extra dependency and it honours HTTPS_PROXY).

    Politeness:  robots.txt (RFC 9309 semantics: 4xx -> allowed, 5xx/unreachable -> disallowed), Crawl-delay,
                 minimum per-host delay, global concurrency cap.
    Resilience:  retries on 429/5xx/timeouts with exponential backoff + jitter, honours Retry-After,
                 per-host circuit breaker (5 consecutive failures -> host skipped for 5 minutes).
    Safety:      http/https only, response size cap, decoded with the declared charset.
    """

    def __init__(self, user_agent=None, min_delay=None, max_concurrency=None, timeout=None, max_retries=None,
                 respect_robots=True, sleep=asyncio.sleep, clock=time.monotonic):
        self.ua = user_agent or config.OSINT_USER_AGENT
        self.min_delay = config.OSINT_MIN_DELAY_S if min_delay is None else min_delay
        self.timeout = timeout or config.OSINT_REQUEST_TIMEOUT_S
        self.max_retries = config.OSINT_MAX_RETRIES if max_retries is None else max_retries
        self.respect_robots = respect_robots
        self._sem = asyncio.Semaphore(max_concurrency or config.OSINT_MAX_CONCURRENCY)
        self._sleep, self._clock = sleep, clock
        self._host_lock: dict[str, asyncio.Lock] = {}
        self._host_next: dict[str, float] = {}
        self._robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._fails: dict[str, int] = {}
        self._blocked_until: dict[str, float] = {}

    # -- blocking primitive (runs in a thread)
    def _get_sync(self, url):
        req = urllib.request.Request(url, headers={"User-Agent": self.ua, "Accept": "text/html,application/xml,*/*;q=0.5",
                                                   "Accept-Language": "en,ur;q=0.8"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read(config.OSINT_MAX_BODY_BYTES + 1)
                charset = r.headers.get_content_charset() or "utf-8"
                return r.status, dict(r.headers), body[:config.OSINT_MAX_BODY_BYTES], charset
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), b"", "utf-8"

    async def _get(self, url):
        status, headers, body, charset = await asyncio.to_thread(self._get_sync, url)
        return status, headers, body.decode(charset, errors="replace")

    # -- politeness
    async def _robots_for(self, scheme, host):
        key = f"{scheme}://{host}"
        if key not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            try:
                status, _, text = await self._get(f"{key}/robots.txt")
                if status == 200:
                    rp.parse(text.splitlines())
                elif 400 <= status < 500:
                    rp.parse([])                      # no robots.txt -> everything allowed
                else:
                    rp = None                         # server error -> assume disallowed
            except Exception:
                rp = None
            self._robots[key] = rp
        return self._robots[key]

    async def _wait_turn(self, host, crawl_delay):
        lock = self._host_lock.setdefault(host, asyncio.Lock())
        async with lock:
            delay = max(self.min_delay, crawl_delay or 0.0)
            wait = self._host_next.get(host, 0.0) - self._clock()
            if wait > 0:
                await self._sleep(wait)
            self._host_next[host] = self._clock() + delay

    async def fetch(self, url: str) -> str:
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            raise FetchError(f"refusing non-http(s) URL: {url!r}")
        host = p.netloc
        if self._blocked_until.get(host, 0.0) > self._clock():
            raise FetchError(f"{host} temporarily skipped after repeated failures")
        crawl_delay = None
        if self.respect_robots:
            rp = await self._robots_for(p.scheme, host)
            if rp is None or not rp.can_fetch(self.ua, url):
                raise FetchError(f"robots.txt does not allow fetching {url}")
            crawl_delay = rp.crawl_delay(self.ua)
        last = None
        for attempt in range(self.max_retries + 1):
            async with self._sem:
                await self._wait_turn(host, crawl_delay)
                try:
                    status, headers, text = await self._get(url)
                except Exception as exc:          # DNS, TLS, timeout, connection reset
                    status, headers, text, last = None, {}, "", f"{type(exc).__name__}: {exc}"
            if status is not None and 200 <= status < 300:
                self._fails[host] = 0
                return text
            if status is not None and status < 500 and status != 429:
                raise FetchError(f"HTTP {status} for {url}")
            last = last if status is None else f"HTTP {status}"
            if attempt < self.max_retries:
                retry_after = None
                try:
                    retry_after = float(headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    pass
                back = min(config.OSINT_BACKOFF_MAX_S, config.OSINT_BACKOFF_BASE_S * 2 ** attempt)
                await self._sleep(max(retry_after or 0.0, back) + random.uniform(0, 0.25 * back))
        self._fails[host] = self._fails.get(host, 0) + 1
        if self._fails[host] >= 5:
            self._blocked_until[host] = self._clock() + 300.0
        raise FetchError(f"giving up on {url}: {last}")


class PlaywrightFetcher:
    """Optional: renders JavaScript-heavy public pages with headless Chromium (`pip install playwright` +
    `playwright install chromium`). Shares HttpFetcher's robots/rate-limit logic by delegating the
    permission check; use it only for pages the plain fetcher cannot read."""

    def __init__(self, http: HttpFetcher | None = None, wait_selector: str | None = None):
        self.http = http or HttpFetcher()
        self.wait_selector = wait_selector

    async def fetch(self, url: str) -> str:
        p = urllib.parse.urlparse(url)
        rp = await self.http._robots_for(p.scheme, p.netloc)
        if rp is None or not rp.can_fetch(self.http.ua, url):
            raise FetchError(f"robots.txt does not allow fetching {url}")
        await self.http._wait_turn(p.netloc, rp.crawl_delay(self.http.ua))
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:
            raise FetchError("playwright is not installed") from exc
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                ctx = await browser.new_context(user_agent=self.http.ua, java_script_enabled=True)
                page = await ctx.new_page()
                await page.goto(url, timeout=int(self.http.timeout * 1000), wait_until="domcontentloaded")
                if self.wait_selector:
                    await page.wait_for_selector(self.wait_selector, timeout=int(self.http.timeout * 1000))
                return (await page.content())[:config.OSINT_MAX_BODY_BYTES]
            finally:
                await browser.close()


# ====================================================================== parsers
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _strip_html(s: str) -> str:
    if not s:
        return ""
    try:
        from bs4 import BeautifulSoup
        return re.sub(r"\s+", " ", BeautifulSoup(s, "html.parser").get_text(" ")).strip()
    except ImportError:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)).strip()


def _parse_time(s):
    if not s:
        return None
    s = s.strip()
    try:
        d = email.utils.parsedate_to_datetime(s)
    except (TypeError, ValueError, IndexError):
        d = None
    if d is None:
        try:
            d = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            return None
    return d.replace(tzinfo=UTC) if d.tzinfo is None else d.astimezone(UTC)


def parse_rss(xml_text: str, source: str, max_items=50) -> list[RawPost]:
    """RSS 2.0 and Atom. Documents with a DOCTYPE/ENTITY declaration are rejected (XML entity-expansion attacks)."""
    head = xml_text[:4096].upper()
    if "<!DOCTYPE" in head or "<!ENTITY" in xml_text.upper():
        raise ValueError("feed rejected: DOCTYPE/ENTITY declarations are not allowed")
    root = ET.fromstring(xml_text)
    out = []
    for el in root.iter():
        if _local(el.tag) not in ("item", "entry"):
            continue
        f = {}
        images = []
        for ch in el:
            n = _local(ch.tag)
            if n in ("enclosure", "content", "thumbnail") and ch.get("url") and \
                    (ch.get("type", "image/").startswith("image") or n == "thumbnail"):
                images.append(ch.get("url"))
            elif n == "link":
                f["link"] = ch.get("href") or (ch.text or "").strip()
            elif n not in f:
                f[n] = (ch.text or "").strip()
        body = " ".join(filter(None, [_strip_html(f.get("title", "")),
                                      _strip_html(f.get("description") or f.get("summary") or f.get("encoded") or "")]))
        if not body:
            continue
        pid = f.get("guid") or f.get("id") or f.get("link") or hashlib.sha256(body.encode()).hexdigest()[:16]
        out.append(RawPost(platform=source, post_id=str(pid)[:200], text=body, url=f.get("link"),
                           author=f.get("creator") or f.get("author") or None,
                           posted_at=_parse_time(f.get("pubdate") or f.get("published") or f.get("updated")),
                           image_urls=images[:4]))
        if len(out) >= max_items:
            break
    return out


def parse_html(html: str, spec: SourceSpec) -> list[RawPost]:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("HTML sources need `pip install beautifulsoup4 lxml`") from exc
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")
    sel = {"item": "article", "text": None, "link": "a[href]", "time": "time", "image": "img", "author": None, **spec.selectors}
    out = []
    for node in soup.select(sel["item"])[:spec.max_items]:
        tnode = node.select_one(sel["text"]) if sel["text"] else node
        text = re.sub(r"\s+", " ", tnode.get_text(" ")).strip() if tnode else ""
        if not text:
            continue
        a = node.select_one(sel["link"]) if sel["link"] else None
        link = urllib.parse.urljoin(spec.url, a["href"]) if a and a.get("href") else None
        tm = node.select_one(sel["time"]) if sel["time"] else None
        when = _parse_time((tm.get("datetime") or tm.get_text()) if tm else None)
        imgs = [urllib.parse.urljoin(spec.url, i["src"]) for i in (node.select(sel["image"]) if sel["image"] else [])
                if i.get("src") and not i["src"].startswith("data:")][:4]
        au = node.select_one(sel["author"]) if sel["author"] else None
        out.append(RawPost(platform=spec.name, post_id=node.get("id") or link or hashlib.sha256(text.encode()).hexdigest()[:16],
                           text=text, url=link, author=au.get_text(strip=True) if au else None, posted_at=when,
                           image_urls=imgs))
    return out


# ====================================================================== mock stream (synthetic, offline)
_TEMPLATES = {
    "waterlogging": {
        "en": ["Knee deep water on the road in {area} since this morning, cars are stuck. Nobody from KMC has come.",
               "{area}: rainwater standing for two days, flooded street, people are fed up #KarachiRains",
               "Road completely flooded near {area}. Dangerous for bikes and children."],
        "roman-ur": ["{area} mein barsat ke baad pani jama hai, gaariyan phans gayi hain. Koi nahi sunta, kab tak?",
                     "{area} ki sadak par paani jama hai 2 din se, bohat pareshan hain log"],
        "ur": ["{area} میں بارش کا پانی جمع ہے، لوگ پریشان ہیں"]},
    "sewage_overflow": {
        "en": ["Sewage overflowing onto the street in {area}, the smell is unbearable and kids walk through it.",
               "Gutter overflow near {area} for weeks now, complaint ignored again"],
        "roman-ur": ["{area} mein gutter ubal raha hai, gandagi ka pani sadak par aa gaya hai. bohat pareshan hain."],
        "ur": ["{area} میں گٹر ابل رہا ہے، بدبو ناقابل برداشت ہے"]},
    "open_manhole": {
        "en": ["Manhole cover missing near {area}, very dangerous especially at night. Someone will fall in.",
               "Open manhole on the main road in {area}, urgent, a child almost fell"],
        "roman-ur": ["{area} ke paas manhole ka dhakkan nahi hai, bacha gir sakta hai, khatarnak hai"]},
    "pothole": {
        "en": ["Huge pothole on the main road in {area}, bikes are slipping, terrible condition.",
               "Another pothole near {area}. Accident waiting to happen."],
        "roman-ur": ["{area} mein sadak par bada gaddha hai, bikes phisal rahi hain, koi nahi sunta"]},
}
_NOISE = ["Great biryani in {area} tonight!", "Traffic moving smoothly near {area} this evening.",
          "Beautiful sunset over the sea in Karachi today", "Anyone know a good dentist in {area}?"]
_RESOLVED = ["The pothole in {area} has been fixed. Thanks to everyone who reported it!",
             "{area} gutter line cleaned and theek ho gaya, thank you"]


class MockStream:
    """Deterministic synthetic stand-in for a rate-limited public feed. Every post is flagged is_synthetic=True.
    Behaves like a real feed: some posts are irrelevant, some resolved, some are copy-paste duplicates,
    some carry image URLs or explicit coordinates, and incidents arrive as BURSTS from several different authors
    (that burst is what the fusion layer rewards - one loud author is not corroboration)."""

    def __init__(self, seed=11, rate_per_s=0.0, areas=None, hazards=None, now_fn=None, burst_prob=0.35):
        self.rng = random.Random(seed)
        self.rate = rate_per_s
        self.areas = areas or ["Gulshan-e-Iqbal", "Korangi", "Saddar", "Malir", "Nazimabad", "Lyari", "Clifton",
                               "Orangi Town", "North Nazimabad", "Gulistan-e-Johar"]
        self.hazards = hazards or list(_TEMPLATES)
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self._n = 0

    def _post(self, text, author_id=None, lat=None, lon=None, images=0):
        self._n += 1
        return RawPost(platform="mock", post_id=f"mock-{self._n:06d}", text=text,
                       author=f"citizen_{author_id if author_id is not None else self.rng.randint(1, 400)}",
                       posted_at=self._now() - dt.timedelta(minutes=self.rng.randint(0, 90)),
                       url=f"https://mock.invalid/post/{self._n}", lat=lat, lon=lon, is_synthetic=True,
                       image_urls=[f"https://mock.invalid/img/{self._n}_{i}.jpg" for i in range(images)])

    def burst(self, area, hazard, n_authors=4, lang_mix=True, with_images=1, with_coords=None):
        """n_authors DIFFERENT people reporting the same hazard in the same area (independent corroboration)."""
        posts = []
        langs = [l for l in _TEMPLATES[hazard] if l in _TEMPLATES[hazard]]
        for i in range(n_authors):
            lang = langs[i % len(langs)] if lang_mix else "en"
            tpl = self.rng.choice(_TEMPLATES[hazard][lang])
            lat, lon = with_coords if (with_coords and i == 0) else (None, None)
            posts.append(self._post(tpl.format(area=area), author_id=9000 + self.rng.randint(0, 99999) + i,
                                    lat=lat, lon=lon, images=with_images if i == 0 else 0))
        return posts

    def one(self):
        r = self.rng.random()
        area = self.rng.choice(self.areas)
        if r < 0.20:
            return [self._post(self.rng.choice(_NOISE).format(area=area))]
        if r < 0.28:
            return [self._post(self.rng.choice(_RESOLVED).format(area=area))]
        hazard = self.rng.choice(self.hazards)
        if r < 0.55:
            return self.burst(area, hazard, n_authors=self.rng.randint(3, 5))
        lang = self.rng.choice(list(_TEMPLATES[hazard]))
        return [self._post(self.rng.choice(_TEMPLATES[hazard][lang]).format(area=area), images=self.rng.randint(0, 1))]

    def duplicate_of(self, post: RawPost):
        """A copy-paste forward of an existing post by a different account."""
        return self._post(post.text + " pls share", author_id=self.rng.randint(5000, 9000))

    async def stream(self, n_posts: int | None = None) -> AsyncIterator[RawPost]:
        sent = 0
        while n_posts is None or sent < n_posts:
            batch = self.one()
            if batch and self.rng.random() < 0.15:
                batch.append(self.duplicate_of(batch[0]))
            for p in batch:
                yield p
                sent += 1
                if self.rate > 0:
                    await asyncio.sleep(1.0 / self.rate)
                if n_posts is not None and sent >= n_posts:
                    return


# ====================================================================== the pipeline
class OSINTPipeline:
    def __init__(self, conn, fetcher=None, verifier=None, *, fuse=True, now_fn=None, max_satellite_concurrency=2):
        self.conn = conn
        self.fetcher = fetcher or HttpFetcher()
        self.verifier = verifier
        self.fuse = fuse
        self._now = now_fn or (lambda: dt.datetime.now(UTC))
        self._dups = nlp.DuplicateDetector()
        self._sat_sem = asyncio.Semaphore(max_satellite_concurrency)
        with kpdb.db_lock:
            recent = conn.execute("SELECT id, text FROM osint_posts ORDER BY id DESC LIMIT ?", (config.OSINT_DEDUP_MEMORY,)).fetchall()
        self._dups.seed(reversed(recent))
        self.on_post: Callable[[dict], Awaitable[None] | None] | None = None   # optional UI hook

    # -- one post
    async def process(self, raw: RawPost, stats: PipelineStats | None = None):
        stats = stats if stats is not None else PipelineStats()
        text = nlp.redact(raw.text)[:2000]           # contact details AND coordinates are removed from the stored text
        if not text.strip():
            return None
        a = nlp.analyze_text(text, len(raw.image_urls))
        # Geocode from the ORIGINAL text (coordinates are redacted from `text`), then make any precise point private
        # BEFORE it is stored or handed to the satellite check. See geospatial/gazetteer.py (geo-indistinguishability).
        author = nlp.author_hash(raw.platform, raw.author)
        if raw.lat is not None and raw.lon is not None:
            geo = {"lat": raw.lat, "lon": raw.lon, "radius_m": 30.0, "method": "source", "place": None}
        else:
            geo = geocode_text(raw.text or "")
        geo = privatize_location(geo, author_hash=author, is_synthetic=raw.is_synthetic)
        near_dup = self._dups.check(text) if a["hazard_type"] else None
        row = dict(platform=raw.platform, post_id=raw.post_id, url=raw.url,
                   author_hash=author,
                   posted_at=(raw.posted_at or self._now()).astimezone(UTC).isoformat(timespec="seconds"),
                   scraped_at=kpdb.utc_now_iso(), text=text, lang=a["lang"], image_urls=raw.image_urls[:4],
                   hazard_type=a["hazard_type"], hazard_strength=a["hazard_strength"], urgency=a["urgency"],
                   negativity=a["negativity"], community_signal=a["community_signal"], status=a["status"],
                   keywords=a["keywords"], lat=geo and geo["lat"], lon=geo and geo["lon"],
                   geo_method=geo and geo["method"], geo_radius_m=geo and geo["radius_m"],
                   geo_place=geo and geo["place"], duplicate_of=near_dup, is_synthetic=raw.is_synthetic)
        post_id = await asyncio.to_thread(kpdb.insert_osint_post, self.conn, **row)
        stats.scraped += 1
        if post_id is None:
            stats.already_seen += 1
            return None
        stats.inserted += 1
        self._dups.add(post_id, text)
        stats.irrelevant += a["status"] == "irrelevant"
        stats.resolved += a["status"] == "resolved"
        stats.near_duplicates += near_dup is not None
        stats.geocoded += geo is not None
        row["id"] = post_id
        if self.on_post:
            res = self.on_post(row)
            if asyncio.iscoroutine(res):
                await res
        await self._maybe_verify(row, stats)
        return row

    async def _maybe_verify(self, row, stats):
        hz = row["hazard_type"]
        if not (self.verifier and hz and row["status"] == "active" and row["duplicate_of"] is None
                and row["lat"] is not None and row["community_signal"] >= config.OSINT_ALERT_THRESHOLD):
            return
        stats.alerts += 1
        if config.SAT_APPLICABILITY.get(hz, 0.0) <= 0.0:
            return                                 # pothole / manhole: not observable at 10 m, don't spend API calls
        async with self._sat_sem:
            ev = await asyncio.to_thread(self.verifier.verify, row["lat"], row["lon"], hz, row["posted_at"], row["id"],
                                         False, row["geo_radius_m"])
        if not ev.cached:
            stats.satellite_checks += 1
        row["satellite_verdict"] = ev.verdict

    # -- scraping
    async def scrape_source(self, spec: SourceSpec) -> list[RawPost]:
        fetcher = self.fetcher
        if spec.kind == "html_js":
            fetcher = PlaywrightFetcher(self.fetcher if isinstance(self.fetcher, HttpFetcher) else None)
        body = await fetcher.fetch(spec.url)
        if spec.kind == "rss":
            return parse_rss(body, spec.name, spec.max_items)
        if spec.kind in ("html", "html_js"):
            return parse_html(body, spec)
        raise ValueError(f"unknown source kind {spec.kind!r}")

    async def _refresh_matrix(self, stats):
        if self.fuse:
            from risk_engine import fusion
            cells = await asyncio.to_thread(fusion.assess_all, self.conn)
            stats.fused_cells = len(cells)

    async def run_once(self, sources: Iterable[SourceSpec]) -> PipelineStats:
        stats, sources = PipelineStats(), list(sources)
        results = await asyncio.gather(*(self.scrape_source(s) for s in sources), return_exceptions=True)
        for spec, res in zip(sources, results):
            if isinstance(res, Exception):
                stats.source_errors.append(f"{spec.name}: {res}")
                log.warning("source %s failed: %s", spec.name, res)
                continue
            for raw in res:
                await self.process(raw, stats)
        await self._refresh_matrix(stats)
        return stats

    async def run_stream(self, stream: AsyncIterator[RawPost], fuse_every: int = 25) -> PipelineStats:
        stats, n = PipelineStats(), 0
        async for raw in stream:
            await self.process(raw, stats)
            n += 1
            if n % fuse_every == 0:
                await self._refresh_matrix(stats)
        await self._refresh_matrix(stats)
        return stats

    async def run_forever(self, sources, interval_s=600.0, stop: asyncio.Event | None = None, max_cycles=None):
        stop, cycles = stop or asyncio.Event(), 0
        total = PipelineStats()
        while not stop.is_set() and (max_cycles is None or cycles < max_cycles):
            total.merge(await self.run_once(sources))
            cycles += 1
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval_s * random.uniform(0.9, 1.1))
            except asyncio.TimeoutError:
                pass
        return total


# ====================================================================== sync helpers for Streamlit / scripts
def run_sync(coro):
    """Runs a coroutine to completion from synchronous code (Streamlit script, CLI). Safe even if the calling
    thread already has an event loop (it uses a private thread in that case)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}

    def _t():
        try:
            box["v"] = asyncio.run(coro)
        except BaseException as e:   # noqa: BLE001 - re-raised in the caller
            box["e"] = e
    th = threading.Thread(target=_t, daemon=True)
    th.start()
    th.join()
    if "e" in box:
        raise box["e"]
    return box["v"]


def run_mock_demo(conn, verifier=None, n_posts=60, seed=11, fuse=True) -> PipelineStats:
    """One-call offline demo: simulated posts -> analysis -> (satellite) -> fusion. Everything is flagged synthetic."""
    pipe = OSINTPipeline(conn, fetcher=None, verifier=verifier, fuse=fuse)
    return run_sync(pipe.run_stream(MockStream(seed=seed).stream(n_posts)))


class BackgroundRunner:
    """Keeps the polite scraper running in a daemon thread next to the Streamlit app.
    Use st.cache_resource so one runner survives script reruns."""

    def __init__(self, conn, sources, verifier=None, interval_s=600.0):
        self.conn, self.sources, self.verifier, self.interval_s = conn, sources, verifier, interval_s
        self.stats = PipelineStats()
        self.running = False
        self._thread = None
        self._loop = None
        self._stop = None

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._main, daemon=True, name="osint-runner")
        self._thread.start()

    def _main(self):
        async def go():
            self._stop = asyncio.Event()
            self._loop = asyncio.get_running_loop()
            pipe = OSINTPipeline(self.conn, verifier=self.verifier)
            while not self._stop.is_set():
                self.stats.merge(await pipe.run_once(self.sources))
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=self.interval_s)
                except asyncio.TimeoutError:
                    pass
        try:
            asyncio.run(go())
        finally:
            self.running = False

    def stop(self):
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)


# ====================================================================== CLI
def _main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m scripts.osint_pipeline")
    ap.add_argument("--mock", type=int, metavar="N", help="process N synthetic posts (offline)")
    ap.add_argument("--sources", help="JSON file of SourceSpec objects (see data/osint_sources.example.json)")
    ap.add_argument("--once", action="store_true"); ap.add_argument("--interval", type=float, default=600.0)
    ap.add_argument("--satellite", choices=["off", "live", "synthetic"], default="off",
                    help="cross-verify alerts with Sentinel-2 ('live' needs pystac-client planetary-computer rasterio)")
    ap.add_argument("--db", default=config.DB_PATH)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")
    conn = kpdb.connect(a.db)
    verifier = None
    if a.satellite != "off":
        from geospatial import satellite as sat
        verifier = sat.SatelliteVerifier(conn, sat.PlanetaryComputerProvider() if a.satellite == "live"
                                         else sat.SyntheticSatelliteProvider())
    if a.mock:
        print(json.dumps(run_mock_demo(conn, verifier, a.mock).as_dict(), indent=2))
        return
    if not a.sources:
        ap.error("give --mock N or --sources FILE")
    pipe = OSINTPipeline(conn, verifier=verifier)
    specs = load_sources(a.sources)
    if a.once:
        print(json.dumps(run_sync(pipe.run_once(specs)).as_dict(), indent=2))
    else:
        try:
            run_sync(pipe.run_forever(specs, a.interval))
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    _main()
