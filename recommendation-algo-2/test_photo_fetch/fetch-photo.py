#!/usr/bin/env python3
"""
fetch_photo.py  •  Auto-attach attractive campus photos to a CSV of universities.

Heuristics
----------
✓ +30 if the file is a Featured / Quality / Valued picture on Commons
✓ +10 if title/description contains campus-ish words
✗ −20 if title/description contains unwanted words (logo, seal, statue…)
✗ −50 if width  < 1 000 px
✓ +2  per Wikipedia use   (capped at +20)

The highest-scoring image ≥ 0 wins.  Otherwise we fall back to Openverse.

Usage
-----
$ pip install pandas aiohttp rapidfuzz tqdm python-slugify
$ python3 fetch_photo.py
$ python3 fetch_photo.py --infile my.csv --column "University Name" --outfile out.csv
"""

import argparse, asyncio, re, sys, urllib.parse
from pathlib import Path

import aiohttp, pandas as pd
from rapidfuzz import fuzz, process
from slugify import slugify
from tqdm.asyncio import tqdm_asyncio as tqdm

# ────────────────────────────── CONFIG ──────────────────────────────
WIKI_API   = "https://www.wikidata.org/w/api.php"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE  = "https://api.openverse.org/v1/images"
UA         = "fetch-photo-script/0.3 (+email)"
CONCURRENCY = 6        # requests in flight
MIN_W      = 1000      # reject tiny files

GOOD = re.compile(r"\b(campus|main building|quad|library|hall|aerial|panorama)\b", re.I)
BAD  = re.compile(r"\b(logo|seal|crest|statue|construction|map|floor plan)\b", re.I)

# ─────────────────────── Commons helper functions ───────────────────
async def wiki_search(session, query):
    """Return the best matching Q-id for *query* (or None)."""
    p = dict(action="wbsearchentities", format="json", language="en",
             type="item", limit=7, search=query)
    async with session.get(WIKI_API, params=p) as r:
        r.raise_for_status()
        hits = (await r.json()).get("search", [])
    if not hits:
        return None
    labels = {h["id"]: h["label"] for h in hits}
    best_id, score, _ = process.extractOne(query, labels, scorer=fuzz.token_set_ratio)
    return best_id if score >= 75 else None

async def get_claim(session, qid, pid):
    """Return list of raw values for a Wikidata claim property."""
    p = dict(action="wbgetclaims", format="json", entity=qid, property=pid)
    async with session.get(WIKI_API, params=p) as r:
        r.raise_for_status()
        data = await r.json()
    claims = data.get("claims", {}).get(pid, [])
    vals   = [c["mainsnak"]["datavalue"]["value"] for c in claims]
    return vals

def commons_thumb(filename, width):
    return f"https://commons.wikimedia.org/wiki/Special:FilePath/{filename}?width={width}"

async def commons_ranked_images(session, category, width):
    """Yield (score, thumb_url) for up to 50 files in a Commons category."""
    p = dict(action="query", format="json", generator="categorymembers",
             gcmtitle=f"Category:{category}", gcmtype="file",
             gcmnamespace=6, gcmlimit=50,
             prop="imageinfo|globalusage",
             iiprop="url|size|extmetadata",
             iiurlwidth=str(width))
    async with session.get(COMMONS_API, params=p) as r:
        if r.status == 404:
            return []
        r.raise_for_status()
        data = await r.json()

    results = []
    for page in data.get("query", {}).get("pages", {}).values():
        info  = page["imageinfo"][0]
        title = page["title"]
        desc  = info["extmetadata"].get("ImageDescription", {}).get("value", "")
        cats  = info["extmetadata"].get("Categories", {}).get("value", "")
        w     = info["width"]

        score = 0
        if "featured pictures" in cats or "quality images" in cats or "valued images" in cats:
            score += 30
        if GOOD.search(title) or GOOD.search(desc):
            score += 10
        if BAD.search(title)  or BAD.search(desc):
            score -= 20
        if w < MIN_W:
            score -= 50
        usage = len(page.get("globalusage", []))
        score += min(usage * 2, 20)

        results.append((score, info["thumburl"]))
    # highest score first
    return sorted(results, key=lambda t: t[0], reverse=True)

# ─────────────────────── Openverse fallback ────────────────────────
async def openverse_image(session, name, width):
    p = dict(
        q=f"{name} campus main building",
        license="cc0,by,by-sa",
        size="large",
        aspect_ratio="wide",
        page_size=1,
    )
    try:
        async with session.get(OPENVERSE, params=p) as r:
            if r.status >= 400:
                return None
            data = await r.json()
    except aiohttp.ClientError:
        return None
    return data["results"][0]["url"] if data["results"] else None

# ────────────────────────── Resolver core ──────────────────────────
async def resolve_one(session, name, width):
    """Return best CDN URL for *name* or empty string."""
    qid = await wiki_search(session, name)
    if not qid:
        return await openverse_image(session, name, width) or ""

    # First: check P18 image itself
    p18_vals = await get_claim(session, qid, "P18")
    if p18_vals:
        first = commons_thumb(urllib.parse.quote(p18_vals[0].replace(" ", "_")), width)
        # quick BAD word test
        if not BAD.search(first):
            return first

    # Next: scan Commons category (P373 or P3602)
    for pid in ("P373", "P3602"):
        cats = await get_claim(session, qid, pid)
        if not cats:
            continue
        cat = cats[0]
        ranked = await commons_ranked_images(session, cat, width)
        if ranked and ranked[0][0] >= 0:
            return ranked[0][1]

    # Fallback
    return await openverse_image(session, name, width) or ""

async def resolve_all(names, width):
    resolved = {}
    sem      = asyncio.Semaphore(CONCURRENCY)
    timeout  = aiohttp.ClientTimeout(total=20)

    async with aiohttp.ClientSession(headers={"User-Agent": UA}, timeout=timeout) as sess:
        async def task(n):
            async with sem:
                resolved[n] = await resolve_one(sess, n, width)
        await tqdm.gather(*(task(n) for n in names), total=len(names))
    return resolved

# ─────────────────────────────── CLI ───────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Attach campus photo URLs to a CSV")
    ap.add_argument("--infile",  default="programs_cleaned.csv")
    ap.add_argument("--outfile", default="programs_with_images.csv")
    ap.add_argument("--column",  default="names", help="column holding university names")
    ap.add_argument("--width",   type=int, default=800, help="desired thumb width")
    args = ap.parse_args()

    if not Path(args.infile).exists():
        sys.exit(f"❌  File {args.infile} not found.")

    df = pd.read_csv(args.infile)
    if args.column not in df.columns:
        sys.exit(f"❌  Column {args.column!r} not in CSV. Found: {list(df.columns)}")

    names = df[args.column].dropna().astype(str).unique().tolist()
    print(f"→  Resolving {len(names)} unique names …")

    image_map = asyncio.run(resolve_all(names, args.width))
    df["cdn_link"] = df[args.column].map(image_map).fillna("")

    df.to_csv(args.outfile, index=False)
    hits = (df["cdn_link"] != "").sum()
    print(f"✅  {hits}/{len(df)} rows now have an image.  Saved ➜ {args.outfile}")

if __name__ == "__main__":
    main()