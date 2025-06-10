#!/usr/bin/env python3
"""
config_autofill.py
~~~~~~~~~~~~~~~~~~
• Beolvassa a projekt gyökerében lévő config.yaml-t
• Szektoronként:
    1) automatic_ticker_discovery.discover_tickers() → override_tickers
    2) StockNews API 50 cikk → kulcsszó-kivonás → keyword
• Biztonsági mentés: config.yaml.bak
"""

from __future__ import annotations
import os, re, yaml, json, logging, datetime
from pathlib import Path
from collections import Counter
from string import punctuation

import requests
from dotenv import load_dotenv

from data_fetch.automatic_ticker_discovery import discover_tickers

# ───────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
CFG    = BASE / "config.yaml"
BACKUP = BASE / f"config.yaml.bak.{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
load_dotenv(override=True)

STOCKNEWS_KEY = os.getenv("STOCKNEWS_API_KEY")
if not STOCKNEWS_KEY:
    raise RuntimeError("STOCKNEWS_API_KEY missing in environment")

log = logging.getLogger(__name__)

# ─── Helper: kulcsszó-kivonás ─────────────────────────────
STOPWORDS = set("""
a az és vagy de hogy ha is egy egyes egyéb nincs van volt lesz vannak voltak
the for with this that from over under into on at of in by to an as be it its
""".split())

def _clean(word: str) -> str:
    return re.sub(rf"[{re.escape(punctuation)}0-9]", "", word.lower())

def extract_keyword(text: str) -> str:
    """Egyszerű TF-számlálós kulcsszó – 4+ betű, nem stopword."""
    words = [_clean(w) for w in text.split()]
    words = [w for w in words if len(w) > 3 and w not in STOPWORDS]
    if not words:
        return "General"
    common = Counter(words).most_common(1)[0][0]
    return common.title()

def stocknews_articles(ticker: str, items: int = 50) -> list[dict]:
    url = "https://stocknewsapi.com/api/v1"
    params = {"tickers": ticker, "items": items, "token": STOCKNEWS_KEY}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])

# ─── Main ─────────────────────────────────────────────────
def main(top_n: int = 10) -> None:
    cfg = yaml.safe_load(open(CFG))
    for sector in cfg["sectors"]:
        etf = sector["etf"]
        # 1) Ticker-lista (ETF holdings)
        tickers = discover_tickers(etf, top_n)
        sector["override_tickers"] = tickers
        # 2) Kulcsszó (StockNews)
        articles = stocknews_articles(etf, 50)
        text = " ".join(a.get("title","") + " " + a.get("text","") for a in articles)
        sector["keyword"] = extract_keyword(text) if text else sector.get("keyword", "General")
        log.info("✓ %s → %s tickers, keyword='%s'", sector["name"], len(tickers), sector["keyword"])

    # Biztonsági mentés & írás
    Path(BACKUP).write_text(Path(CFG).read_text(), encoding="utf-8")
    yaml.safe_dump(cfg, open(CFG, "w"), sort_keys=False, allow_unicode=True)
    print(f"✅ config.yaml frissítve – backup: {BACKUP.name}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()