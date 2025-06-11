"""
build_config.py
—————————————————————
Generate config.yaml with:
 • 11 Sector-SPDR ETFs
 • top-10 holdings per sector (from SSGA XLSX)
 • one high-signal keyword per sector (StockNewsAPI)
Requires: python-dotenv, pandas, requests, pyyaml, openpyxl
"""

import os, re, csv, requests, tempfile, yaml, pandas as pd
from collections import Counter
from dotenv import load_dotenv

# 0 ────────────────────────────────────────────────────────────────────────────
load_dotenv()
STOCKNEWS_KEY = os.getenv("STOCKNEWS_API_KEY")  # put in .env
if not STOCKNEWS_KEY:
    raise ValueError("STOCKNEWS_API_KEY missing from .env")

# 1 ────────────────────────────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Real Estate": "XLRE",
}

XLSX_BASE = (
    "https://www.ssga.com/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx"
)

# 2 ────────────────────────────────────────────────────────────────────────────
STOP = {
    "the","and","for","with","that","from","this","are","will","into","more","than",
    "has","over","amid","as","by","about","new","its","in","on","of","to","at","a",
    "is","sector","stocks","stock","market","shares","etf","etfs","trading",
    "company","companies","industry","industries"
}

def stem(word: str) -> str:
    """Simple heuristic stemmer (no external libs)."""
    for suf in ("ing","ers","ies","ied","tion","ions","ed","es","s"):
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)]
    return word

def extract_keyword(text: str) -> str:
    """Return the single most frequent non-stopword (or bigram)."""
    words = re.sub(r"[^A-Za-z0-9 ]", " ", text).lower().split()
    words = [stem(w) for w in words if w not in STOP and len(w) > 3]
    bigrams = [" ".join(pair) for pair in zip(words, words[1:])]
    for w, _ in Counter(words + bigrams).most_common(20):
        if not w.isnumeric():
            return w
    return ""

def top10_holdings(ticker: str) -> list:
    url = XLSX_BASE.format(ticker=ticker.lower())
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
        tmp.write(resp.content)
        tmp.flush()
        df = pd.read_excel(tmp.name, skiprows=4, usecols=["Ticker"])
    return df["Ticker"].head(10).str.replace(" ", "").tolist()

def one_keyword(etf: str) -> str:
    url = (
        "https://stocknewsapi.com/api/v1"
        f"?tickers={etf}&items=100&token={STOCKNEWS_KEY}"
    )
    data = requests.get(url, timeout=15).json().get("data", [])
    combined = " ".join(a.get("title","")+" "+a.get("text","") for a in data)
    kw = extract_keyword(combined)
    return kw or etf  # fallback: ETF ticker

# 3 ────────────────────────────────────────────────────────────────────────────
config = {"sectors": [], "max_firms_per_sector": 10}

for sector, etf in SECTOR_ETFS.items():
    print(f"Processing {sector} …")
    config["sectors"].append(
        {
            "name": sector,
            "etf": etf,
            "keyword": one_keyword(etf),
            "override_tickers": top10_holdings(etf),
        }
    )

# 4 ────────────────────────────────────────────────────────────────────────────
with open("config.yaml", "w") as f:
    yaml.dump(config, f, sort_keys=False)

print("✅  config.yaml written with 10 tickers + 1 keyword per sector")