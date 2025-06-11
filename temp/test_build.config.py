"""
build_config.py  –  YAML-generálás 1 kulcsszó + 10 ticker / szektor
StockNews ➜ jelölt kulcsszavak  ➜ Brave Search cross-check
"""

import os, re, requests, tempfile, pandas as pd, yaml
from collections import Counter
from dotenv import load_dotenv

# ── ENV ───────────────────────────────────────────────────────────
load_dotenv()
STOCKNEWS_KEY = os.getenv("STOCKNEWS_API_KEY")
BRAVE_KEY     = os.getenv("BRAVE_SEARCH_API_KEY")
assert STOCKNEWS_KEY and BRAVE_KEY, "API keys missing in .env"

# ── ETF lista ─────────────────────────────────────────────────────
SECTOR_ETFS = {
    "Energy":"XLE","Materials":"XLB","Industrials":"XLI","Utilities":"XLU",
    "Healthcare":"XLV","Financials":"XLF","Consumer Discretionary":"XLY",
    "Consumer Staples":"XLP","Information Technology":"XLK",
    "Communication Services":"XLC","Real Estate":"XLRE"
}

XLS_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{t}.xlsx"

# ── Szótő + stop lista ────────────────────────────────────────────
STOP = set("""the and for with that from this are will into more than has over amid
as by about new its in on of to at a is sector stocks stock market shares etf etfs
trading company companies industry industries energy material materials industrial
industrials utility utilities healthcare health financial financials consumer
discretionary staples information technology tech communication services real
estate reit reits earnings earning""".split())

def stem(w: str) -> str:
    for suf in ("ing","ers","ies","ied","tion","ions","ed","es","s"):
        if w.endswith(suf) and len(w) > len(suf)+2:
            return w[:-len(suf)]
    return w

# ── Kulcsszó jelöltek StockNews-ból ───────────────────────────────
def stocknews_candidates(etf: str, n=5):
    url = f"https://stocknewsapi.com/api/v1?tickers={etf}&items=100&token={STOCKNEWS_KEY}"
    data = requests.get(url, timeout=15).json().get("data", [])
    text = " ".join(a.get("title","")+" "+a.get("text","") for a in data)
    words = [stem(w) for w in re.sub(r"[^A-Za-z0-9 ]"," ", text).lower().split()
             if w not in STOP and len(w) > 3]
    bigrams = [" ".join(p) for p in zip(words, words[1:])]
    cand = [w for w,_ in Counter(bigrams+words).most_common(n*3)]
    return cand

# ── Brave keresés cross-check ─────────────────────────────────────
DOMAINS = ["reuters.com","ft.com","wsj.com","barrons.com","iea.org",
           "mckinsey.com","deloitte.com","bloomberg.com","fidelity.com",
           "businessinsider.com"]

def keyword_passes(keyword: str, sector: str) -> bool:
    q = f'"{keyword}" "{sector}" 2025 ' + " OR ".join(f'site:{d}' for d in DOMAINS)
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept":"application/json","X-Subscription-Token":BRAVE_KEY}
    res = requests.get(url, params={"q":q,"count":20}, headers=headers, timeout=15)
    hits = res.json().get("web", {}).get("results", [])
    return len(hits) >= 2   # min. két találat

def pick_keyword(sector:str, etf:str):
    for kw in stocknews_candidates(etf):
        if keyword_passes(kw, sector):
            return kw
    return sector.lower().replace(" ","-")   # fallback

# ── Top10 ticker SPDR-XLSX ────────────────────────────────────────
def top10(etf:str):
    resp = requests.get(XLS_URL.format(t=etf.lower()), timeout=15).content
    with tempfile.NamedTemporaryFile(suffix=".xlsx") as tmp:
        tmp.write(resp); tmp.flush()
        df = pd.read_excel(tmp.name, skiprows=4, usecols=["Ticker"])
    return df["Ticker"].head(10).str.replace(" ","").tolist()

# ── YAML építés ───────────────────────────────────────────────────
cfg = {"sectors":[], "max_firms_per_sector":10}

for name, etf in SECTOR_ETFS.items():
    print(f"{name}: building entry…")
    cfg["sectors"].append({
        "name": name,
        "etf": etf,
        "keyword": pick_keyword(name, etf),
        "override_tickers": top10(etf)
    })

with open("config.yaml","w") as f:
    yaml.dump(cfg, f, sort_keys=False)
print("✅  config.yaml ready (1 validated keyword + 10 tickers / sector)")