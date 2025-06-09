#!/usr/bin/env python3
"""
Egyszerűsített input-generálás:
  • sector_inputs.json  – minden szektor (config.yaml)
  • firm_inputs.json    – csak override_tickers alapján
Nem hív Yahoo holdings-API-t / scrape-et!
"""

import os, json, datetime, requests, yfinance as yf
from pathlib import Path
import yaml
from dotenv import load_dotenv
from fredapi import Fred

BASE = Path(__file__).resolve().parent.parent   # projekt gyökere
load_dotenv(override=True)

# --- API-kulcsok
FRED_KEY      = os.getenv("FRED_API_KEY")
STOCKNEWS_KEY = os.getenv("STOCKNEWS_API_KEY")
fred = Fred(api_key=FRED_KEY)

CFG   = yaml.safe_load(open(BASE / "config.yaml"))
TODAY = datetime.date.today().isoformat()

# ---------- Helper függvények -----------------------------------------------
def fred_latest(series_id):
    return float(fred.get_series_latest_release(series_id).dropna().iloc[-1])

def macro_indicators():
    return {
        "GDP": round(fred_latest("GDP") / 1_000, 2),   # USD-billion
        "CPI": round(fred_latest("CPIAUCSL"), 2),
        "Unemployment": round(fred_latest("UNRATE"), 2),
        "InterestRate": round(fred_latest("FEDFUNDS"), 2),
    }

def stocknews(ticker_or_kw, items=3):
    url = "https://stocknewsapi.com/api/v1"
    params = {"tickers": ticker_or_kw, "items": items, "token": STOCKNEWS_KEY}
    resp = requests.get(url, params=params, timeout=30)
    return [a["title"] for a in resp.json().get("data", [])] if resp.ok else []

def firm_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    mapping = {
        "trailingPE": "P/E",
        "pegRatio": "PEG",
        "beta": "Beta",
        "returnOnEquity": "ROE",
        "quickRatio": "Quick Ratio",
    }
    out = {}
    for k, new in mapping.items():
        v = info.get(k)
        if v is not None:
            out[new] = round(v, 2)
    return out

# ---------- Main ------------------------------------------------------------
def main():
    sector_inputs, firm_inputs = [], []
    macro_json = macro_indicators()

    for s in CFG["sectors"]:
        # --- Sector record ------------------------------
        sector_inputs.append({
            "name": s["name"],
            "macro_indicators_json": macro_json,
            "sector_news_snippets": stocknews(s["keyword"]),
            "today": TODAY,
            "sector_score": ""
        })

        # --- Firm records (csak override_tickers) -------
        tickers = s.get("override_tickers", [])
        for tkr in tickers:
            firm_inputs.append({
                "sector": s["name"],
                "ticker": tkr,
                "company_name": tkr,
                "industry": s["name"].title(),
                "firm_financials_json": firm_fundamentals(tkr),
                "firm_news_snippets": stocknews(tkr),
                "today": TODAY,
                "firm_score": ""
            })

    Path(BASE / "inputs").mkdir(exist_ok=True)
    json.dump(sector_inputs, open(BASE / "inputs/sector_inputs.json", "w"), indent=2)
    json.dump(firm_inputs,   open(BASE / "inputs/firm_inputs.json",   "w"), indent=2)
    print(f"✅ {len(sector_inputs)} sector, {len(firm_inputs)} firm input mentve.")

if __name__ == "__main__":
    main()