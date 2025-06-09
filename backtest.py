#!/usr/bin/env python3
"""
Backtest Stooq EOD árak alapján (nincs API-kulcs, nincs rate-limit)

Kimenet:
    outputs/backtest_equity.json
    outputs/backtest_stats.json
"""
import importlib, types
try:
    import distutils
except ModuleNotFoundError:
    import types, sys
    import setuptools._distutils as _d
    sys.modules['distutils'] = _d
    sys.modules['distutils.version'] = _d.version
    
import json, datetime, pandas as pd
from pathlib import Path
from pandas_datareader import data as pdr   # ← Stooq forrás

BASE = Path(__file__).resolve().parent
PORT = json.load(open(BASE / "outputs/portfolio_latest.json"))["table"]

# ---- Paraméterek -----------------------------------------------------------
START = "2023-01-01"
END   = datetime.date.today().isoformat()
BENCH = "SPY"
UNIT  = 1_000_000        # induló portfólió USD

# ---- Ticker & Weight tisztítás ---------------------------------------------
def clean(t): return t.replace("\xa0", "").strip().upper()
weights = {clean(r["Asset"]): float(r["Weight (%)"])/100 for r in PORT}
tickers = list(weights) + [BENCH]

# Stooq ticker formátum: "AAPL.US"
stooq_syms = [t + ".US" for t in tickers]

print("→ Letöltés Stooq-ról …")
px = (
    pdr.DataReader(stooq_syms, "stooq", START, END)["Close"]
    .rename(columns=lambda c: c.split(".")[0])   # "AAPL.US" → "AAPL"
    .dropna(how="all")
)

# ---- Hiányzó ticker(ek) kezelése -------------------------------------------
missing = [t for t in weights if t not in px.columns]
if missing:
    print(f"⚠️  Hiányzó árfolyam: {missing} – súlyok törlése")
    for t in missing: weights.pop(t)
if not weights:
    raise RuntimeError("Nincs érvényes árfolyam – backtest megszakítva.")

# ---- Portfolio equity -------------------------------------------------------
alloc_qty = {t: weights[t] * UNIT / px[t].iloc[0] for t in weights}
equity = (px[list(weights)] * pd.Series(alloc_qty)).sum(axis=1)
bench  = px[BENCH] / px[BENCH].iloc[0] * UNIT

# ---- Statisztikák -----------------------------------------------------------
def cagr(ts):
    yrs = (ts.index[-1] - ts.index[0]).days / 365.25
    return (ts.iloc[-1] / ts.iloc[0]) ** (1 / yrs) - 1

def max_dd(ts):
    roll = ts.cummax()
    return (ts / roll - 1).min()

def sharpe(ts):
    ret = ts.pct_change().dropna()
    return (ret.mean() / ret.std()) * (252 ** 0.5)

stats = {
    "Portfolio": {
        "CAGR":   round(cagr(equity) * 100, 2),
        "MaxDD":  round(max_dd(equity) * 100, 2),
        "Sharpe": round(sharpe(equity), 2),
    },
    "Benchmark": {
        "CAGR":   round(cagr(bench) * 100, 2),
        "MaxDD":  round(max_dd(bench) * 100, 2),
        "Sharpe": round(sharpe(bench), 2),
    },
}

# ---- Mentés -----------------------------------------------------------------
out_dir = BASE / "outputs"; out_dir.mkdir(exist_ok=True)
pd.DataFrame({"Portfolio": equity, BENCH: bench}).to_json(
    out_dir / "backtest_equity.json",
    orient="split",
    date_format="iso",
)
json.dump(stats, open(out_dir / "backtest_stats.json", "w"), indent=2)

print("✅ Backtest kész • files in outputs/")