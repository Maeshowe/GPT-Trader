#!/usr/bin/env python3
"""
Risk-budget + Black-Litterman overlay

1. 60-napos rolling volatilitás alapján inverse-vol súlyok (defenzív)
2. BL-modell: egyensúlyi súly = inv-vol, „vélemény” = firm_score-ből származó
   relatív hozamelvárás (0–3% skálázva)
3. 50-50 blend → végső risk-budget súlyok

Kimenet: outputs/portfolio_risk_budget.json   [{ticker, weight %}, …]
"""

import json, datetime, numpy as np, pandas as pd
from pathlib import Path
from pandas_datareader import data as pdr

# ── Fájl-útvonalak ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
PORT_FILE  = BASE / "outputs/portfolio_latest.json"
FIRM_FILE  = BASE / "inputs/firm_inputs.json"
OUT_FILE   = BASE / "outputs/portfolio_risk_budget.json"

# ── Beállítások ─────────────────────────────────────────────────────────────
START  = "2024-01-01"
END    = datetime.date.today().isoformat()
VOL_WIN = 60          # nap
BL_TAU  = 0.05
VIEW_SCALE = 0.03     # max 3% excess return
BL_BLEND   = 0.5      # 0=csak inv-vol, 1=csak BL

# ── Adatok beolvasása ───────────────────────────────────────────────────────
PORT = json.load(open(PORT_FILE))["table"]
FIRMS = json.load(open(FIRM_FILE))

tickers = [row["Asset"].strip().upper() for row in PORT]
score_map = {f["ticker"].upper(): f["firm_score"] or 0 for f in FIRMS}

# ── Árfolyamok Stooq-ról ────────────────────────────────────────────────────
px = pdr.DataReader([t + ".US" for t in tickers], "stooq", START, END)["Close"]
px.columns = [c.split(".")[0] for c in px.columns]
px = px.dropna(how="all")

# Hiányzó árak – töröljük a tickert és a score-t is
missing = [t for t in tickers if t not in px.columns]
if missing:
    print("⚠️  Hiányzó árfolyam:", missing)
    px = px.drop(columns=missing, errors="ignore")
    tickers = [t for t in tickers if t not in missing]

# ── 1) Inverse-vol súlyok ───────────────────────────────────────────────────
vol = px.pct_change().rolling(VOL_WIN).std().iloc[-1]
inv_vol_w = (1 / vol) / (1 / vol).sum()

# ── 2) Black-Litterman számítás ────────────────────────────────────────────
#   — Egyensúlyi súly (m): inv-vol
#   — Nézetek: firm_score → 0-1 skála → *VIEW_SCALE hozamelvárás
raw_scores = np.array([score_map.get(t, 0) for t in tickers])
min_s, max_s = raw_scores.min(), raw_scores.max()
views = (raw_scores - min_s) / (max_s - min_s) if max_s > min_s else raw_scores
Q = views * VIEW_SCALE                # várt excess return

# Szórás-mátrix évesítve
cov = px.pct_change().cov().values * 252
P   = np.eye(len(tickers))
tau = BL_TAU
eq_w = inv_vol_w.values
Pi = cov @ eq_w                       # piaci risk premium (proxy)

# BL zárt formula
M = np.linalg.inv(np.linalg.inv(tau * cov) + P.T @ P / 0.25)
adj_ret = M @ (np.linalg.inv(tau * cov) @ Pi + P.T @ Q / 0.25)

# Max-Sharpe portfólió (risk-aversion = 1)
w_bl = np.linalg.inv(cov) @ adj_ret
w_bl = w_bl / w_bl.sum()              # normálás 1-re

# ── 3) Végső blend ─────────────────────────────────────────────────────────
w_final = BL_BLEND * w_bl + (1 - BL_BLEND) * inv_vol_w
w_final = w_final / w_final.sum()

# ── Mentés ──────────────────────────────────────────────────────────────────
out = [
    {"ticker": t, "weight": round(float(w) * 100, 2)}
    for t, w in zip(tickers, w_final)
]

Path(OUT_FILE).parent.mkdir(exist_ok=True)
json.dump(out, open(OUT_FILE, "w"), indent=2)
print("✅ Risk-budget súlyok mentve →", OUT_FILE.relative_to(BASE))