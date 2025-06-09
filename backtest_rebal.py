#!/usr/bin/env python3
"""
Back-test két stratégiára ugyanazzal a Stooq-price feed-del
(1) Buy-&-hold   (BH)
(2) Havi rebalansz hónap-utolsó kereskedési napján   (REB)

• Bemenet : outputs/portfolio_latest.json   (Weight %)
• Kimenet : outputs/backtest_rebal_equity.json
            outputs/backtest_rebal_stats.json
"""
import warnings, pandas as pd
warnings.simplefilter("ignore", FutureWarning)

import json, datetime, pandas as pd
from pathlib import Path
from pandas_datareader import data as pdr

BASE   = Path(__file__).resolve().parent
PORT   = json.load(open(BASE/"outputs/portfolio_latest.json"))["table"]

# ── Paraméterek ─────────────────────────────────────────────────────────────
START = "2023-01-01"
END   = datetime.date.today().isoformat()
BENCH = "SPY"
UNIT  = 1_000_000                    # induló portfólió USD

# ── Ticker & Weight tisztítás ───────────────────────────────────────────────
clean  = lambda s: s.replace("\xa0", "").strip().upper()
weights0 = {clean(r["Asset"]): float(r["Weight (%)"])/100 for r in PORT}
tickers  = list(weights0) + [BENCH]

# ── Árfolyamok Stooq-ról ────────────────────────────────────────────────────
def stooq(tks):
    df = pdr.DataReader([t + ".US" for t in tks], "stooq", START, END)["Close"]
    df.columns = [c.split(".")[0] for c in df.columns]
    return df.sort_index()

px = stooq(tickers).dropna(how="all")
missing = [t for t in weights0 if t not in px.columns]
for t in missing: weights0.pop(t)
if not weights0:
    raise RuntimeError("No valid price data for portfolio tickers")

# ── Helper: equity-curve számítása súlysorozatból ───────────────────────────
def equity_from_weights(price_df: pd.DataFrame, weight_df: pd.DataFrame):
    """price_df: daily Close; weight_df: daily weights (sorösszeg =1)"""
    w_aligned = weight_df.reindex(price_df.index).fillna(method="ffill")
    daily_ret = price_df.pct_change().fillna(0)
    port_ret  = (w_aligned * daily_ret).sum(axis=1)
    equity    = (1 + port_ret).cumprod() * UNIT
    return equity

# ── (1) Buy-&-hold (BH) ────────────────────────────────────────────────────
alloc_qty = {t: weights0[t] * UNIT / px[t].iloc[0] for t in weights0}
bh_equity = (px[list(weights0)] * pd.Series(alloc_qty)).sum(axis=1)

# ── (2) Havi rebalansz (REB) ───────────────────────────────────────────────
#   • minden hónap utolsó valid kereskedési napján weights0 szerint újrasúlyoz
month_ends = px.index.to_series().groupby(px.index.to_period("M")).last()
w_rebal = pd.DataFrame(index=month_ends, columns=weights0.keys()).fillna(0.0)
for t in weights0: w_rebal[t] = weights0[t]          # fix súly-profil
reb_equity = equity_from_weights(px[list(weights0)], w_rebal)

# ── Benchmark (BH-stílusú SPY) ─────────────────────────────────────────────
bench = px[BENCH] / px[BENCH].iloc[0] * UNIT

# ── Metrikák ────────────────────────────────────────────────────────────────
def stats(ts):
    yrs = (ts.index[-1] - ts.index[0]).days / 365.25
    cagr   = (ts.iloc[-1]/ts.iloc[0])**(1/yrs) - 1
    dd     = (ts/ts.cummax() - 1).min()
    sharpe = ((ts.pct_change().dropna()).agg(["mean","std"])).pipe(
        lambda s: (s["mean"]/s["std"])*252**0.5 if s["std"] else 0
    )
    return {"CAGR": round(cagr*100,2), "MaxDD": round(dd*100,2), "Sharpe": round(sharpe,2)}

stats_out = {
    "Buy&Hold":  stats(bh_equity),
    "Rebalance": stats(reb_equity),
    "Benchmark": stats(bench)
}

# ── Mentés ──────────────────────────────────────────────────────────────────
out_dir = BASE/"outputs"; out_dir.mkdir(exist_ok=True)
pd.DataFrame({
    "BH":  bh_equity,
    "REB": reb_equity,
    "SPY": bench
}).to_json(out_dir/"backtest_rebal_equity.json", orient="split", date_format="iso")

json.dump(stats_out, open(out_dir/"backtest_rebal_stats.json","w"), indent=2)
print("✅ Rebalansz back-test elkészült • outputs/backtest_rebal_*")