#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT Portfolio Dashboard – minden modul + rugalmas Sector Score forrás
"""
import json, datetime, pandas as pd
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ────────────────────────── Path setup ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "outputs"
INP  = ROOT / "inputs"

SECTOR_SCORES = OUT / "sector_scores.json"      # új
SECTOR_INPUT  = INP / "sector_inputs.json"      # régi

FIRM_FILE   = INP / "firm_inputs.json"
PORT_FILE   = OUT / "portfolio_latest.json"
SENT_FILE   = OUT / "news_sentiment.json"
BH_EQ_FILE  = OUT / "backtest_equity.json"
BH_ST_FILE  = OUT / "backtest_stats.json"
REB_EQ_FILE = OUT / "backtest_rebal_equity.json"
REB_ST_FILE = OUT / "backtest_rebal_stats.json"
RISK_FILE   = OUT / "portfolio_risk_budget.json"

# ────────────────────────── Streamlit   ───────────────────────────────────
st.set_page_config(page_title="GPT Portfolio Dashboard", layout="wide")
st.sidebar.header("📊 GPT Portfolio Dashboard")
st.sidebar.markdown(f"**Dátum:** {datetime.date.today()}")

# ---------- helper --------------------------------------------------------
def read_json(path: Path, orient_split_ok: bool = True) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size < 3:
        return pd.DataFrame()
    try:
        return pd.read_json(path)
    except ValueError:
        if orient_split_ok:
            try:
                return pd.read_json(path, orient="split")
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

# ╔═════════  Sector Scores  ═══════════════════════════════════════════════╗
df_sector = read_json(SECTOR_SCORES)
if df_sector.empty:                         # fallback régi forrásra
    df_sector = pd.DataFrame([
        {"Sector": s["name"].title(), "Score": s.get("sector_score", 0)}
        for s in read_json(SECTOR_INPUT, orient_split_ok=False).to_dict(orient="records")
    ])

if not df_sector.empty:
    st.subheader("Sector Scores")
    st.dataframe(df_sector, use_container_width=True)
    st.plotly_chart(px.bar(df_sector, x="Sector", y="Score",
                           title="Sector Score Comparison"),
                    use_container_width=True)
else:
    st.info("Nincs sector score – futtasd a sector_runner promptot.")

# ╔═════════  Top 20 Firm Scores  ═══════════════════════════════════════════╗
df_firm = read_json(FIRM_FILE, orient_split_ok=False)
if not df_firm.empty:
    top20 = df_firm.nlargest(20, "firm_score")
    st.subheader("Top 20 Firm Scores")
    st.dataframe(top20[["ticker", "sector", "firm_score"]],
                 use_container_width=True)

    st.plotly_chart(
        px.bar(top20.sort_values("firm_score"),
               x="firm_score", y="ticker", orientation="h",
               title="Top 20 Firm Scores"),
        use_container_width=True
    )

# ╔═════════  News Sentiment  ═══════════════════════════════════════════════╗
df_sent = read_json(SENT_FILE)
if {"ticker", "avg_sent"}.issubset(df_sent.columns) and not df_sent.empty:
    st.subheader("7-day Average News Sentiment")
    st.bar_chart(df_sent.set_index("ticker")["avg_sent"],
                 height=250, use_container_width=True)
    st.caption("Cut-off < −0.05 → −30 % weight (Edge-jelzés)")
else:
    st.info("Nincs hír-szentiment – futtasd a news_sentiment.py-t.")

# ╔═════════  SHAP-szerű Feature Hatások  ══════════════════════════════════╗
if not df_firm.empty:
    top_firm = df_firm.nlargest(1, "firm_score").iloc[0]
    shap_vals = next(
        (f.get("firm_shap") for f in df_firm.to_dict("records")
         if f["ticker"] == top_firm["ticker"] and f.get("firm_shap")),
        None
    )

    if shap_vals:
        st.subheader(f"SHAP-szerű Feature Hatások – {top_firm['ticker']}")
        shap_df = (pd.DataFrame(
            [{"Feature": k, "SHAP": v} for k, v in shap_vals.items()])
            .sort_values("SHAP")
        )
        st.plotly_chart(
            px.bar(shap_df, x="SHAP", y="Feature", orientation="h",
                   title=f"{top_firm['ticker']} – Feature Contributions"),
            use_container_width=True
        )
    else:
        st.info(
            f"Nincs SHAP-adata a(z) {top_firm['ticker']} számára – "
            "futtasd újra a firm-promptot."
        )

# ╔═════════  15-Asset Allocation  ══════════════════════════════════════════╗
if PORT_FILE.exists():
    port = json.load(open(PORT_FILE))
    st.subheader("Current 15-asset Allocation")
    alloc_df = pd.DataFrame(port["table"])
    st.dataframe(alloc_df, use_container_width=True)

    if not alloc_df.empty:
        st.plotly_chart(
            px.bar(alloc_df.sort_values("Weight (%)"),
                   x="Weight (%)", y="Asset", orientation="h",
                   title="Portfolio Allocation Weights"),
            use_container_width=True
        )

# ╔═════════  Buy & Hold Back-test  ════════════════════════════════════════╗
df_bh_eq = read_json(BH_EQ_FILE)
df_bh_st = read_json(BH_ST_FILE)
if not df_bh_eq.empty and not df_bh_st.empty:
    st.header("Performance Back-test – Buy & Hold")
    fig_bh = go.Figure()
    for col in df_bh_eq.columns:
        fig_bh.add_scatter(x=df_bh_eq.index, y=df_bh_eq[col], name=col)
    fig_bh.update_layout(xaxis_title="Date", yaxis_title="Value (USD)")
    st.plotly_chart(fig_bh, use_container_width=True)
    st.table(df_bh_st.T)

# ╔═════════  Monthly Rebalance Back-test  ══════════════════════════════════╗
df_reb_eq = read_json(REB_EQ_FILE)
df_reb_st = read_json(REB_ST_FILE)
if not df_reb_eq.empty and not df_reb_st.empty:
    st.subheader("Monthly Rebalance Back-test")
    fig_reb = go.Figure()
    for col in df_reb_eq.columns:
        style = dict(dash="dot") if col.upper() == "SPY" else {}
        fig_reb.add_scatter(x=df_reb_eq.index, y=df_reb_eq[col],
                            name=col, line=style)
    fig_reb.update_layout(xaxis_title="Date", yaxis_title="Value (USD)")
    st.plotly_chart(fig_reb, use_container_width=True)
    st.table(df_reb_st.T)

# ╔═════════  Risk-Budget vs. LLM Weights  ══════════════════════════════════╗
df_rb = read_json(RISK_FILE)
if not df_rb.empty and PORT_FILE.exists():
    df_rb["ticker"] = df_rb["ticker"].str.strip()
    df_llm = (
        pd.DataFrame(port["table"])
        [["Asset", "Weight (%)"]]
        .rename(columns={"Asset": "ticker", "Weight (%)": "llm_w"})
        .assign(ticker=lambda d: d["ticker"].str.strip())
    )
    merged = df_llm.merge(df_rb, on="ticker", how="inner")
    if not merged.empty:
        st.subheader("LLM vs. Risk-Budget Weights")
        st.dataframe(merged.set_index("ticker"))
        fig_cmp = go.Figure()
        fig_cmp.add_bar(x=merged["ticker"], y=merged["llm_w"], name="LLM")
        fig_cmp.add_bar(x=merged["ticker"], y=merged["weight"], name="Risk-Budget")
        fig_cmp.update_layout(barmode="group",
                              xaxis_title="Ticker",
                              yaxis_title="Weight (%)")
        st.plotly_chart(fig_cmp, use_container_width=True)
    else:
        st.info("Ticker-mezők nem egyeznek – ellenőrizd a JSON-ok formátumát.")
else:
    st.info("Risk-budget fájl hiányzik – futtasd a risk_budget.py-t.")