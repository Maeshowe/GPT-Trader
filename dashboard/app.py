import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ── Fájl-útvonalak -----------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent
def load(rel_path): return json.load(open(BASE / rel_path))

sector_data = load("inputs/sector_inputs.json")
firm_data   = load("inputs/firm_inputs.json")

# ── Streamlit layout ---------------------------------------------------------
st.set_page_config(page_title="GPT Portfolio Dashboard", layout="wide")
st.sidebar.header("📊 GPT Portfolio Dashboard")
st.sidebar.write(f"**Dátum:** {sector_data[0]['today']}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Sector Scores                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
st.subheader("Sector Scores")
sector_df = pd.DataFrame(
    [{"Sector": s["name"].title(), "Score": s["sector_score"]} for s in sector_data]
)
st.dataframe(sector_df, use_container_width=True, hide_index=True)

sector_chart = px.bar(sector_df, x="Sector", y="Score", title="Sector Score Comparison")
st.plotly_chart(sector_chart, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Top 20 Firm Scores                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
firm_df = pd.DataFrame(firm_data)
top20 = firm_df.sort_values("firm_score", ascending=False).head(20)
st.subheader("Top 20 Firm Scores")
st.dataframe(top20[["ticker", "sector", "firm_score"]], use_container_width=True)

firm_chart = px.bar(
    top20.sort_values("firm_score", ascending=True),
    x="firm_score", y="ticker", orientation="h", title="Top 20 Firm Scores"
)
st.plotly_chart(firm_chart, use_container_width=True)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ SHAP-szerű magyarázat (top firm)                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
top_firm = top20.iloc[0]
shap_vals = next(
    (f.get("firm_shap") for f in firm_data
     if f["ticker"] == top_firm["ticker"] and f.get("firm_shap")),
    None
)

if shap_vals:
    shap_df = pd.DataFrame(
        [{"Feature": k, "SHAP": v} for k, v in shap_vals.items()]
    ).sort_values("SHAP")
    st.subheader(f"SHAP-szerű Feature Hatások – {top_firm['ticker']}")
    shap_chart = px.bar(
        shap_df, x="SHAP", y="Feature", orientation="h",
        title=f"{top_firm['ticker']} – Feature Contributions"
    )
    st.plotly_chart(shap_chart, use_container_width=True)
else:
    st.info(f"Nincs SHAP-adata a(z) {top_firm['ticker']} számára – futtasd újra a firm-promptot.")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Portfólió – 15 Asset Allocation                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
port_path = BASE / "outputs/portfolio_latest.json"
if port_path.exists():
    portfolio = load("outputs/portfolio_latest.json")
    st.subheader("Current 15-asset Allocation")
    alloc_df = pd.DataFrame(portfolio["table"])
    st.dataframe(alloc_df, use_container_width=True)

    if not alloc_df.empty:
        alloc_chart = px.bar(
            alloc_df.sort_values("Weight (%)", ascending=True),
            x="Weight (%)", y="Asset", orientation="h",
            title="Portfolio Allocation Weights"
        )
        st.plotly_chart(alloc_chart, use_container_width=True)
else:
    st.info("Nincs még portfólió – futtasd le a generator_runner scriptet.")