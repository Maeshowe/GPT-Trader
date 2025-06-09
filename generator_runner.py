#!/usr/bin/env python3
"""
Portfólió-generátor
• Beolvassa az aktuális firm-score listát (inputs/firm_inputs.json)
• Top 15 alapján promptot futtat a Generator LLM-mel
• Súlyokat hír-szentimenttel korrigálja (StockNews 7-napos átlag)
• Kimenet: outputs/portfolio_latest.json
"""

import os, json, re
from pathlib import Path
from io import StringIO

import yaml
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

# ── Konstansok / fájl-útvonalak ─────────────────────────────────────────────
BASE  = Path(__file__).resolve().parent
INPUT = BASE / "inputs"
OUT   = BASE / "outputs"
PROMPT_DIR = BASE / "prompts"

load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

# ────────────────────────────────────────────────────────────────────────────
# 1. Top 15 firm kiválasztása
# ────────────────────────────────────────────────────────────────────────────
firm_records = json.load(open(INPUT / "firm_inputs.json"))
top_firms = sorted(
    firm_records,
    key=lambda x: x["firm_score"] or 0,
    reverse=True
)[:15]

# ────────────────────────────────────────────────────────────────────────────
# 2. ETF-univerzum (config.yaml)
# ────────────────────────────────────────────────────────────────────────────
cfg_sectors = yaml.safe_load(open(BASE / "config.yaml"))["sectors"]
etf_list = [s["etf"] for s in cfg_sectors if "etf" in s]

# ────────────────────────────────────────────────────────────────────────────
# 3. Hír-szentiment beolvasása és súlykorrekció
#    −30 % vágás, ha 7-napos átlag < –0.05
# ────────────────────────────────────────────────────────────────────────────
sentiment_path = OUT / "news_sentiment.json"
sent_map = {}
if sentiment_path.exists():
    sent_map = {d["ticker"].upper(): d["avg_sent"] for d in json.load(open(sentiment_path))}

NEG_TH = -0.05     # küszöb
CUT    = 0.30      # 30 % súlycsökkentés

for row in top_firms:
    tkr = row["ticker"].upper()
    s   = sent_map.get(tkr)
    if s is not None and s < NEG_TH:
        # eredeti súly a Generator promptban szereplő "Weight (%)" kulcsszóval
        row["Weight (%)"] = round(row.get("Weight (%)", 10) * (1 - CUT), 2)
        # “Edge” oszlop kiegészítése jelzéssel
        row["Edge"] = row.get("Edge", "") + f" | SENTIMENT↓{s}"

# ────────────────────────────────────────────────────────────────────────────
# 4. Prompt-input összeállítása
# ────────────────────────────────────────────────────────────────────────────
generator_input = {
    "top_firms_list": [
        {
            "name":  f["ticker"],
            "score": f["firm_score"],
            "thesis": f.get("Edge", "Top-ranked firm"),
            "weight": f.get("Weight (%)", 10)
        }
        for f in top_firms
    ],
    "macro_forecast_table": {"Note": "Auto-generated run"},
    "etf_universe_list": etf_list,
    "today": top_firms[0]["today"]
}

# ────────────────────────────────────────────────────────────────────────────
# 5. Prompt futtatás OpenAI-val
# ────────────────────────────────────────────────────────────────────────────
prompt_tpl = Template(open(PROMPT_DIR / "generator_prompt.j2").read())
prompt = prompt_tpl.render(**generator_input)

response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=900
).choices[0].message.content.strip()

# ────────────────────────────────────────────────────────────────────────────
# 6. Markdown→DataFrame konvertálás
# ────────────────────────────────────────────────────────────────────────────
clean_lines = [
    ln for ln in response.splitlines()
    if "|" in ln and not re.match(r"^\s*\|[-:]+\|", ln)
]
md = "\n".join(clean_lines)
df = pd.read_csv(StringIO(md), sep="\\|", engine="python").dropna(axis=1, how="all").iloc[1:]
df.columns = [c.strip() for c in df.columns]

# ────────────────────────────────────────────────────────────────────────────
# 7. Kimenet JSON-fájlba
# ────────────────────────────────────────────────────────────────────────────
portfolio_json = {
    "date": generator_input["today"],
    "table": df.to_dict(orient="records")
}

OUT.mkdir(exist_ok=True)
json.dump(portfolio_json, open(OUT / "portfolio_latest.json", "w"), indent=2)

print("✅ Portfólió mentve: outputs/portfolio_latest.json")