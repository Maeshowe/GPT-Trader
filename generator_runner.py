#!/usr/bin/env python3
"""
A legmagasabb firm_score-ú 15 cégre épít portfóliót
és menti outputs/portfolio_latest.json-ba.
"""

import os, json, re
from pathlib import Path
from io import StringIO

import yaml                    # YAML beolvasáshoz
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

BASE = Path(__file__).resolve().parent
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── 1) Top 15 firm -----------------------------------------------------------
firm_records = json.load(open(BASE / "inputs/firm_inputs.json"))
top_firms = sorted(
    firm_records,
    key=lambda x: x["firm_score"] or 0,
    reverse=True
)[:15]

# ── 2) ETF-univerzum a config.yaml-ból --------------------------------------
cfg_sectors = yaml.safe_load(open(BASE / "config.yaml"))["sectors"]
etf_list = [s["etf"] for s in cfg_sectors if "etf" in s]

# ── 3) Generator prompt input ------------------------------------------------
generator_input = {
    "top_firms_list": [
        {"name": f["ticker"], "score": f["firm_score"], "thesis": "Top-ranked firm"}
        for f in top_firms
    ],
    "macro_forecast_table": {"Note": "Auto-generated run"},
    "etf_universe_list": etf_list,
    "today": top_firms[0]["today"]
}

# ── 4) Prompt futtatás -------------------------------------------------------
prompt_tpl = Template(open(BASE / "prompts/generator_prompt.j2").read())
prompt = prompt_tpl.render(**generator_input)

md_table = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=900
).choices[0].message.content.strip()

# ── 5) Markdown → DataFrame --------------------------------------------------
clean_lines = [
    ln for ln in md_table.splitlines()
    if not re.match(r"^\s*\|[-:]+\|", ln) and "|" in ln
]
clean_md = "\n".join(clean_lines)
df = pd.read_csv(
    StringIO(clean_md), sep="\\|", engine="python"
).dropna(axis=1, how="all").iloc[1:]
df.columns = [c.strip() for c in df.columns]

# ── 6) JSON mentés -----------------------------------------------------------
portfolio_json = {
    "date": generator_input["today"],
    "table": df.to_dict(orient="records")
}

out_dir = BASE / "outputs"
out_dir.mkdir(exist_ok=True)
out_file = out_dir / "portfolio_latest.json"
json.dump(portfolio_json, open(out_file, "w"), indent=2)

print(f"✅ Portfólió mentve: {out_file}")