#!/usr/bin/env python3
"""
Firm prompt futtatása:
• GPT-score kinyerése
• SHAP-szerű feature-hatások kiszámítása (fix lineáris súlyokkal)
• Eredmény visszaírása inputs/firm_inputs.json-be
"""

import os, json, re
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

# ── Beállítások --------------------------------------------------------------
BASE = Path(__file__).resolve().parent
load_dotenv(override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── 1) Bemenet ---------------------------------------------------------------
firm_path = BASE / "inputs/firm_inputs.json"
firm_records = json.load(open(firm_path))

# Itt példaként csak a legfrissebb 1. rekordot dolgozzuk fel; végig is iterálhatnál.
rec = firm_records[0]

# ── 2) Prompt renderelés -----------------------------------------------------
prompt = Template(open(BASE / "prompts/firm_prompt.j2").read()).render(**rec)

resp = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=700
).choices[0].message.content.strip()

print("🧾 GPT-válasz (Firm):\n", resp)

# ── 3) Score kinyerése -------------------------------------------------------
m = re.search(r"Score:\s*(\d+)", resp)
rec["firm_score"] = int(m.group(1)) if m else None

# ── 4) SHAP-szerű magyarázat (fix súlyok) ------------------------------------
feature_weights = {
    "P/E": 0.20,
    "PEG": -0.10,
    "Beta": -0.10,
    "ROE": 0.40,
    "Quick Ratio": 0.30
}
fin = rec["firm_financials_json"]
rec["firm_shap"] = {
    k: round(feature_weights[k] * fin.get(k, 0), 2) for k in feature_weights
}

# ── 5) Visszaírás a JSON-listába -------------------------------------------
firm_records[0] = rec
json.dump(firm_records, open(firm_path, "w"), indent=2)
print(f"✅ firm_score ({rec['firm_score']}), SHAP-értékek mentve a {firm_path} fájlba")