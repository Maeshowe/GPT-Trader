#!/usr/bin/env python3
"""
Sector prompt futtatása, Score kinyerése és visszaírása inputs/sector_input.json-be
"""

import os, json, re
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(override=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model  = os.getenv("OPENAI_MODEL", "gpt-4o")

# ── 1) Input JSON betöltés
sector_path = BASE_DIR / "inputs/sector_input.json"
with open(sector_path) as f:
    data = json.load(f)

# ── 2) Prompt renderelés
with open(BASE_DIR / "prompts/sector_prompt.j2") as f:
    prompt = Template(f.read()).render(**data)

# ── 3) GPT-hívás
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,
    max_tokens=600,
)
output = response.choices[0].message.content.strip()
print("🧾 GPT-válasz (Sector):\n", output)

# ── 4) Score kinyerése regex-szel
m = re.search(r"Score:\s*(\d+)", output)
if m:
    data["sector_score"] = int(m.group(1))
    with open(sector_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ sector_score ({data['sector_score']}) mentve a {sector_path} fájlba")
else:
    print("⚠️  Nem találtam Score-t a válaszban.")