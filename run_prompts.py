#!/usr/bin/env python3
"""
Aszinkron batch futtató:
• Sector-score  (3 rekord)    – async, timeout+retry
• Firm-score + SHAP (9 rekord)– async, timeout+retry, max 5 párhuzamos
"""

import os, json, re, asyncio, time
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from openai import AsyncOpenAI, APITimeoutError, RateLimitError

BASE = Path(__file__).resolve().parent
load_dotenv(override=True)
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL  = os.getenv("OPENAI_MODEL","gpt-4o")

MAX_CONCURRENCY = 2
REQ_TIMEOUT     = 60        # mp
RETRY_LIMIT     = 4
RETRY_BACKOFF   = 8         # mp

# ── GPT hívás retry-val, timeout-tal ----------------------------------------
async def gpt_call(prompt, temperature=0):
    for attempt in range(1, RETRY_LIMIT+1):
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=700,
                timeout=REQ_TIMEOUT            # 1.3.x paraméter
            )
            return resp.choices[0].message.content
        except (APITimeoutError, RateLimitError) as e:
            wait = RETRY_BACKOFF * attempt
            print(f"⚠️  Retry {attempt}/{RETRY_LIMIT} in {wait}s – {e}")
            await asyncio.sleep(wait)
    raise RuntimeError("GPT call failed after retries")

# ════════════════════════════════════════════════════════════════════════════
# 1) Sector batch (async)                                                     -
# ════════════════════════════════════════════════════════════════════════════
async def run_sectors_async():
    path = BASE/"inputs/sector_inputs.json"
    sectors = json.load(open(path))
    tpl = Template(open(BASE/"prompts/sector_prompt.j2").read())

    async def job(s):
        print(f"→ Sector: {s['name']}")
        out = await gpt_call(tpl.render(**s))
        m = re.search(r"Score:\s*(\d+)", out)
        s["sector_score"] = int(m.group(1)) if m else None
        print(f"✓ {s['name']} score = {s['sector_score']}")

    await asyncio.gather(*[job(s) for s in sectors])
    json.dump(sectors, open(path,"w"), indent=2)
    print("✅ Sector-scores frissítve.")

# ════════════════════════════════════════════════════════════════════════════
# 2) Firm batch (async)                                                       -
# ════════════════════════════════════════════════════════════════════════════
FIRM_W = {"P/E":0.2,"PEG":-0.1,"Beta":-0.1,"ROE":0.4,"Quick Ratio":0.3}
firm_tpl = Template(open(BASE/"prompts/firm_prompt.j2").read())

async def run_firms_async():
    path = BASE/"inputs/firm_inputs.json"
    firms = json.load(open(path))
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def job(f):
        if f.get("firm_score") and f.get("firm_shap"):
            print(f"⌛ Skip {f['ticker']} – already done")
            return
        async with sem:
            print(f"→ GPT {f['ticker']}")
            out = await gpt_call(firm_tpl.render(**f))
            m = re.search(r"Score:\s*(\d+)", out)
            f["firm_score"] = int(m.group(1)) if m else None
            fin = f["firm_financials_json"]
            f["firm_shap"] = {k: round(FIRM_W[k]*fin.get(k,0),2) for k in FIRM_W}
            print(f"✓ {f['ticker']} score = {f['firm_score']}")

    await asyncio.gather(*[job(f) for f in firms])
    json.dump(firms, open(path,"w"), indent=2)
    print("✅ Firm-scores + SHAP frissítve.")

# ════════════════════════════════════════════════════════════════════════════
# Main async függvény - EZ A KULCS!
# ════════════════════════════════════════════════════════════════════════════
async def main():
    """Főprogram - egyetlen event loop-ban futtatja mindkét batch-et"""
    t0 = time.time()
    
    # Szekvenciálisan futtatjuk őket ugyanabban az event loop-ban
    await run_sectors_async()
    await run_firms_async()
    
    print(f"⏱️  Összidő: {round(time.time()-t0,1)} mp")

# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Csak egyetlen asyncio.run() hívás!
    asyncio.run(main())