#!/usr/bin/env python3
"""
StockNews: 7-napos átlagolt hír-szentiment
Positive = +1 · Neutral = 0 · Negative = –1
Cut-off: avg < –0.05 → súly –30 %
"""
import os, json, requests, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv() 

BASE  = Path(__file__).resolve().parent
PORT  = json.load(open(BASE/"outputs/portfolio_latest.json"))["table"]
API   = os.getenv("STOCKNEWS_API_KEY")

LABEL2SCORE = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}

# news_sentiment.py – csak a URL és a DEBUG módosul
def avg_sentiment(tkr):
    url = (
        "https://stocknewsapi.com/api/v1"
        f"?tickers={tkr}&items=100&date=last7days&token={API}"
    )
    js = requests.get(url, timeout=30).json()
    if not js.get("data"):        # DEBUG
        print("⚠️", tkr, "→", js.get("message") or "0 articles")
        return None
    scores = [
        {"Positive": 1, "Neutral": 0, "Negative": -1}.get(a["sentiment"], 0)
        for a in js["data"] if a.get("sentiment")
    ]
    return sum(scores)/len(scores) if scores else None

out = []
for row in PORT:
    tkr = row["Asset"].strip().upper()
    s   = avg_sentiment(tkr)
    if s is not None:
        out.append({"ticker": tkr, "avg_sent": round(s, 3)})
    time.sleep(0.2)         # 5 req/sec limit

Path(BASE/"outputs").mkdir(exist_ok=True)
json.dump(out, open(BASE/"outputs/news_sentiment.json","w"), indent=2)
print(f"✅ {len(out)} sentiment-sor mentve • outputs/news_sentiment.json")