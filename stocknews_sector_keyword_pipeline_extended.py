
import requests
from collections import Counter
import re
import csv

API_KEY = "ygjgfact2goapnz5utf7olo7xekl6sj5dl1x38ny"

SECTOR_ETFS = {
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Healthcare": "XLV",
    "Financials": "XLF",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Information Technology": "XLK",
    "Communication Services": "XLC",
    "Real Estate": "XLRE"
}

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text.lower()

def extract_keywords(text, stopwords=None, top_n=3):
    if stopwords is None:
        stopwords = set([
            "the", "and", "for", "with", "that", "from", "this", "are", "will", "into", "more", "than",
            "has", "over", "amid", "as", "by", "about", "new", "its", "in", "on", "of", "to", "at", "a", "is"
        ])
    words = clean_text(text).split()
    filtered = [w for w in words if w not in stopwords and len(w) > 3]
    counter = Counter(filtered)
    return counter.most_common(top_n)

def get_sector_keywords():
    headers = {"Accept": "application/json"}
    results = []

    for sector, etf in SECTOR_ETFS.items():
        print(f"Fetching news for {sector} ({etf})...")
        url = f"https://stocknewsapi.com/api/v1?tickers={etf}&items=50&token={API_KEY}"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            results.append({
                "Sector": sector,
                "Keyword 1": "API ERROR",
                "Keyword 2": "",
                "Keyword 3": "",
                "Avg Sentiment": ""
            })
            continue

        articles = resp.json().get("data", [])
        combined_text = " ".join([
            a.get("title", "") + " " + a.get("text", "")
            for a in articles
        ])
        sentiments = [float(a.get("sentiment_score", 0.0)) for a in articles if "sentiment_score" in a]

        keywords = extract_keywords(combined_text)
        avg_sentiment = round(sum(sentiments)/len(sentiments), 3) if sentiments else "N/A"

        results.append({
            "Sector": sector,
            "Keyword 1": keywords[0][0] if len(keywords) > 0 else "",
            "Keyword 2": keywords[1][0] if len(keywords) > 1 else "",
            "Keyword 3": keywords[2][0] if len(keywords) > 2 else "",
            "Avg Sentiment": avg_sentiment
        })

    return results

def export_to_csv(results, filename="sector_keywords.csv"):
    with open(filename, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Sector", "Keyword 1", "Keyword 2", "Keyword 3", "Avg Sentiment"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"✅ Results exported to {filename}")

if __name__ == "__main__":
    results = get_sector_keywords()
    print("\n📊 Top 3 Keywords per Sector:")
    for r in results:
        print(f"{r['Sector']}: {r['Keyword 1']}, {r['Keyword 2']}, {r['Keyword 3']} (Sentiment: {r['Avg Sentiment']})")
    export_to_csv(results)
