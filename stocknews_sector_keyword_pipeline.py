import requests
from collections import Counter
import re

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
    text = re.sub(r"[^a-zA-Z0-9\\s]", "", text)
    return text.lower()

def extract_keywords(text, stopwords=None, top_n=1):
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
    results = {}
    for sector, etf in SECTOR_ETFS.items():
        print(f"Fetching news for {sector} ({etf})...")
        url = f"https://stocknewsapi.com/api/v1?tickers={etf}&items=50&token={API_KEY}"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            results[sector] = "API ERROR"
            continue
        articles = resp.json().get("data", [])
        combined_text = " ".join([
            a.get("title", "") + " " + a.get("text", "")
            for a in articles
        ])
        keywords = extract_keywords(combined_text)
        results[sector] = keywords[0][0] if keywords else "N/A"
    return results

if __name__ == "__main__":
    keywords_by_sector = get_sector_keywords()
    print("\n📊 Top Keywords by Sector:")
    for sector, keyword in keywords_by_sector.items():
        print(f"{sector}: {keyword}")