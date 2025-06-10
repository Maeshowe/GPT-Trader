GPT‑Trader – Részletes fejlesztési terv

Verzió: 2025‑06‑10
Állapot: Draft / Közös szerkesztésre kész

⸻

🎯 Fő célkitűzések (2025 Q3)
	1.	Automatikus ticker‑felfedező modul a szektor‑ETF‑holdings alapján (Option 1, ajánlott).
	2.	composite_scoring.py funkcionális bővítése (technikai‑, fundamentális‑, és LLM‑score kompozit).
	3.	Konfigurációs séma kibővítése (YAML), manuális override támogatása.
	4.	Black‑Litterman & SHAP integráció produkciós szintre emelése.
	5.	CI + Tesztlefedettség ≥ 80 % (PyTest, GitHub Actions).

⸻

✅ Elkészítendő feladatok (checkboxokkal)

1. composite_scoring.py – funkcionális kiegészítések
	•	Refaktor: moduláris ScoreComponent osztály (fundamental, technical, sentiment, ESG).
	•	Súlykezelés: dinamikus súlyok YAML‑ből, fallback default értékekkel.
	•	Technikai mutatók: RSI 14d, 50/200 DMA trend, ATR‑volatilitás.
	•	Dokumentáció & példafüzet (notebooks/composite_scoring_demo.ipynb).
	•	Unit tesztek (tests/test_composite_scoring.py).

2. automatic_ticker_discovery.py

Prioritás: P0 – ez nyitja a teljes univerzumot.

2.1 Option 1 – ETF holdings (ajánlott)
	•	Függőségek: yfinance + requests → adatokhoz fallback API/web‑scraping.
	•	Fő függvény get_etf_holdings(etf_symbol: str, top_n=10) -> list[str].
	•	Kivételkezelés: hálózati hibára retry + log warning.
	•	Cache: 1 napos lokális JSON (cache/etf_holdings/{symbol}.json).
	•	CLI parancs python -m tools.auto_ticker --update (argumentumok: --sector, --top-n).

2.2 Option 2 – Market Cap rangsor
	•	API integráció: FinancialModelingPrep (/stock-screener) vagy Alpha Vantage.
	•	get_sector_top_stocks(sector: str, top_n=5).
	•	Input univerzum szűkítés: S&P 500 vagy Russell 1000.

2.3 Option 3 – Hybrid mód
	•	YAML kulcsok: auto_discovery, auto_count, manual_additions, manual_exclusions.
	•	Merge‑logika: (ETF ∪ MarketCap ∪ ManualAdd) \ ManualEx.

2.4 Option 4 – Több forrás (Union / Intersection)
	•	Keretrendszer a jövőbeni források (e.g. MSCI ACWI) bevonására.

3. Konfiguráció & séma (config/sectors.yaml)
	•	Új mezők definiálása (lásd Option 3 példa).
	•	JSON‑schema generálás (schemas/sectors.schema.json).
	•	Validátor: pre‑commit hook, ami YAML‑t JSON‑schema ellenőriz.

4. Black‑Litterman & Risk‑Budget modul
	•	risk_budget.py → parametrikus „views” YAML (példa: views/2025‑Q3.yaml).
	•	Stress‑teszt: Monte Carlo 10 000 szimuláció; SHAP a risk hozzájárulásra.
	•	Notebook: notebooks/bl_demo.ipynb – Exhibit 4 reprodukció a Whitepaperből.

5. Explainability & SHAP
	•	Integráció a composite_scoring outputra (feature importance).
	•	Streamlit widget: SHAP beeswarm / heatmap, asset‑ és factor‑szinten.
	•	Mentés: outputs/shap_values/{run_id}.parquet.

6. CI / CD
	•	GitHub Actions: python -m pytest, coverage badge, flake8.
	•	Schedule‑trigger (cron) – hetente egyszer friss ETF holdings & universe.

7. Dokumentáció & README
	•	Frissített architektúra‑diagram (draw.io → PNG/SVG).
	•	„Getting Started” szekció ETF‑ticker felfedezéshez.
	•	Whitepaper ↔ Kód mátrix (funkció × fájl).

⸻

📅 Sprint‑javaslat (kéthetes)

Sprint	Fókusz	Kimenet
S‑1	Option 1 ETF holdings, YAML séma, unit tesztek	auto_discovery modul + 80 % branch coverage
S‑2	Composite scoring bővítés, SHAP prototípus	új score pipeline + SHAP demo notebook
S‑3	BL optimizer & Risk‑budget, Streamlit SHAP widget	kockázati dashboard v2
S‑4	Hybride / több univerzum opciók, CI‑finomítás	Option 3 & 4, teljes CI badge > 80 %


⸻

🗂️ Fájl‑ és mappa‑struktúra változások

project_root/
├── auto_discovery/
│   ├── __init__.py
│   └── automatic_ticker_discovery.py
├── config/
│   └── sectors.yaml
├── schemas/
│   └── sectors.schema.json
├── tests/
│   ├── test_composite_scoring.py
│   └── test_auto_discovery.py
└── notebooks/
    ├── composite_scoring_demo.ipynb
    └── bl_demo.ipynb


⸻

ℹ️ Megjegyzések / Függőségek
	•	API‑kulcsok (Alpha Vantage, FMP) – .env / GitHub Secrets.
	•	Licence – ETF holdings scraping ellenőrizendő (legal notice).
	•	Open source – későbbi cél: PyPI csomag (gpt_trader).

⸻

Következő lépés: jelöld meg kommentben, ha módosítanál, vagy húzz fel branch‑et az S‑1 sprinthez.