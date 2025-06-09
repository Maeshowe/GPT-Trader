fetch_data.py
GDP skálázása
get_macro_indicators()-ben válts:
out["GDP"] = round(float(data.iloc[-1]) / 1000, 2)  # ezres USD-mrd helyett USD-billion

# 0) .env + config.yaml beállítva

# 1) Adatok és JSON inputok
python data_fetch/fetch_data.py

# 2) GPT Score-ok (minden szektor és cég)
python run_prompts.py     # ~ 1 GPT-hívás / rekord

# 3) Portfólió generálás
python generator_runner.py

# 4) Dashboard megtekintése
streamlit run dashboard/app.py

https://github.com/Maeshowe/GPT-Trader.git
git@github.com:Maeshowe/GPT-Trader.git

echo "# GPT-Trader" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:Maeshowe/GPT-Trader.git
git push -u origin main

git remote add origin git@github.com:Maeshowe/GPT-Trader.git
git branch -M main
git push -u origin main

echo "# GPT-Trader" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Maeshowe/GPT-Trader.git
git push -u origin main

git remote add origin https://github.com/Maeshowe/GPT-Trader.git
git branch -M main
git push -u origin main

