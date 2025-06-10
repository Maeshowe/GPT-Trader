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

python data_fetch/fetch_data.py
python run_prompts.py
python news_sentiment.py
python backtest.py
python backtest_rebal.py
python risk_budget.py

streamlit run dashboard/app.py

git commit -am "chore(docs): trigger CI" 


# CI lefutott és pusholt változásokat
git pull origin main  # Lehúzod a CI változásait

# Most folytathatod a munkát
vim news_sentiment.py
git add .
git commit -m "feat: add new feature"
git push origin main


# Komplett directory structure VENV nélkül
tree -a -I 'venv|.git|__pycache__|outputs|*.pyc|node_modules|.env'

# Vagy ha túl mély, korlátozd a szinteket
tree -a -L 4 -I 'venv|.git|__pycache__|outputs|*.pyc|node_modules|.env'

# Save to file
tree -a -I 'venv|.git|__pycache__|outputs|*.pyc|node_modules|.env' > project_structure.txt

# 1. Directory structure
tree -a -I 'venv|.git|__pycache__|outputs|*.pyc|node_modules|.env' > project_structure.txt

# 2. All Python files
find . -name "*.py" -not -path "./venv/*" -not -path "./.git/*" -not -path "./__pycache__/*" \
  -exec echo -e "\n\n=== FILE: {} ===" \; -exec cat {} \; > all_python_files.txt

# 3. Config files  
find . \( -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o -name "*.md" \) \
  -not -path "./venv/*" -not -path "./.git/*" -not -path "./outputs/*" \
  -exec echo -e "\n\n=== CONFIG: {} ===" \; -exec cat {} \; > all_config_files.txt

# 4. Check results
echo "Generated files:"
ls -la *.txt
wc -l *.txt

# Get the .j2 template files too
find . -name "*.j2" -not -path "./venv/*" \
  -exec echo -e "\n\n=== TEMPLATE: {} ===" \; -exec cat {} \; > prompt_templates.txt

# Check what .j2 files we have
find . -name "*.j2" -not -path "./venv/*"




composite_scoring <-- nem teljes funkcionalitás -->
Next step: Implementáljuk az ETF holdings automatikus lekérést?
