import yaml, json, types
from pathlib import Path
import pytest

# Monkeypatch-elünk, hogy ne hívjunk külső API-kat
from gpt_trader.data_fetch import config_autofill as mod

SAMPLE_CFG = """
sectors:
  - name: Energy
    etf: XLE
    keyword: dummy
    override_tickers: []
"""

def fake_discover(etf, top_n):
    return ["AAA", "BBB", "CCC"]  # determinisztikus

def fake_articles(ticker, items=50):
    return [{"title": "Oil prices surge higher", "text": "Energy market rally"}]

def test_main(tmp_path, monkeypatch):
    # Ideiglenes config.yaml
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(SAMPLE_CFG)
    monkeypatch.setattr(mod, "CFG", cfg_file)

    # Mock API-k
    monkeypatch.setattr(mod, "discover_tickers", fake_discover)
    monkeypatch.setattr(mod, "stocknews_articles", fake_articles)

    # Futtatás
    mod.main(top_n=3)

    new_cfg = yaml.safe_load(cfg_file.read_text())
    sec = new_cfg["sectors"][0]
    assert sec["override_tickers"] == ["AAA", "BBB", "CCC"]
    assert sec["keyword"] == "Energy"  # extract_keyword → 'Energy'