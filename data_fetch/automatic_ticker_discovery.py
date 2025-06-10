"""
ETF-holdings alapú ticker-gyűjtés.

✓  Dependenciák: yfinance, requests, pandas
✓  Használat:
      from data_fetch.automatic_ticker_discovery import discover_tickers

      tickers = discover_tickers("XLE", top_n=10)
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import requests
import yfinance as yf

__all__ = ["discover_tickers"]

log = logging.getLogger(__name__)


def _fallback_yfinance(etf_symbol: str, top_n: int) -> List[str]:
    """B-terv: yfinance institutional_holders táblából szedjük ki a top_n tickert."""
    etf = yf.Ticker(etf_symbol)
    try:
        df: pd.DataFrame = etf.institutional_holders  # type: ignore
        if df is None or "Holder" not in df.columns:
            return []
        return df["Holder"].head(top_n).tolist()
    except Exception as exc:  # pragma: no cover
        log.warning("yfinance fallback sikertelen: %s", exc)
        return []


def _etfdb_api(etf_symbol: str, top_n: int) -> List[str]:
    """
    Próbálkozunk az ETFdb (ingyenes) JSON-feeddel.
    DOC: https://etfdb.com/api/docs/ (nincs kulcs, de korlátozott sebesség).
    """
    url = f"https://etfdb.com/api/v3/etfs/{etf_symbol}/holdings/"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    holdings = data.get("holdings", [])[:top_n]
    return [h["ticker"] for h in holdings if h.get("ticker")]
    

def discover_tickers(etf_symbol: str, top_n: int = 10) -> List[str]:
    """
    Visszaadja az ETF top_n holding tickerjeit.
    Folyamat:
        1. ETFdb API → ha 200 OK és van ticker lista → return
        2. Fallback: yfinance institutional_holders
    """
    try:
        tickers = _etfdb_api(etf_symbol, top_n)
        if tickers:
            return tickers
    except Exception as exc:  # pragma: no cover
        log.info("ETFdb API nem elérhető (%s), fallback yfinance.", exc)

    return _fallback_yfinance(etf_symbol, top_n)