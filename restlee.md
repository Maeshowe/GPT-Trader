fetch_data.py
GDP skálázása
get_macro_indicators()-ben válts:
out["GDP"] = round(float(data.iloc[-1]) / 1000, 2)  # ezres USD-mrd helyett USD-billion

