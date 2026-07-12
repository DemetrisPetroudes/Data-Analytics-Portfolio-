# S&P 500 Portfolio Dashboard  (weight-based model)

import json, os, warnings, webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from threading import Timer

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from arch import arch_model
from dash import Input, Output, State, dcc, html, dash_table, no_update
from scipy.optimize import minimize

import time, threading
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# ── Data-source mode ───────────────────────────────────────────────────────────
# "live"  (default, local):  fetch from Yahoo per request, original behaviour.
# "cached" (set in Docker):  serve from a server-side cache refreshed on a schedule,
#                            so cloud page loads never wait on or get blocked by Yahoo.
USE_CACHE     = os.environ.get("DATA_MODE", "live").strip().lower() == "cached"
if USE_CACHE:
    # A blocked refresh is expected and handled; keep the logs readable.
    import logging
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
CACHE_DIR     = os.environ.get("CACHE_DIR", "cache")   # runtime cache (ephemeral on HF)
SNAPSHOT_DIR  = "data"                                  # committed cold-start snapshot
HISTORY_START = "2007-01-01"                            # long lookback for scenarios
_ET           = ZoneInfo("America/New_York")

PORTFOLIO_FILE = os.environ.get("PORTFOLIO_FILE", "portfolio.json")
BENCHMARK      = "^SP500TR"       # S&P 500 Total Return: dividend-inclusive, so the
                                  # benchmark matches the dividend-adjusted holdings
BENCH_LABEL    = "S&P 500 TR"
RF_TICKER      = "^IRX"
DEFAULT_VALUE  = 100000.0
DATA_PERIOD    = "3y"             # estimation window: 36 months of daily data
T_DOF          = 5                # Student-t degrees of freedom for MC shocks
CLR_GREEN, CLR_RED, CLR_YELLOW = "#2DD4A4", "#FB5E6E", "#F5B13D"
CLR_BLUE,  CLR_BG,  CLR_CARD   = "#5B8DEF", "#0E1117", "#161B24"
CLR_TEXT, CLR_BORDER, CLR_GRID = "#E6E9EF", "#252B38", "#232936"

RISK_MODES = {
    "Ultra-Defensive":  dict(beta=0.40, sharpe=0.60, max_dd=0.08, vol=0.06, calmar=0.50, var95=0.010),
    "Defensive":        dict(beta=0.60, sharpe=0.80, max_dd=0.12, vol=0.09, calmar=0.65, var95=0.015),
    "Moderate":         dict(beta=0.90, sharpe=1.20, max_dd=0.15, vol=0.12, calmar=0.80, var95=0.020),
    "Balanced Growth":  dict(beta=1.10, sharpe=1.40, max_dd=0.20, vol=0.16, calmar=0.90, var95=0.025),
    "Aggressive":       dict(beta=1.40, sharpe=1.60, max_dd=0.28, vol=0.22, calmar=1.00, var95=0.035),
    "Ultra-Aggressive": dict(beta=1.80, sharpe=1.80, max_dd=0.40, vol=0.35, calmar=1.10, var95=0.050),
}

SCENARIOS = {
    "gfc":       ("2008-09-01", "2009-03-31", "2008 - Global Financial Crisis"),
    "flash2010": ("2010-04-23", "2010-07-02", "2010 - Flash Crash"),
    "edc":       ("2011-07-01", "2011-10-03", "2011 - European Debt Crisis"),
    "china1516": ("2015-06-01", "2016-02-29", "2015-16 - China Slowdown"),
    "q42018":    ("2018-10-01", "2018-12-31", "2018 - Q4 Rate-Hike Selloff"),
    "covid":     ("2020-02-19", "2020-03-23", "2020 - COVID-19 Crash"),
    "bear2022":  ("2022-01-03", "2022-10-12", "2022 - Rate-Hike Bear Market"),
}

SP500_TOP200 = list(dict.fromkeys([
    "AAPL","MSFT","NVDA","AMZN","GOOGL","GOOG","META","TSLA","AVGO","ORCL",
    "BRK-B","JPM","V","MA","BAC","GS","MS","WFC","BLK","SCHW","AXP","SPGI","COF","USB","PGR",
    "LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","DHR","ISRG","BSX","SYK","VRTX","REGN","BMY",
    "GILD","CI","ELV","HCA","COST","WMT","PG","KO","PEP","MCD","NKE","SBUX","TGT","HD","LOW",
    "TJX","MO","PM","CL","EL","KHC","GIS","HSY","GE","CAT","HON","UNP","RTX","LMT","BA","DE",
    "GEV","ETN","EMR","ITW","MMM","FDX","UPS","CSX","NSC","WM","ROK","PH",
    "XOM","CVX","COP","SLB","EOG","MPC","PSX","VLO","OXY","HAL",
    "AMD","INTC","QCOM","TXN","AMAT","LRCX","KLAC","MU","MRVL","ADI","CDNS","SNPS","ANSS",
    "FTNT","PANW","CRWD","ZS","ANET","CSCO","IBM","HPQ","DELL","STX","WDC","KEYS",
    "CRM","NOW","INTU","ADBE","ADSK","WDAY","VEEV","DDOG","SNOW","PLTR","APP","TEAM",
    "HUBS","ZM","OKTA","SPLK","MCHP","MPWR","SWKS","QRVO","ON","ENPH",
    "AMT","PLD","EQIX","CCI","PSA","O","SPG","WELL","DLR","EXR","AVB","EQR","VTR","NXT",
    "NEE","DUK","SO","AEP","EXC","XEL","PCG","SRE","ED","ETR",
    "LIN","APD","SHW","ECL","NEM","FCX","NUE","VMC","MLM","DOW","DD","PPG","IFF",
    "ZBH","BDX","BAX","CAH","MCK","ABC","CVS","HUM","MOH","CNC",
    "NFLX","DIS","CMCSA","T","VZ","TMUS","EA","TTWO","WBD","FOXA","FOX",
    "BKNG","ABNB","MAR","HLT","RCL","CCL","GM","F","APTV","RL","TPR","VFC",
    "CME","ICE","MCO","MSCI","FIS","FISV","PYPL","SQ","COIN",
]))

NAME_MAP = {
    "AAPL":"Apple Inc.","MSFT":"Microsoft Corp.","NVDA":"NVIDIA Corp.",
    "AMZN":"Amazon.com Inc.","GOOGL":"Alphabet (Class A)","GOOG":"Alphabet (Class C)",
    "META":"Meta Platforms","TSLA":"Tesla Inc.","AVGO":"Broadcom Inc.","ORCL":"Oracle Corp.",
    "BRK-B":"Berkshire Hathaway B","JPM":"JPMorgan Chase","V":"Visa Inc.","MA":"Mastercard",
    "BAC":"Bank of America","GS":"Goldman Sachs","MS":"Morgan Stanley","WFC":"Wells Fargo",
    "BLK":"BlackRock","SCHW":"Charles Schwab","AXP":"American Express","SPGI":"S&P Global",
    "COF":"Capital One","USB":"U.S. Bancorp","PGR":"Progressive Corp.",
    "LLY":"Eli Lilly","UNH":"UnitedHealth Group","JNJ":"Johnson & Johnson",
    "ABBV":"AbbVie Inc.","MRK":"Merck & Co.","TMO":"Thermo Fisher Scientific",
    "ABT":"Abbott Laboratories","DHR":"Danaher Corp.","ISRG":"Intuitive Surgical",
    "BSX":"Boston Scientific","SYK":"Stryker Corp.","VRTX":"Vertex Pharmaceuticals",
    "REGN":"Regeneron","BMY":"Bristol-Myers Squibb","GILD":"Gilead Sciences",
    "CI":"Cigna Group","ELV":"Elevance Health","HCA":"HCA Healthcare",
    "COST":"Costco Wholesale","WMT":"Walmart Inc.","PG":"Procter & Gamble",
    "KO":"Coca-Cola Co.","PEP":"PepsiCo Inc.","MCD":"McDonald\'s Corp.",
    "NKE":"Nike Inc.","SBUX":"Starbucks Corp.","TGT":"Target Corp.",
    "HD":"Home Depot","LOW":"Lowe\'s Companies","TJX":"TJX Companies",
    "MO":"Altria Group","PM":"Philip Morris International","CL":"Colgate-Palmolive",
    "EL":"Estee Lauder","KHC":"Kraft Heinz","GIS":"General Mills","HSY":"Hershey Co.",
    "GE":"GE Aerospace","CAT":"Caterpillar Inc.","HON":"Honeywell International",
    "UNP":"Union Pacific","RTX":"RTX Corp.","LMT":"Lockheed Martin","BA":"Boeing Co.",
    "DE":"Deere & Co.","GEV":"GE Vernova","ETN":"Eaton Corp.","EMR":"Emerson Electric",
    "ITW":"Illinois Tool Works","MMM":"3M Co.","FDX":"FedEx Corp.",
    "UPS":"United Parcel Service","CSX":"CSX Corp.","NSC":"Norfolk Southern",
    "WM":"Waste Management","ROK":"Rockwell Automation","PH":"Parker-Hannifin",
    "XOM":"Exxon Mobil","CVX":"Chevron Corp.","COP":"ConocoPhillips","SLB":"SLB",
    "EOG":"EOG Resources","MPC":"Marathon Petroleum","PSX":"Phillips 66",
    "VLO":"Valero Energy","OXY":"Occidental Petroleum","HAL":"Halliburton",
    "AMD":"Advanced Micro Devices","INTC":"Intel Corp.","QCOM":"Qualcomm Inc.",
    "TXN":"Texas Instruments","AMAT":"Applied Materials","LRCX":"Lam Research",
    "KLAC":"KLA Corp.","MU":"Micron Technology","MRVL":"Marvell Technology",
    "ADI":"Analog Devices","CDNS":"Cadence Design Systems","SNPS":"Synopsys Inc.",
    "ANSS":"Ansys Inc.","FTNT":"Fortinet","PANW":"Palo Alto Networks",
    "CRWD":"CrowdStrike","ZS":"Zscaler","ANET":"Arista Networks",
    "CSCO":"Cisco Systems","IBM":"IBM Corp.","HPQ":"HP Inc.",
    "DELL":"Dell Technologies","STX":"Seagate Technology","WDC":"Western Digital",
    "KEYS":"Keysight Technologies","CRM":"Salesforce Inc.","NOW":"ServiceNow",
    "INTU":"Intuit Inc.","ADBE":"Adobe Inc.","ADSK":"Autodesk","WDAY":"Workday Inc.",
    "VEEV":"Veeva Systems","DDOG":"Datadog Inc.","SNOW":"Snowflake","PLTR":"Palantir Technologies",
    "APP":"AppLovin Corp.","TEAM":"Atlassian Corp.","HUBS":"HubSpot Inc.",
    "ZM":"Zoom Video Communications","OKTA":"Okta Inc.","SPLK":"Splunk Inc.",
    "MCHP":"Microchip Technology","MPWR":"Monolithic Power Systems",
    "SWKS":"Skyworks Solutions","QRVO":"Qorvo Inc.","ON":"ON Semiconductor","ENPH":"Enphase Energy",
    "AMT":"American Tower","PLD":"Prologis","EQIX":"Equinix","CCI":"Crown Castle",
    "PSA":"Public Storage","O":"Realty Income","SPG":"Simon Property Group",
    "WELL":"Welltower","DLR":"Digital Realty","EXR":"Extra Space Storage",
    "AVB":"AvalonBay Communities","EQR":"Equity Residential","VTR":"Ventas Inc.","NXT":"Nextracker",
    "NEE":"NextEra Energy","DUK":"Duke Energy","SO":"Southern Co.",
    "AEP":"American Electric Power","EXC":"Exelon Corp.","XEL":"Xcel Energy",
    "PCG":"PG&E Corp.","SRE":"Sempra Energy","ED":"Consolidated Edison","ETR":"Entergy Corp.",
    "LIN":"Linde plc","APD":"Air Products & Chemicals","SHW":"Sherwin-Williams",
    "ECL":"Ecolab Inc.","NEM":"Newmont Corp.","FCX":"Freeport-McMoRan","NUE":"Nucor Corp.",
    "VMC":"Vulcan Materials","MLM":"Martin Marietta Materials","DOW":"Dow Inc.",
    "DD":"DuPont de Nemours","PPG":"PPG Industries","IFF":"International Flavors & Fragrances",
    "ZBH":"Zimmer Biomet","BDX":"Becton Dickinson","BAX":"Baxter International",
    "CAH":"Cardinal Health","MCK":"McKesson Corp.","ABC":"AmerisourceBergen",
    "CVS":"CVS Health Corp.","HUM":"Humana Inc.","MOH":"Molina Healthcare","CNC":"Centene Corp.",
    "NFLX":"Netflix Inc.","DIS":"Walt Disney Co.","CMCSA":"Comcast Corp.",
    "T":"AT&T Inc.","VZ":"Verizon Communications","TMUS":"T-Mobile US",
    "EA":"Electronic Arts","TTWO":"Take-Two Interactive","WBD":"Warner Bros. Discovery",
    "FOXA":"Fox Corp. Class A","FOX":"Fox Corp. Class B","BKNG":"Booking Holdings",
    "ABNB":"Airbnb Inc.","MAR":"Marriott International","HLT":"Hilton Worldwide",
    "RCL":"Royal Caribbean Group","CCL":"Carnival Corp.","GM":"General Motors",
    "F":"Ford Motor Co.","APTV":"Aptiv plc","RL":"Ralph Lauren Corp.",
    "TPR":"Tapestry Inc.","VFC":"VF Corp.","CME":"CME Group","ICE":"Intercontinental Exchange",
    "MCO":"Moody\'s Corp.","MSCI":"MSCI Inc.","FIS":"Fidelity National Info Services",
    "FISV":"Fiserv Inc.","PYPL":"PayPal Holdings","SQ":"Block Inc.","COIN":"Coinbase Global",
}

def get_display_name(t):
    return NAME_MAP.get(t, t)

# ── Portfolio persistence (weight-based schema) ───────────────────────────────
EMPTY_PORTFOLIO = {"total_value": DEFAULT_VALUE, "positions": []}

def load_portfolio():
    if not os.path.exists(PORTFOLIO_FILE):
        return {"total_value": DEFAULT_VALUE, "positions": []}
    try:
        with open(PORTFOLIO_FILE) as f:
            data = json.load(f)
        raw = data.get("positions", [])
        clean, legacy = [], False
        for p in raw:
            if "weight" in p:
                clean.append({"ticker": p["ticker"],
                              "name": p.get("name", get_display_name(p["ticker"])),
                              "weight": float(p["weight"])})
            elif "units" in p and "avg_buy_price" in p:   # migrate old schema
                legacy = True
                clean.append({"ticker": p["ticker"],
                              "name": p.get("name", get_display_name(p["ticker"])),
                              "weight": float(p["units"]) * float(p["avg_buy_price"])})
        if legacy:
            tot = sum(p["weight"] for p in clean)
            if tot > 0:
                for p in clean:
                    p["weight"] = round(p["weight"] / tot * 100, 2)
        return {"total_value": float(data.get("total_value", DEFAULT_VALUE)), "positions": clean}
    except Exception:
        return {"total_value": DEFAULT_VALUE, "positions": []}

def save_portfolio(positions, total_value=DEFAULT_VALUE):
    clean = []
    for p in positions:
        try:
            clean.append({"ticker": str(p["ticker"]),
                          "name": str(p.get("name", get_display_name(p["ticker"]))),
                          "weight": float(p["weight"])})
        except Exception:
            continue
    try:
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump({"total_value": float(total_value), "positions": clean}, f, indent=2)
        return None
    except OSError as e:
        return str(e)

# ── Data fetch helpers ─────────────────────────────────────────────────────────
def safe_fetch(ticker_str, period=DATA_PERIOD):
    try:
        hist = yf.Ticker(ticker_str).history(period=period, auto_adjust=True)
        return (None, f"No data for {ticker_str}") if hist.empty else (hist, None)
    except Exception as e:
        return None, str(e)

def bulk_fetch_closes(tickers, period=DATA_PERIOD):
    if USE_CACHE:
        return _cache_closes(tickers, period)
    if not tickers:
        return {}
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True)
        if raw.empty:
            return {}
        result = {}
        if len(tickers) == 1:
            t = tickers[0]
            if "Close" in raw.columns:
                s = raw["Close"].dropna()
                if not s.empty:
                    result[t] = s
        else:
            if "Close" in raw.columns:
                cls = raw["Close"]
                if isinstance(cls, pd.Series):
                    cls = cls.to_frame(name=tickers[0])
                for t in tickers:
                    if t in cls.columns:
                        s = cls[t].dropna()
                        if not s.empty:
                            result[t] = s
        return result
    except Exception as e:
        print(f"[WARN] bulk_fetch_closes: {e}")
        return {}

def _fetch_meta(ticker):
    if USE_CACHE:
        return _cache_meta_one(ticker)
    try:
        info = yf.Ticker(ticker).info
        return ticker, {
            "sector":   info.get("sector",   "Unknown") or "Unknown",
            "country":  info.get("country",  "Unknown") or "Unknown",
            "currency": info.get("currency", "USD")     or "USD",
            "name":     info.get("longName", get_display_name(ticker)) or get_display_name(ticker),
        }
    except Exception:
        return ticker, {"sector":"Unknown","country":"Unknown","currency":"USD","name":get_display_name(ticker)}


# ── Server-side price cache (cloud deployment) ─────────────────────────────────
# With DATA_MODE=cached, the whole ticker universe is fetched a couple of times a
# day on a background schedule and served from memory, so recruiter page loads
# never wait on — or get blocked by — live Yahoo requests. A blocked refresh keeps
# the last good cache; a cold start loads a snapshot committed to the repo.

def _universe():
    return list(dict.fromkeys(list(SP500_TOP200) + [BENCHMARK, RF_TICKER]))

def _period_to_days(period):
    period = (period or DATA_PERIOD).strip().lower()
    return {"1mo":31,"3mo":93,"6mo":186,"1y":366,"2y":731,"3y":1096,"5y":1827,"max":None}.get(period, 1096)

class _DataCache:
    def __init__(self):
        self.closes = {}          # {ticker: pd.Series indexed by date}
        self.meta   = {}          # {ticker: {sector,country,currency,name}}
        self.updated_at = None
        self.lock = threading.Lock()

DATA_CACHE = _DataCache()

def _frame_to_closes(df):
    out = {}
    for t in df.columns:
        s = df[t].dropna()
        if not s.empty:
            s.name = t
            out[t] = s
    return out

def _write_cache(closes, meta, directory):
    os.makedirs(directory, exist_ok=True)
    frame = pd.concat(closes, axis=1) if closes else pd.DataFrame()
    frame.index = pd.to_datetime(frame.index)
    frame.to_parquet(os.path.join(directory, "prices.parquet"))
    with open(os.path.join(directory, "meta.json"), "w") as f:
        json.dump(meta, f)

def _read_cache(directory):
    ppath = os.path.join(directory, "prices.parquet")
    if not os.path.exists(ppath):
        return None, None
    frame = pd.read_parquet(ppath)
    frame.index = pd.to_datetime(frame.index)
    closes = _frame_to_closes(frame)
    meta = {}
    mpath = os.path.join(directory, "meta.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            meta = json.load(f)
    return closes, meta

def _live_download_closes(tickers):
    """Download long-history closes for the universe as {ticker: Series}."""
    raw = yf.download(tickers, start=HISTORY_START, auto_adjust=True,
                      progress=False, threads=True)
    if raw is None or raw.empty:
        return {}
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            return {}
        cls = raw["Close"]
    else:
        if "Close" not in raw.columns:
            return {}
        cls = raw[["Close"]].copy()
        cls.columns = [tickers[0]]
    return _frame_to_closes(cls)

def _live_meta_one(ticker):
    try:
        info = yf.Ticker(ticker).info
        return ticker, {
            "sector":   info.get("sector",   "Unknown") or "Unknown",
            "country":  info.get("country",  "Unknown") or "Unknown",
            "currency": info.get("currency", "USD")     or "USD",
            "name":     info.get("longName", get_display_name(ticker)) or get_display_name(ticker),
        }
    except Exception:
        return ticker, {"sector":"Unknown","country":"Unknown","currency":"USD",
                        "name":get_display_name(ticker)}

def refresh_cache():
    """Refresh the whole-universe cache. On any failure, keep the last good cache."""
    universe = _universe()
    try:
        closes = _live_download_closes(universe)
    except Exception as e:
        print(f"[cache] price refresh blocked/failed ({e}). Keeping last good cache")
        return False
    if not closes:
        print("[cache] price refresh returned no data. Keeping last good cache")
        return False
    stock_tickers = [t for t in universe if t not in (BENCHMARK, RF_TICKER)]
    meta = {}
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_live_meta_one, t): t for t in stock_tickers}
            for fut in as_completed(futs):
                t, m = fut.result()
                meta[t] = m
    except Exception as e:
        print(f"[cache] metadata refresh partial ({e})")
    with DATA_CACHE.lock:
        DATA_CACHE.closes = closes
        if meta:
            DATA_CACHE.meta = meta
        DATA_CACHE.updated_at = datetime.now(_ET)
    try:
        _write_cache(DATA_CACHE.closes, DATA_CACHE.meta, CACHE_DIR)
    except Exception as e:
        print(f"[cache] could not persist cache to disk ({e})")
    print(f"[cache] refreshed {len(closes)} tickers at {DATA_CACHE.updated_at}")
    return True

def _cache_closes(tickers, period=DATA_PERIOD):
    if not tickers:
        return {}
    days = _period_to_days(period)
    out = {}
    with DATA_CACHE.lock:
        src = DATA_CACHE.closes
        for t in tickers:
            s = src.get(t)
            if s is None or s.empty:
                continue
            if days:
                s = s[s.index >= (s.index.max() - pd.Timedelta(days=days))]
            out[t] = s.copy()
    return out

def _cache_closes_df(tickers, start, end):
    with DATA_CACHE.lock:
        cols = {t: DATA_CACHE.closes[t] for t in tickers
                if t in DATA_CACHE.closes and not DATA_CACHE.closes[t].empty}
    if not cols:
        return pd.DataFrame()
    df = pd.concat(cols, axis=1)
    return df[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]

def _cache_meta_one(ticker):
    with DATA_CACHE.lock:
        m = DATA_CACHE.meta.get(ticker)
    if m:
        return ticker, dict(m)
    return ticker, {"sector":"Unknown","country":"Unknown","currency":"USD",
                    "name":get_display_name(ticker)}

def bootstrap_cache():
    closes, meta = _read_cache(CACHE_DIR)
    src = "runtime cache"
    if not closes:
        closes, meta = _read_cache(SNAPSHOT_DIR)
        src = "committed snapshot"
    if closes:
        with DATA_CACHE.lock:
            DATA_CACHE.closes = closes
            DATA_CACHE.meta = meta or {}
        print(f"[cache] loaded {len(closes)} tickers from {src}")
    else:
        print("[cache] no runtime cache or snapshot found. Will fetch on first refresh")
    threading.Thread(target=refresh_cache, daemon=True).start()

def _next_run_delay():
    """Seconds until the next refresh: 12:30 and 16:15 ET on weekdays."""
    now = datetime.now(_ET)
    candidates = []
    for add_days in range(0, 5):
        day = now + timedelta(days=add_days)
        if day.weekday() >= 5:      # skip Sat/Sun
            continue
        for hh, mm in ((12, 30), (16, 15)):
            run = day.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if run > now:
                candidates.append(run)
    return max(30.0, (min(candidates) - now).total_seconds())

def _scheduler_loop():
    while True:
        try:
            time.sleep(_next_run_delay())
            refresh_cache()
        except Exception as e:
            print(f"[cache] scheduler error ({e}). Retrying in 5 min")
            time.sleep(300)

def start_scheduler():
    threading.Thread(target=_scheduler_loop, daemon=True).start()
    print("[cache] refresh scheduler started (12:30 and 16:15 ET, weekdays)")

def _get_rf_annual(prices_store):
    try:
        if prices_store and RF_TICKER in prices_store:
            cls = prices_store[RF_TICKER]["closes"]
            if cls:
                return float(cls[-1]) / 100.0
    except Exception:
        pass
    return 0.05

def _prices_to_series(prices_store, ticker):
    if not prices_store or ticker not in prices_store:
        return None
    d = prices_store[ticker]
    return pd.Series(data=d["closes"], index=pd.to_datetime(d["dates"]), name=ticker)

def _series_dict(prices_store, tickers):
    """Build {ticker: price Series} in a single pass.
    Avoids constructing each Series twice (filter + value) as the old
    dict-comprehensions did, which doubled pandas object creation on every
    Overview/Forecast refresh."""
    out = {}
    for t in tickers:
        s = _prices_to_series(prices_store, t)
        if s is not None:
            out[t] = s
    return out

# ── Quant helpers ──────────────────────────────────────────────────────────────
def _get_weights(portfolio_data):
    """Return {ticker: normalized_weight_fraction} (sums to 1)."""
    positions = (portfolio_data or {}).get("positions", [])
    w = {}
    for p in positions:
        try:
            val = float(p.get("weight", 0))
            if val > 0:
                w[p["ticker"]] = val
        except Exception:
            continue
    tot = sum(w.values())
    if tot > 0:
        w = {t: v / tot for t, v in w.items()}
    return w

# ── Backtest windows (trading days) ────────────────────────────────────────────
OV_WINDOWS = {"6M": 126, "1Y": 252, "3Y": None}      # None = full 36-month window
OV_LABELS  = {"6M": "6 Months", "1Y": "1 Year", "3Y": "3 Years"}

def _slice_window(returns, window):
    """Last N trading days of a daily return series ('3Y' = everything)."""
    n = OV_WINDOWS.get(window or "3Y")
    if n is None or returns is None or len(returns) <= n:
        return returns
    return returns.iloc[-n:]

def compute_portfolio_returns(weights, price_series_dict):
    frames = {t: price_series_dict[t].rename(t)
              for t, w in weights.items()
              if t in price_series_dict and w > 0}
    if not frames:
        return pd.Series(dtype=float)
    prices_df = pd.concat(frames, axis=1).dropna()
    rets = prices_df.pct_change().dropna()
    w_arr = np.array([weights.get(c, 0.0) for c in rets.columns], dtype=float)
    s = w_arr.sum()
    if s > 0:
        w_arr /= s
    return pd.Series(rets.values @ w_arr, index=rets.index)

def compute_metrics(returns, benchmark_returns=None, rf_annual=0.05):
    if returns is None or returns.empty or len(returns) < 5:
        return {}
    rf_d = rf_annual / 252.0
    std_d = returns.std()
    # Geometric (compound) annualization: consistent with the Monte Carlo drift
    # and free of the upward arithmetic-mean bias under volatility.
    n_obs   = len(returns)
    ann_ret = float((1 + returns).prod() ** (252.0 / n_obs) - 1)
    ann_vol = std_d * np.sqrt(252)
    sharpe  = (ann_ret - rf_annual) / ann_vol if ann_vol > 1e-8 else np.nan
    down = returns[returns < 0]
    down_std = down.std() * np.sqrt(252) if len(down) > 1 else np.nan
    sortino  = (ann_ret - rf_annual) / down_std if (down_std and down_std > 1e-8) else np.nan
    cum    = (1 + returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    var95  = float(-np.percentile(returns, 5))
    mask   = returns <= -var95
    cvar95 = float(-returns[mask].mean()) if mask.any() else var95
    beta = beta_raw = corr = ir = te = bench_ar = np.nan
    if benchmark_returns is not None and not benchmark_returns.empty:
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        if len(aligned) > 5:
            p, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
            bench_ar = float((1 + b).prod() ** (252.0 / len(b)) - 1)   # geometric
            vb = float(np.var(b))
            beta_raw = float(np.cov(p, b)[0, 1] / vb) if vb > 1e-12 else np.nan
            # Blume adjustment: shrink the sample beta toward 1 to counter the
            # estimation noise of daily-frequency betas.
            beta = 0.67 * beta_raw + 0.33 if not np.isnan(beta_raw) else np.nan
            corr = float(np.corrcoef(p, b)[0, 1])
            diff = p - b
            te   = diff.std() * np.sqrt(252)
            ir   = (ann_ret - bench_ar) / te if te > 1e-8 else np.nan
    gains  = returns[returns > rf_d].sum()
    losses = abs(returns[returns < rf_d].sum())
    omega  = gains / losses if losses > 1e-8 else np.nan
    return {
        "ann_return":    round(ann_ret,  4),
        "ann_vol":       round(ann_vol,  4),
        "sharpe":        None if np.isnan(sharpe)  else round(sharpe,   4),
        "sortino":       None if np.isnan(sortino) else round(sortino,  4),
        "max_drawdown":  round(max_dd,   4),
        "calmar":        None if np.isnan(calmar)  else round(calmar,   4),
        "var95":         round(var95,    4),
        "cvar95":        round(cvar95,   4),
        "beta":          None if np.isnan(beta)     else round(beta,     4),
        "beta_raw":      None if np.isnan(beta_raw) else round(beta_raw, 4),
        "correlation":   None if np.isnan(corr)    else round(corr,     4),
        "info_ratio":    None if np.isnan(ir)       else round(ir,      4),
        "tracking_err":  None if np.isnan(te)       else round(te,      4),
        "omega":         None if np.isnan(omega)   else round(omega,    4),
        "bench_ann_ret": None if np.isnan(bench_ar) else round(bench_ar,4),
    }

def fit_garch_vol(returns_series):
    try:
        m = arch_model(returns_series * 100, vol="Garch", p=1, q=1, rescale=False)
        res = m.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=1, reindex=False)
        daily_var = float(fc.variance.iloc[-1, 0]) / 10_000
        return float(np.sqrt(max(daily_var, 1e-10) * 252))
    except Exception:
        return float(returns_series.std() * np.sqrt(252))

def fit_garch_params(returns_series):
    """GARCH(1,1) parameters for variance-path simulation.

    Returns (omega, alpha, beta, sigma2_0) in DAILY DECIMAL variance units, or
    None when the fit fails or is non-stationary (alpha + beta >= 1), in which
    case the caller falls back to constant-volatility GBM. Simulating the
    variance path (rather than using one point forecast) preserves what GARCH
    actually models: volatility clustering and mean reversion in variance.
    """
    try:
        m = arch_model(returns_series * 100, vol="Garch", p=1, q=1, rescale=False)
        res = m.fit(disp="off", show_warning=False)
        p = res.params
        omega = float(p["omega"]) / 10_000          # percent^2 -> decimal^2
        alpha = float(p["alpha[1]"])
        beta  = float(p["beta[1]"])
        fc    = res.forecast(horizon=1, reindex=False)
        s2_0  = float(fc.variance.iloc[-1, 0]) / 10_000
        vals  = np.array([omega, alpha, beta, s2_0])
        if not np.isfinite(vals).all():
            return None
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999 or s2_0 <= 0:
            return None
        return omega, alpha, beta, s2_0
    except Exception:
        return None

def shrunk_cov(R):
    """
    Ledoit-Wolf shrinkage estimator of the daily covariance matrix.
    Stabilises the noisy 1-year sample covariance so mean-variance
    optimisation stops producing extreme corner solutions.
    Graceful fallback chain: sklearn LedoitWolf -> manual diagonal
    shrinkage -> raw sample covariance. Always returns a valid matrix.
    """
    try:
        from sklearn.covariance import LedoitWolf
        return LedoitWolf().fit(R).covariance_
    except Exception:
        try:
            S = np.cov(R, rowvar=False)
            n = S.shape[0]
            target = np.eye(n) * (np.trace(S) / n)   # shrink toward scaled identity
            shrink = 0.2
            return shrink * target + (1 - shrink) * S
        except Exception:
            return np.cov(R, rowvar=False)

def shrink_mean(mu_hist, intensity=0.5):
    """
    James-Stein style shrinkage of historical mean returns toward the
    cross-sectional grand mean. Raw historical means are the weakest
    input to MVO ('error maximisation'). Blending toward the average
    tames the optimiser's overconfidence. intensity=0 keeps history,
    1 collapses to the grand mean.
    """
    try:
        mu = np.asarray(mu_hist, dtype=float)
        grand = float(np.mean(mu))
        return (1 - intensity) * mu + intensity * grand
    except Exception:
        return np.asarray(mu_hist, dtype=float)

# ── UI helpers ─────────────────────────────────────────────────────────────────
def _dark_layout(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(color=CLR_TEXT, size=13)) if title else None,
        paper_bgcolor=CLR_CARD, plot_bgcolor=CLR_CARD,
        font=dict(color=CLR_TEXT, size=11, family="Inter, 'Segoe UI', system-ui, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=CLR_TEXT)),
        margin=dict(t=45 if title else 20, b=45, l=55, r=20),
        xaxis=dict(gridcolor=CLR_GRID, zerolinecolor="#38404F"),
        yaxis=dict(gridcolor=CLR_GRID, zerolinecolor="#38404F"),
    )
    return fig

def _metric_card(label, value, color=CLR_TEXT, delta=None):
    delta_el = None
    if delta:
        dc = CLR_GREEN if "+" in delta else (CLR_RED if "-" in delta else CLR_TEXT)
        delta_el = html.Small(delta, style={"color": dc})
    return dbc.Col(
        dbc.Card(dbc.CardBody([
            html.P(label, className="mb-0 text-muted",
                   style={"fontSize":"0.72rem","textTransform":"uppercase","letterSpacing":"0.05em"}),
            html.H4(value, style={"color":color,"fontWeight":"700","margin":"4px 0 0"}),
            delta_el or html.Span(),
        ]), style={"backgroundColor":CLR_CARD,"border":f"1px solid {CLR_BORDER}","borderRadius":"12px"}),
        md=3, sm=6, xs=12,
    )

def _no_data_alert(msg="No data - click 'Refresh All Data' to load prices."):
    return dbc.Alert(msg, color="secondary", className="my-2 py-2")

def _empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5,
                       showarrow=False, font=dict(color="#888", size=14))
    return _dark_layout(fig)

def _make_metrics_table(m):
    if not m:
        return html.P("No metrics.", className="text-muted small")
    DEFS = [
        ("ann_return",   "Annualised Return",      lambda v: f"{v:.2%}", "ret"),
        ("ann_vol",      "Annualised Volatility",  lambda v: f"{v:.2%}", "vol"),
        ("sharpe",       "Sharpe Ratio",           lambda v: f"{v:.4f}", "sharpe"),
        ("sortino",      "Sortino Ratio",          lambda v: f"{v:.4f}", "other"),
        ("max_drawdown", "Max Drawdown",           lambda v: f"{v:.2%}", "dd"),
        ("calmar",       "Calmar Ratio",           lambda v: f"{v:.4f}", "other"),
        ("var95",        "VaR 95%",                lambda v: f"{v:.2%}", "vol"),
        ("cvar95",       "CVaR 95%",               lambda v: f"{v:.2%}", "vol"),
        ("beta",         f"Beta (Blume-adj, vs {BENCH_LABEL})", lambda v: f"{v:.4f}", "other"),
        ("beta_raw",     "Beta (raw sample)",      lambda v: f"{v:.4f}", "other"),
        ("correlation",  "Correlation",            lambda v: f"{v:.4f}", "other"),
        ("info_ratio",   "Information Ratio",      lambda v: f"{v:.4f}", "other"),
        ("omega",        "Omega Ratio",            lambda v: f"{v:.4f}", "other"),
        ("bench_ann_ret","Benchmark Ann. Return",  lambda v: f"{v:.2%}", "ret"),
    ]
    rows = []
    for key, label, fmt, kind in DEFS:
        val = m.get(key)
        if val is None:
            continue
        try:
            disp = fmt(val)
        except Exception:
            disp = str(val)
        if   kind == "ret":    c = CLR_GREEN if val > 0 else CLR_RED
        elif kind == "dd":     c = CLR_RED if val < -0.15 else (CLR_YELLOW if val < -0.05 else CLR_GREEN)
        elif kind == "sharpe": c = CLR_GREEN if val > 1.0 else (CLR_YELLOW if val > 0.5 else CLR_RED)
        elif kind == "vol":    c = CLR_RED if val > 0.25 else (CLR_YELLOW if val > 0.15 else CLR_GREEN)
        else:                  c = CLR_TEXT
        rows.append(html.Tr([
            html.Td(label, style={"color":"#aaa","fontSize":"11px","padding":"3px 8px"}),
            html.Td(disp,  style={"color":c,"fontWeight":"600","fontSize":"12px",
                                  "fontFamily":"monospace","padding":"3px 8px"}),
        ]))
    return dbc.Table(html.Tbody(rows), bordered=False, size="sm",
                     style={"backgroundColor":"transparent","marginBottom":0})

def _parse_table_data(table_data):
    if not table_data:
        return []
    result = []
    for row in table_data:
        try:
            w = float(row.get("weight", 0) or 0)
            if w > 0:
                result.append({"ticker": str(row["ticker"]),
                               "name": str(row.get("name", get_display_name(row["ticker"]))),
                               "weight": round(w, 4)})
        except Exception:
            continue
    return result

def _get_current_positions(table_data, portfolio_data):
    if table_data is not None:
        parsed = _parse_table_data(table_data)
        if parsed:
            return parsed
    return list((portfolio_data or {}).get("positions", []))

# ── App init ───────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True,
                meta_tags=[{"name":"viewport","content":"width=device-width, initial-scale=1"}],
                title="Frontier | S&P 500 Portfolio Dashboard")
server = app.server

# Modern visual layer: Inter typeface + CSS refinements over the DARKLY base.
app.index_string = """<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#0E1117; --card:#161B24; --border:#252B38; --text:#E6E9EF;
  --accent:#5B8DEF; --green:#2DD4A4; --radius:12px;
}
html,body{background:var(--bg)!important;color:var(--text);
  font-family:'Inter','Segoe UI',system-ui,-apple-system,sans-serif!important;
  -webkit-font-smoothing:antialiased;}
h1,h2,h3,h4,h5,h6{font-weight:700;letter-spacing:-0.01em;}
.navbar{background:rgba(14,17,23,0.85)!important;backdrop-filter:blur(10px);
  border-bottom:1px solid var(--border);box-shadow:none!important;}
.card{background:var(--card)!important;border:1px solid var(--border)!important;
  border-radius:var(--radius)!important;box-shadow:0 1px 2px rgba(0,0,0,0.25);
  transition:border-color .15s ease;}
.card:hover{border-color:#323B4D!important;}
.nav-tabs{border-bottom:1px solid var(--border)!important;gap:2px;}
.nav-tabs .nav-link{color:#8B93A7!important;border:none!important;
  border-bottom:2px solid transparent!important;border-radius:0!important;
  font-weight:600;font-size:0.85rem;letter-spacing:0.04em;text-transform:uppercase;
  padding:0.65rem 1.1rem;background:transparent!important;transition:color .15s ease;}
.nav-tabs .nav-link:hover{color:var(--text)!important;}
.nav-tabs .nav-link.active{color:var(--accent)!important;background:transparent!important;
  border-bottom:2px solid var(--accent)!important;}
.btn{border-radius:8px!important;font-weight:600;letter-spacing:0.01em;
  transition:transform .1s ease, box-shadow .15s ease;}
.btn:hover{transform:translateY(-1px);}
.btn:active{transform:translateY(0);}
.form-control,.form-select{background:#10141C!important;border:1px solid var(--border)!important;
  border-radius:8px!important;color:var(--text)!important;}
.form-control:focus{border-color:var(--accent)!important;
  box-shadow:0 0 0 3px rgba(91,141,239,0.15)!important;}
.alert{border-radius:10px!important;border:1px solid var(--border)!important;}
.toast{border-radius:10px!important;}
.Select-control,.Select-menu-outer{border-radius:8px!important;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:var(--bg);}
::-webkit-scrollbar-thumb{background:#2A3242;border-radius:6px;border:2px solid var(--bg);}
::-webkit-scrollbar-thumb:hover{background:#38445A;}
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""

# ── Navbar ─────────────────────────────────────────────────────────────────────
# ── Start guide (guided tour) ──────────────────────────────────────────────────
def _g(*children):
    return html.Div(list(children))

GUIDE_STEPS = [
    ("Welcome to Frontier", _g(
        html.P("Frontier takes a portfolio of S&P 500 large caps from construction to "
               "risk analysis, optimization, and simulation. The workflow follows the tabs "
               "left to right:"),
        html.Ul([
            html.Li([html.B("Builder"), " - pick stocks and assign percentage weights."]),
            html.Li([html.B("Overview"), " - performance, risk metrics, and allocation breakdowns."]),
            html.Li([html.B("Optimizer"), " - the mean-variance optimal allocation and efficient frontier."]),
            html.Li([html.B("Forecasting"), " - Monte Carlo simulation and historical stress tests."]),
        ]),
        html.P("This guide walks through each tab and explains the models behind the numbers. "
               "Use Next and Back to move through it. Close it at any time and reopen it from "
               "the Start Guide button."))),
    ("Builder: defining the portfolio", _g(
        html.P("A portfolio here is a set of tickers with percentage weights plus a total "
               "dollar value. Pick a stock from the dropdown, give it a weight, and click Add. "
               "The table is editable, so weights can be tuned in place."),
        html.P("Weights should sum to 100%. Normalize rescales whatever you entered "
               "proportionally, which is convenient when ideas arrive faster than arithmetic. "
               "Save writes the portfolio to disk so it survives a restart, and Launch jumps "
               "straight to the analytics."),
        html.P("Click Refresh All Data after changing positions: it loads three years of "
               "dividend-adjusted daily prices for your holdings, the benchmark, and the "
               "risk-free rate.", className="mb-0"))),
    ("The data behind everything", _g(
        html.P("All analytics run on a 36-month window of daily closing prices, adjusted for "
               "dividends and splits. Three years balances two failure modes: shorter windows "
               "make tail metrics like VaR statistical noise, while much longer windows "
               "average over regimes that no longer describe the market."),
        html.P(["The benchmark is the ", html.B("S&P 500 Total Return index"), ", not the "
               "familiar price index. Holdings include reinvested dividends, so the benchmark "
               "must too. Comparing against the price-only index would flatter any portfolio "
               "by roughly the dividend yield every year."]),
        html.P("The risk-free rate is read from the 13-week US Treasury bill yield.",
               className="mb-0"))),
    ("Overview: performance cards", _g(
        html.P("The top row shows portfolio value (initial value compounded through the "
               "backtest), the latest one-day move, annualized volatility, and the position "
               "count. The 6M / 1Y / 3Y toggle changes the backtest window for these cards "
               "and the cumulative chart."),
        html.P("Reported returns are geometric, meaning the constant compound rate that "
               "reproduces the actual end value. The alternative, annualizing the average "
               "daily return, systematically overstates growth for volatile assets, which is "
               "why the geometric convention is used consistently here, including inside the "
               "Monte Carlo drift."),
        html.P("A caution built into the design: shorter windows are noisier estimates. Six "
               "months of daily data says little about true volatility, which is why the "
               "deeper risk metrics elsewhere always use the full 36 months.", className="mb-0"))),
    ("Overview: reading the risk metrics", _g(
        html.P([html.B("Sharpe"), " is excess return per unit of total volatility. ",
               html.B("Sortino"), " counts only downside volatility, since upside surprises "
               "are not risk in any sense an investor minds. ", html.B("Calmar"),
               " divides return by the worst peak-to-trough drawdown of the window."]),
        html.P([html.B("VaR 95%"), " is the daily loss exceeded on the worst 5% of days. ",
               html.B("CVaR"), " is the average loss on those days, which answers the better "
               "question: when things go wrong, how wrong on average? CVaR is also the "
               "theoretically sounder (coherent) of the two."]),
        html.P([html.B("Beta"), " measures co-movement with the benchmark. Sample betas from "
               "daily data are noisy, so the displayed value is Blume-adjusted: shrunk one "
               "third of the way toward 1, reflecting the empirical tendency of betas to "
               "drift toward the market average. ", html.B("Information Ratio"),
               " is benchmark-relative return per unit of tracking error, and ",
               html.B("Omega"), " is the ratio of probability-weighted gains to losses "
               "around the risk-free threshold."], className="mb-0"))),
    ("Overview: allocation and the backtest chart", _g(
        html.P("The three donuts split the portfolio by weight, sector, and country or "
               "currency, using per-instrument metadata. Concentration that looks fine in a "
               "ticker list is usually visible at a glance here."),
        html.P("The cumulative chart rebases both the portfolio and the benchmark to 100 at "
               "the start of the selected window, so the vertical gap at any date is the "
               "cumulative out- or under-performance. The position table below it shows "
               "trailing 1M / 6M / 1Y / 3Y returns per holding plus each position's own "
               "volatility and Blume-adjusted beta.", className="mb-0"))),
    ("Optimizer: mean-variance and its repairs", _g(
        html.P("The optimizer solves the Markowitz problem: among long-only, fully invested "
               "portfolios of your tickers, find the weights that maximize Sharpe, minimize "
               "volatility, or maximize Sortino, subject to your per-position minimum and "
               "maximum bounds."),
        html.P("Naive mean-variance amplifies estimation error: it loads weight onto whatever "
               "was lucky in-sample. Two standard repairs are applied. Ledoit-Wolf shrinkage "
               "stabilizes the covariance matrix by pulling it toward a structured target, "
               "and James-Stein-style shrinkage pulls each stock's historical mean toward the "
               "cross-sectional average, damping the influence of outlier histories."),
        html.P("The grey cloud is 10,000 random long-only portfolios. Its upper-left edge "
               "traces the efficient frontier, and the marker is your optimized point. The "
               "frontier axis is labeled in the arithmetic convention because that is the "
               "space mean-variance mathematics is defined on. Honest caveat: even shrunk "
               "historical means are weak forecasts, which is why institutional practice "
               "uses equilibrium-implied returns, something that is beyond this build's scope.",
               className="mb-0"))),
    ("Forecasting: the Monte Carlo engine", _g(
        html.P("The simulation projects thousands of possible portfolio paths. Each daily "
               "step adds a fixed drift plus a random shock scaled by that day's volatility."),
        html.P("Volatility is not constant: it is simulated from a fitted GARCH(1,1) model, "
               "so calm and turbulent stretches cluster the way real markets do and "
               "volatility mean-reverts toward its long-run level. The shocks are Student-t "
               "with 5 degrees of freedom, standardized to unit variance, giving fat tails "
               "(more extreme days than a normal distribution) without inflating the "
               "calibrated volatility."),
        html.P("The drift is anchored to the realized geometric return, so the median path "
               "compounds at the same rate the Overview reports. Read the fan chart as a "
               "distribution, not a prediction: the bands show where the chosen percentage "
               "of simulated outcomes fell. If the GARCH fit fails or is unstable, the "
               "engine falls back to constant volatility and says so in the chart title.",
               className="mb-0"))),
    ("Scenarios, limitations, and honest caveats", _g(
        html.P("Scenario replay applies your current weights to seven historical stress "
               "windows, from the 2008 financial crisis to the 2022 rate-hike bear market, "
               "showing how the allocation would have ridden through each."),
        html.P("Two limitations worth knowing. First, survivorship bias: the replay uses "
               "today's index constituents, the companies that survived, so crisis results "
               "look more resilient than the index did at the time. Second, correlations are "
               "estimated over the full window and treated as fixed, but in real crises "
               "correlations rise and diversification weakens exactly when it is needed."),
        html.P("Everything here is a historical estimate, and the forecast is illustrative. "
               "Nothing in this dashboard is investment advice. That is the end of the tour. "
               , className="mb-0"))),
]

def make_guide_modal():
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle(id="guide-title"), close_button=True),
        dbc.ModalBody(id="guide-body"),
        dbc.ModalFooter([
            html.Span(id="guide-progress", className="text-muted me-auto",
                      style={"fontSize":"0.78rem"}),
            dbc.Button("Back", id="btn-guide-back", color="secondary", outline=True, size="sm"),
            dbc.Button("Next", id="btn-guide-next", color="info", size="sm"),
        ]),
    ], id="modal-guide", size="lg", is_open=False, scrollable=True, backdrop=True)

def make_navbar():
    return dbc.Navbar(dbc.Container([
        dbc.NavbarBrand([html.Span("Frontier", className="fw-bold fs-5", style={"letterSpacing":"0.02em"}),
                         html.Span("  S&P 500 PORTFOLIO ANALYTICS", className="text-muted ms-2",
                                   style={"fontSize":"0.68rem","letterSpacing":"0.14em"})]),
        dbc.Nav([
            dbc.NavItem(dbc.Button("Start Guide", id="btn-guide",
                color="info", outline=True, size="sm", className="me-2", n_clicks=0)),
            dbc.NavItem(dbc.Button("Refresh All Data", id="btn-refresh",
                color="primary", outline=True, size="sm", className="me-2", n_clicks=0)),
            dbc.NavItem(dbc.Button("Save Portfolio", id="btn-save",
                color="success", outline=True, size="sm", n_clicks=0)),
        ], navbar=True, className="ms-auto gap-1"),
    ], fluid=True), color="dark", dark=True, sticky="top", className="mb-3 shadow")

# ── Tab 1 — Builder (weight-based) ────────────────────────────────────────────
def make_tab_builder(boot_value, boot_rows):
    return dbc.Container([
        dbc.Row(dbc.Col(html.H4("Portfolio Builder", className="mt-3 mb-1 text-info"))),
        dbc.Alert("Build a portfolio by picking stocks and assigning each a weight (%). "
                  "Weights should add up to 100% (use the Normalize button to auto-scale).",
                  color="info", className="py-2 small"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col([dbc.Label("Total Portfolio Value ($)"),
                    dbc.Input(id="inp-total-value", type="number", value=boot_value,
                              min=0, step="any", debounce=True)], md=4),
            ], className="g-2"),
        ]), className="mb-3 shadow-sm"),

        dbc.Card(dbc.CardBody([
            html.H6("Add a Stock", className="card-title text-muted mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Stock"),
                    dcc.Dropdown(id="dd-ticker",
                        options=[{"label": f"{t}  -  {get_display_name(t)}", "value": t} for t in SP500_TOP200],
                        placeholder="Search ticker or company name...", clearable=True, searchable=True,
                        style={"color":"#111"})], md=7),
                dbc.Col([dbc.Label("Weight (%)"),
                    dbc.Input(id="inp-weight", type="number", placeholder="e.g. 25",
                              min=0, max=100, step="any")], md=3),
                dbc.Col([dbc.Label("\u00a0"),
                    dbc.Button("\u002b Add", id="btn-add-position", color="info",
                               className="w-100", n_clicks=0)], md=2),
            ], className="g-2 align-items-end"),
            html.Div(id="alert-add-position", className="mt-2"),
        ]), className="mb-3 shadow-sm"),

        dbc.Card(dbc.CardBody([
            dbc.Row([
                dbc.Col(html.H6("Current Positions", className="card-title text-muted mb-0"), width="auto"),
                dbc.Col(html.Div(id="div-total-weight"), width="auto", className="ms-3"),
                dbc.Col([
                    dbc.Button("Normalize to 100%", id="btn-normalize",
                               color="secondary", outline=True, size="sm", className="me-2", n_clicks=0),
                    dbc.Button("Clear All", id="btn-clear-all",
                               color="danger", outline=True, size="sm", n_clicks=0),
                ], className="ms-auto", width="auto"),
            ], align="center", className="mb-2"),
            dash_table.DataTable(
                id="positions-table",
                columns=[
                    {"name":"Ticker",     "id":"ticker", "editable":False},
                    {"name":"Company",    "id":"name",   "editable":False},
                    {"name":"Weight (%)", "id":"weight", "editable":True, "type":"numeric"},
                ],
                data=boot_rows, editable=True, row_deletable=True, page_size=30,
                style_table={"overflowX":"auto"},
                style_header={"backgroundColor":"#1a1a2e","color":CLR_TEXT,
                              "fontWeight":"bold","border":f"1px solid {CLR_BORDER}"},
                style_data={"backgroundColor":CLR_CARD,"color":CLR_TEXT,"border":f"1px solid {CLR_BORDER}"},
                style_data_conditional=[
                    {"if":{"state":"active"},"backgroundColor":"#3a3a50",
                     "border":f"1px solid {CLR_BLUE}","color":CLR_TEXT},
                    {"if":{"column_editable":True},"backgroundColor":"#2e2e3e"},
                ],
                style_cell={"fontFamily":"monospace","fontSize":"13px","padding":"7px 12px","textAlign":"left"},
                style_cell_conditional=[
                    {"if":{"column_id":"ticker"}, "width":"90px","fontWeight":"bold"},
                    {"if":{"column_id":"name"},   "width":"260px","maxWidth":"260px",
                     "overflow":"hidden","textOverflow":"ellipsis"},
                    {"if":{"column_id":"weight"}, "width":"120px"},
                ],
            ),
            html.Div(id="div-no-positions"),
        ]), className="mb-3 shadow-sm"),

        dbc.Row(dbc.Col(dbc.Button("Launch Dashboard \u2192", id="btn-launch",
            color="success", size="lg", className="w-100", n_clicks=0)), className="mb-4"),
        html.Div(id="div-launch-feedback"),
    ], fluid=True)

# ── Tab 2 — Overview ───────────────────────────────────────────────────────────
def make_tab_overview():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H4("Portfolio Overview", className="mt-3 mb-1 text-info"), md=6),
            dbc.Col(html.Div([
                dbc.RadioItems(id="radio-ov-window", inline=True, value="3Y",
                    options=[{"label":"6M","value":"6M"},
                             {"label":"1Y","value":"1Y"},
                             {"label":"3Y","value":"3Y"}],
                    inputClassName="btn-check", labelClassName="btn btn-outline-info btn-sm",
                    labelCheckedClassName="active", className="btn-group"),
                html.Div("Backtest window. Shorter windows are noisier estimates. "
                         "Risk metrics elsewhere use the full 36 months.",
                         className="text-muted", style={"fontSize":"0.68rem","marginTop":"4px"}),
            ], className="mt-3 text-md-end"), md=6),
        ], align="center"),
        dbc.Row(id="row-summary-metrics", className="mb-3 g-3"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Weight Allocation", className="card-title text-muted"),
                dcc.Graph(id="fig-weight-donut", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Sector Breakdown", className="card-title text-muted"),
                dcc.Graph(id="fig-sector-donut", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), className="shadow-sm"), md=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Currency / Country Split", className="card-title text-muted"),
                dcc.Graph(id="fig-country-donut", config={"displayModeBar":False}, style={"height":"280px"}),
            ]), className="shadow-sm"), md=4),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody([
            html.H6(f"Cumulative Return vs {BENCH_LABEL} (Normalised to 100)", className="card-title text-muted"),
            dcc.Loading(dcc.Graph(id="fig-cumulative-return", config={"displayModeBar":True},
                                  style={"height":"380px"}), type="circle"),
        ]), className="mb-3 shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Position Detail", className="card-title text-muted"),
            dcc.Loading(html.Div(id="div-position-detail", style={"overflowX":"auto"}), type="circle"),
        ]), className="mb-4 shadow-sm"),
    ], fluid=True)

# ── Tab 3 — Optimizer ──────────────────────────────────────────────────────────
def make_tab_optimizer():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H4("Portfolio Optimizer", className="mt-3 mb-1 text-info"))),
        dbc.Card(dbc.CardBody([
            html.H6("Optimization Settings", className="card-title text-muted mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Risk Mode"),
                    dcc.Dropdown(id="dd-risk-mode", options=[{"label":k,"value":k} for k in RISK_MODES],
                        value="Moderate", clearable=False, style={"color":"#111"})], md=3),
                dbc.Col([dbc.Label("Objective"),
                    dcc.Dropdown(id="dd-objective",
                        options=[{"label":"Maximize Sharpe Ratio","value":"sharpe"},
                                 {"label":"Minimize Volatility","value":"min_vol"},
                                 {"label":"Maximize Sortino Ratio","value":"sortino"}],
                        value="sharpe", clearable=False, style={"color":"#111"})], md=3),
                dbc.Col([dbc.Label("Min Weight (%)"),
                    dbc.Input(id="inp-min-weight", type="number", value=1, min=0, max=20, step=0.5)], md=2),
                dbc.Col([dbc.Label("Max Weight (%)"),
                    dbc.Input(id="inp-max-weight", type="number", value=40, min=5, max=100, step=1)], md=2),
                dbc.Col([dbc.Label("\u00a0"),
                    dbc.Button("Run Optimizer", id="btn-run-optimizer",
                               color="warning", className="w-100", n_clicks=0)], md=2),
            ], className="g-2 align-items-end"),
            dbc.Row(dbc.Col([dbc.Label("Exclude Tickers (locked at 0%)", className="mt-3"),
                dcc.Dropdown(id="dd-exclude", multi=True,
                             placeholder="Select tickers to lock at 0%...", style={"color":"#111"})])),
        ]), className="mb-3 shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Risk Mode Targets", className="card-title text-muted"),
            html.Div(id="div-risk-targets"),
        ]), className="mb-3 shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Efficient Frontier  (10 000 random portfolios + optimised spine)",
                    className="card-title text-muted"),
            dcc.Loading(dcc.Graph(id="fig-efficient-frontier", config={"displayModeBar":True},
                                  style={"height":"450px"}), type="circle"),
        ]), className="mb-3 shadow-sm"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Current Weights - Metrics", className="card-title text-muted"),
                html.Div(id="div-metrics-before")]), className="shadow-sm"), md=6),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Optimized Weights - Metrics", className="card-title text-muted"),
                html.Div(id="div-metrics-after")]), className="shadow-sm"), md=6),
        ], className="mb-3"),
        dbc.Card(dbc.CardBody([
            html.H6("Current vs Optimized Weight Allocation", className="card-title text-muted"),
            dcc.Graph(id="fig-opt-weights", config={"displayModeBar":False}, style={"height":"320px"}),
        ]), className="mb-4 shadow-sm"),
    ], fluid=True)

# ── Tab 4 — Forecasting ────────────────────────────────────────────────────────
def make_tab_forecasting():
    return dbc.Container([
        dbc.Row(dbc.Col(html.H4("Forecasting & Scenario Analysis", className="mt-3 mb-1 text-info"))),
        dbc.Card(dbc.CardBody([
            html.H6("Monte Carlo Simulation  (GARCH(1,1) vol seeding, >=50 000 paths)",
                    className="card-title text-muted mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Horizon (trading days)"),
                    dbc.Input(id="inp-horizon", type="number", value=252, min=21, max=1260, step=21)], md=3),
                dbc.Col([dbc.Label("Number of Paths"),
                    dbc.Input(id="inp-paths", type="number", value=50000, min=10000, max=200000, step=10000)], md=3),
                dbc.Col([dbc.Label("Confidence Bands"),
                    dcc.Dropdown(id="dd-conf-bands",
                        options=[{"label":"5% / 95%","value":"5_95"},
                                 {"label":"10% / 90%","value":"10_90"},
                                 {"label":"25% / 75%","value":"25_75"}],
                        value="5_95", clearable=False, style={"color":"#111"})], md=3),
                dbc.Col([dbc.Label("\u00a0"),
                    dbc.Button("\u25b6 Run Forecast", id="btn-run-forecast",
                               color="primary", className="w-100", n_clicks=0)], md=3),
            ], className="g-2 align-items-end"),
        ]), className="mb-3 shadow-sm"),
        dbc.Card(dbc.CardBody([
            html.H6("Monte Carlo Fan Chart", className="card-title text-muted"),
            dcc.Loading(dcc.Graph(id="fig-monte-carlo", config={"displayModeBar":True},
                                  style={"height":"450px"}), type="circle"),
        ]), className="mb-3 shadow-sm"),
        dbc.Row(id="row-mc-stats", className="mb-3 g-3"),
        dbc.Card(dbc.CardBody([
            html.H6("Historical Scenario Analysis", className="card-title text-muted mb-3"),
            dbc.Row([
                dbc.Col([dbc.Label("Scenario"),
                    dcc.Dropdown(id="dd-scenario",
                        options=[{"label":v[2],"value":k} for k,v in SCENARIOS.items()],
                        value="covid", clearable=False, style={"color":"#111"})], md=6),
                dbc.Col([dbc.Label("\u00a0"),
                    dbc.Button("Run Scenario", id="btn-run-scenario",
                               color="danger", outline=True, className="w-100", n_clicks=0)], md=2),
            ], className="g-2 align-items-end"),
            html.Div(id="div-scenario-result", className="mt-3"),
        ]), className="mb-3 shadow-sm"),
        dbc.Card(dbc.CardBody([
            dcc.Graph(id="fig-scenario", config={"displayModeBar":True}, style={"height":"400px"}),
            html.P("Note: replays apply current index constituents to past windows, so results carry "
                   "survivorship bias and will look more resilient than the index did at the time.",
                   className="text-muted mb-0 mt-2", style={"fontSize":"0.72rem"}),
        ]), className="mb-4 shadow-sm"),
    ], fluid=True)

# ── Main layout ────────────────────────────────────────────────────────────────
_boot_pf    = load_portfolio()
_boot_value = _boot_pf.get("total_value", DEFAULT_VALUE)
_boot_tab   = "overview" if _boot_pf.get("positions") else "builder"
_boot_rows  = [{"ticker": p["ticker"],
                "name":   p.get("name", get_display_name(p["ticker"])),
                "weight": round(float(p.get("weight", 0)), 2)}
               for p in _boot_pf.get("positions", [])]

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    dcc.Store(id="store-prices",      storage_type="memory"),
    dcc.Store(id="store-portfolio",   storage_type="memory", data=_boot_pf),
    dcc.Store(id="store-total-value", storage_type="memory", data=_boot_value),
    dcc.Store(id="store-metadata",    storage_type="memory"),
    dcc.Store(id="store-opt-result",  storage_type="memory"),
    dcc.Store(id="store-guide-step",  storage_type="memory", data=0),
    make_guide_modal(),
    make_navbar(),
    dbc.Tabs(id="main-tabs", active_tab=_boot_tab, className="nav-fill", children=[
        dbc.Tab(make_tab_builder(_boot_value, _boot_rows), label="Builder",     tab_id="builder"),
        dbc.Tab(make_tab_overview(),           label="Overview",    tab_id="overview"),
        dbc.Tab(make_tab_optimizer(),          label="Optimizer",       tab_id="optimizer"),
        dbc.Tab(make_tab_forecasting(),        label="Forecasting", tab_id="forecasting"),
    ]),
    dbc.Toast(id="toast-notification", header="Notification", is_open=False,
              dismissable=True, duration=4500,
              style={"position":"fixed","top":72,"right":20,"width":360,"zIndex":9999}),
], style={"backgroundColor":CLR_BG,"minHeight":"100vh"})


# CALLBACKS


# ── CB-01  Refresh all data ────────────────────────────────────────────────────
@app.callback(
    Output("store-prices",   "data"),
    Output("store-metadata", "data"),
    Output("toast-notification", "children"),
    Output("toast-notification", "header"),
    Output("toast-notification", "is_open"),
    Output("toast-notification", "icon"),
    Input("btn-refresh", "n_clicks"),
    State("store-portfolio", "data"),
    prevent_initial_call=True,
)
def cb_refresh_data(n, portfolio_data):
    try:
        positions = (portfolio_data or {}).get("positions", [])
        if not positions:
            return no_update, no_update, "Add positions first.", "Warning", True, "warning"
        tickers  = list({p["ticker"] for p in positions})
        closes   = bulk_fetch_closes(tickers + [BENCHMARK, RF_TICKER], period=DATA_PERIOD)
        if not closes:
            return no_update, no_update, "Failed to fetch data. Check internet.", "Error", True, "danger"
        store_prices = {}
        for t, series in closes.items():
            series = series.dropna()
            if len(series) < 2:
                continue
            store_prices[t] = {"dates":[d.strftime("%Y-%m-%d") for d in series.index],
                               "closes":[round(float(v),4) for v in series.values]}
        metadata = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_fetch_meta, t): t for t in tickers}
            for fut in as_completed(futs):
                t, meta = fut.result()
                metadata[t] = meta
        n_ok  = sum(1 for t in tickers if t in store_prices)
        n_fail = len(tickers) - n_ok
        msg = f"Loaded {n_ok}/{len(tickers)} tickers." + (f" {n_fail} had no data." if n_fail else "")
        return store_prices, metadata, msg, "Data Refreshed", True, ("success" if n_fail==0 else "warning")
    except Exception as e:
        return no_update, no_update, f"Refresh error: {e}", "Error", True, "danger"



# PORTFOLIO STATE — single-owner architecture


def _trigger_id():
    """Robust 'which input fired' across all Dash versions."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    return ctx.triggered[0]["prop_id"].split(".")[0]


# ── CB-02  Position reducer — owns positions-table.data (Add / Normalize / Clear)
@app.callback(
    Output("positions-table",    "data",     allow_duplicate=True),
    Output("alert-add-position", "children"),
    Input("btn-add-position", "n_clicks"),
    Input("btn-normalize",    "n_clicks"),
    Input("btn-clear-all",    "n_clicks"),
    State("dd-ticker",       "value"),
    State("inp-weight",      "value"),
    State("positions-table", "data"),
    prevent_initial_call=True,
)
def cb_position_reducer(add_n, norm_n, clear_n, ticker, weight, rows):
    rows = list(rows or [])
    trig = _trigger_id()

    if trig == "btn-clear-all":
        return [], None

    if trig == "btn-normalize":
        valid = [r for r in rows if float(r.get("weight", 0) or 0) > 0]
        tot = sum(float(r["weight"]) for r in valid)
        if tot <= 0:
            return no_update, dbc.Alert("Nothing to normalize.", color="warning",
                                        dismissable=True, duration=2500)
        for r in rows:
            w = float(r.get("weight", 0) or 0)
            r["weight"] = round(w / tot * 100, 2)
        return rows, None

    if trig == "btn-add-position":
        try:
            if not ticker:
                return no_update, dbc.Alert("Please select a stock.", color="warning", dismissable=True)
            if weight in (None, ""):
                return no_update, dbc.Alert("Enter a weight (%) for this stock.",
                                            color="warning", dismissable=True)
            try:
                w = float(weight)
            except (TypeError, ValueError):
                return no_update, dbc.Alert("Weight must be a number.", color="warning", dismissable=True)
            if w <= 0:
                return no_update, dbc.Alert("Weight must be greater than 0.", color="warning", dismissable=True)
            if w > 100:
                return no_update, dbc.Alert("Weight cannot exceed 100%.", color="warning", dismissable=True)
            if ticker not in SP500_TOP200:
                return no_update, dbc.Alert(f"{ticker} not in supported list.", color="warning", dismissable=True)
            if any(r.get("ticker") == ticker for r in rows):
                return no_update, dbc.Alert(f"{ticker} already added. Edit its row to change the weight.",
                                            color="info", dismissable=True)
            rows.append({"ticker": ticker, "name": get_display_name(ticker), "weight": round(w, 2)})
            return rows, dbc.Alert(f"Added {ticker} at {w:.1f}% weight", color="success",
                                   dismissable=True, duration=2500)
        except Exception as e:
            return no_update, dbc.Alert(f"Error: {e}", color="danger", dismissable=True)

    return no_update, no_update


# ── CB-03  Mirror — owns store-portfolio.data (one-way projection of the table) ─
@app.callback(
    Output("store-portfolio",  "data"),
    Output("div-no-positions", "children"),
    Input("positions-table",   "data"),
    Input("store-total-value", "data"),
)
def cb_mirror_store(table_data, total_value):
    positions = _parse_table_data(table_data)
    pf = {"total_value": float(total_value or DEFAULT_VALUE), "positions": positions}
    msg = "" if positions else html.P(
        "No positions yet. Pick a stock and weight above to get started.",
        className="text-muted small mt-2 mb-0")
    return pf, msg


# ── CB-04  Live total-weight indicator ────────────────────────────────────────
@app.callback(
    Output("div-total-weight", "children"),
    Input("positions-table",   "data"),
)
def cb_total_weight(table_data):
    total = sum(float(r.get("weight", 0) or 0) for r in (table_data or []))
    if total == 0:
        return html.Span("Total: 0%", style={"color":"#888","fontWeight":"bold"})
    if abs(total - 100) < 0.5:
        return html.Span(f"\u2713 Total: {total:.1f}%", style={"color":CLR_GREEN,"fontWeight":"bold"})
    return html.Span(f"\u26a0 Total: {total:.1f}%  (should be 100%)",
                     style={"color":CLR_YELLOW,"fontWeight":"bold"})


# ── CB-07  Track total portfolio value ────────────────────────────────────────
@app.callback(
    Output("store-total-value", "data"),
    Input("inp-total-value",    "value"),
    prevent_initial_call=True,
)
def cb_total_value(value):
    try:
        v = float(value)
        return v if v > 0 else DEFAULT_VALUE
    except (ValueError, TypeError):
        return DEFAULT_VALUE


# ── CB-08  Save portfolio (writes portfolio.json; store is owned by the mirror) ─
@app.callback(
    Output("toast-notification", "children", allow_duplicate=True),
    Output("toast-notification", "header",   allow_duplicate=True),
    Output("toast-notification", "is_open",  allow_duplicate=True),
    Output("toast-notification", "icon",     allow_duplicate=True),
    Input("btn-save",          "n_clicks"),
    State("positions-table",   "data"),
    State("store-total-value", "data"),
    prevent_initial_call=True,
)
def cb_save_portfolio(n, table_data, total_value):
    try:
        positions = _parse_table_data(table_data)
        if not positions:
            return "Nothing to save.", "Warning", True, "warning"
        tv  = float(total_value or DEFAULT_VALUE)
        err = save_portfolio(positions, tv)
        if err:
            return f"Save failed: {err}", "Error", True, "danger"
        return f"Saved {len(positions)} position(s) to {PORTFOLIO_FILE}", "Saved", True, "success"
    except Exception as e:
        return f"Save error: {e}", "Error", True, "danger"


# ── CB-09  Launch dashboard (persists + switches tab; store owned by mirror) ────
@app.callback(
    Output("main-tabs",           "active_tab"),
    Output("div-launch-feedback", "children"),
    Input("btn-launch",        "n_clicks"),
    State("positions-table",   "data"),
    State("store-total-value", "data"),
    prevent_initial_call=True,
)
def cb_launch_dashboard(n, table_data, total_value):
    try:
        positions = _parse_table_data(table_data)
        if not positions:
            return no_update, dbc.Alert("Add at least one stock before launching.", color="warning")
        tv  = float(total_value or DEFAULT_VALUE)
        err = save_portfolio(positions, tv)
        if err:
            return no_update, dbc.Alert(f"Could not save: {err}", color="danger")
        return "overview", dbc.Alert(
            f"Dashboard launched with {len(positions)} position(s). "
            f"Go to Overview - and click 'Refresh All Data' to load prices.", color="success")
    except Exception as e:
        return no_update, dbc.Alert(f"Launch error: {e}", color="danger")


# ── CB-10  Overview — summary metric cards ────────────────────────────────────
@app.callback(
    Output("row-summary-metrics", "children"),
    Input("store-prices",       "data"),
    Input("store-portfolio",    "data"),
    Input("store-total-value",  "data"),
    Input("radio-ov-window",    "value"),
)
def cb_overview_metrics(prices, portfolio_data, total_value, window):
    try:
        window  = window or "3Y"
        weights = _get_weights(portfolio_data)
        if not weights:
            return _metric_card("Status", "No Positions", CLR_YELLOW)
        tv = float(total_value or DEFAULT_VALUE)
        if not prices:
            return [_metric_card("Portfolio Value", f"${tv:,.0f}", CLR_TEXT),
                    _metric_card("Status", "Click Refresh", CLR_YELLOW),
                    _metric_card("Positions", f"{len(weights)}", CLR_BLUE)]
        price_dict = _series_dict(prices, weights)
        port_rets  = compute_portfolio_returns(weights, price_dict)
        if port_rets.empty or len(port_rets) < 2:
            return [_metric_card("Portfolio Value", f"${tv:,.0f}", CLR_TEXT),
                    _metric_card("Status", "No Price Data", CLR_YELLOW),
                    _metric_card("Positions", f"{len(weights)}", CLR_BLUE)]
        port_rets = _slice_window(port_rets, window)
        cum_ret  = float((1 + port_rets).prod() - 1)
        day_chg  = float(port_rets.iloc[-1])
        ann_vol  = float(port_rets.std() * np.sqrt(252))
        cur_val  = tv * (1 + cum_ret)
        sgn  = "+" if cum_ret >= 0 else ""
        dsgn = "+" if day_chg >= 0 else ""
        return [
            _metric_card("Portfolio Value", f"${cur_val:,.0f}", CLR_TEXT,
                         delta=f"{sgn}{cum_ret*100:.2f}% ({window} backtest)"),
            _metric_card("Day Change",  f"{dsgn}{day_chg*100:.2f}%",
                         CLR_GREEN if day_chg >= 0 else CLR_RED),
            _metric_card(f"Volatility ({window})", f"{ann_vol*100:.2f}%",
                         CLR_RED if ann_vol > 0.25 else (CLR_YELLOW if ann_vol > 0.15 else CLR_GREEN)),
            _metric_card("Positions", f"{len(weights)}", CLR_BLUE),
        ]
    except Exception as e:
        return _metric_card("Error", str(e)[:30], CLR_RED)


# ── CB-11  Overview — donut charts ────────────────────────────────────────────
@app.callback(
    Output("fig-weight-donut",  "figure"),
    Output("fig-sector-donut",  "figure"),
    Output("fig-country-donut", "figure"),
    Input("store-portfolio", "data"),
    Input("store-metadata",  "data"),
)
def cb_overview_donuts(portfolio_data, metadata):
    def _donut(labels, values, title):
        if not labels or sum(values) == 0:
            return _empty_fig(f"No {title} data")
        fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45,
            textinfo="label+percent", textfont=dict(color="white", size=10),
            marker=dict(line=dict(color="#1e1e1e", width=1.5))))
        _dark_layout(fig, title)
        fig.update_layout(margin=dict(t=30,b=10,l=10,r=10), showlegend=False)
        return fig
    try:
        weights = _get_weights(portfolio_data)
        if not weights:
            e = _empty_fig("No positions"); return e, e, e
        wl = list(weights.keys())
        fig_w = _donut(wl, [weights[t]*100 for t in wl], "Portfolio Weights (%)")
        sec = {}
        for t, w in weights.items():
            s = (metadata or {}).get(t, {}).get("sector", "Unknown") or "Unknown"
            sec[s] = sec.get(s, 0) + w*100
        fig_s = _donut(list(sec), list(sec.values()), "Sector Breakdown (%)")
        co = {}
        for t, w in weights.items():
            c = (metadata or {}).get(t, {}).get("country", "Unknown") or "Unknown"
            co[c] = co.get(c, 0) + w*100
        fig_c = _donut(list(co), list(co.values()), "Country Split (%)")
        return fig_w, fig_s, fig_c
    except Exception as e:
        err = _empty_fig(f"Error: {e}"); return err, err, err


# ── CB-12  Overview — cumulative return ───────────────────────────────────────
@app.callback(
    Output("fig-cumulative-return", "figure"),
    Input("store-prices",    "data"),
    Input("store-portfolio", "data"),
    Input("radio-ov-window", "value"),
)
def cb_cumulative_return(prices, portfolio_data, window):
    try:
        window = window or "3Y"
        if not prices:
            return _empty_fig("Click 'Refresh All Data' to load prices")
        weights = _get_weights(portfolio_data)
        if not weights:
            return _empty_fig("Add positions first")
        price_dict = _series_dict(prices, weights)
        port_rets = compute_portfolio_returns(weights, price_dict)
        if port_rets.empty or len(port_rets) < 5:
            return _empty_fig("Insufficient aligned price history")
        port_rets = _slice_window(port_rets, window)
        port_cum = (1 + port_rets).cumprod(); port_cum = port_cum / port_cum.iloc[0] * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode="lines",
            name="Portfolio", line=dict(color=CLR_GREEN, width=2.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Portfolio</extra>"))
        bench_s = _prices_to_series(prices, BENCHMARK)
        if bench_s is not None:
            br = bench_s.pct_change().dropna()
            aligned = pd.concat([port_rets, br], axis=1).dropna()
            if len(aligned) > 5:
                bc = (1 + aligned.iloc[:, 1]).cumprod(); bc = bc / bc.iloc[0] * 100
                fig.add_trace(go.Scatter(x=bc.index, y=bc.values, mode="lines",
                    name=BENCH_LABEL, line=dict(color=CLR_BLUE, width=2, dash="dash"),
                    hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>" + BENCH_LABEL + "</extra>"))
        fig.add_hline(y=100, line_dash="dot", line_color="#666")
        _dark_layout(fig, f"Cumulative Return - {OV_LABELS.get(window, window)} vs {BENCH_LABEL} (Normalised to 100)")
        fig.update_yaxes(title_text="Index (100 = start)")
        fig.update_xaxes(title_text="Date")
        return fig
    except Exception as e:
        return _empty_fig(f"Error: {e}")


# ── CB-13  Overview — position detail ─────────────────────────────────────────
@app.callback(
    Output("div-position-detail", "children"),
    Input("store-prices",       "data"),
    Input("store-portfolio",    "data"),
    Input("store-total-value",  "data"),
)
def cb_position_detail(prices, portfolio_data, total_value):
    try:
        positions = (portfolio_data or {}).get("positions", [])
        if not positions:
            return html.P("No positions.", className="text-muted small")
        weights = _get_weights(portfolio_data)
        tv = float(total_value or DEFAULT_VALUE)
        if not prices:
            return _no_data_alert("Click 'Refresh All Data' to load position details.")
        def _c(v): return CLR_GREEN if v >= 0 else CLR_RED
        def _ret_cell(closes, n):
            """Trailing return over the last n trading days, or None if history is short."""
            if len(closes) <= n:
                return None
            base = float(closes[-(n + 1)])
            return (float(closes[-1]) - base) / base * 100 if base > 0 else None
        def _fmt_ret(v):
            return html.Td("-", style={"color":"#666"}) if v is None else \
                   html.Td(f"{v:+.1f}%", style={"color":_c(v)})
        # Benchmark daily returns, computed once for the per-position betas.
        bench_s = _prices_to_series(prices, BENCHMARK)
        bench_r = bench_s.pct_change().dropna() if bench_s is not None else None
        N_COLS = 12
        rows = []
        for p in positions:
            t = p["ticker"]; w = weights.get(t, 0.0); alloc = w * tv
            if t not in prices or not prices[t]["closes"]:
                rows.append(html.Tr(
                    [html.Td(t, style={"fontWeight":"bold","fontFamily":"monospace","color":CLR_BLUE}),
                     html.Td(p.get("name",""), style={"color":"#aaa","fontSize":"11px"}),
                     html.Td(f"{w*100:.2f}%"), html.Td(f"${alloc:,.0f}")]
                    + [html.Td("-") for _ in range(N_COLS - 4)]))
                continue
            cls  = prices[t]["closes"]; curr = float(cls[-1])
            prev = float(cls[-2]) if len(cls) >= 2 else curr
            first = float(cls[0])
            day_c = (curr - prev) / prev * 100 if prev > 0 else 0.0
            r_1m, r_6m, r_1y = _ret_cell(cls, 21), _ret_cell(cls, 126), _ret_cell(cls, 252)
            r_3y = (curr - first) / first * 100 if first > 0 else None
            # Per-position risk over the full 36-month window
            s = _prices_to_series(prices, t)
            tr = s.pct_change().dropna()
            vol = float(tr.std() * np.sqrt(252)) * 100 if len(tr) > 20 else None
            beta_txt = "-"
            if bench_r is not None and len(tr) > 20:
                al = pd.concat([tr, bench_r], axis=1).dropna()
                if len(al) > 20:
                    vb = float(np.var(al.iloc[:, 1]))
                    if vb > 1e-12:
                        b_raw = float(np.cov(al.iloc[:, 0], al.iloc[:, 1])[0, 1] / vb)
                        beta_txt = f"{0.67*b_raw + 0.33:.2f}"     # Blume-adjusted
            rows.append(html.Tr([
                html.Td(t, style={"fontWeight":"bold","fontFamily":"monospace","color":CLR_BLUE}),
                html.Td(p.get("name",""), style={"color":"#bbb","fontSize":"11px","maxWidth":"160px",
                        "overflow":"hidden","textOverflow":"ellipsis"}),
                html.Td(f"{w*100:.2f}%"),
                html.Td(f"${alloc:,.0f}"),
                html.Td(f"${curr:.2f}"),
                html.Td(f"{day_c:+.2f}%", style={"color":_c(day_c)}),
                _fmt_ret(r_1m), _fmt_ret(r_6m), _fmt_ret(r_1y), _fmt_ret(r_3y),
                html.Td("-" if vol is None else f"{vol:.1f}%",
                        style={"color": CLR_RED if (vol or 0) > 35 else CLR_TEXT}),
                html.Td(beta_txt),
            ]))
        header = html.Tr([html.Th(h, style={"color":"#aaa","fontSize":"11px","padding":"4px 8px",
                          "borderBottom":f"1px solid {CLR_BORDER}","whiteSpace":"nowrap"})
                          for h in ["Ticker","Company","Weight (%)","Allocation ($)","Price",
                                    "Day (%)","1M (%)","6M (%)","1Y (%)","3Y (%)",
                                    "Vol 3Y (ann.)","Beta 3Y"]])
        note = html.P("Trailing total returns from dividend-adjusted closes. Volatility is annualized "
                      f"from 36 months of daily returns. Beta is Blume-adjusted versus {BENCH_LABEL}.",
                      className="text-muted mb-0 mt-2", style={"fontSize":"0.68rem"})
        return html.Div([dbc.Table([html.Thead(header), html.Tbody(rows)],
                         bordered=False, hover=True, size="sm", responsive=True,
                         style={"fontSize":"12px","marginBottom":0}), note])
    except Exception as e:
        return _no_data_alert(f"Error: {e}")


# ── CB-14  Optimizer — exclude dropdown options ───────────────────────────────
@app.callback(
    Output("dd-exclude",     "options"),
    Input("store-portfolio", "data"),
)
def cb_populate_exclude_dd(portfolio_data):
    positions = (portfolio_data or {}).get("positions", [])
    return [{"label": f"{p['ticker']}  -  {get_display_name(p['ticker'])}", "value": p["ticker"]}
            for p in positions]


# ── CB-14b  Start guide navigation ─────────────────────────────────────────────
@app.callback(
    Output("modal-guide",      "is_open"),
    Output("store-guide-step", "data"),
    Output("guide-title",      "children"),
    Output("guide-body",       "children"),
    Output("guide-progress",   "children"),
    Output("btn-guide-back",   "disabled"),
    Output("btn-guide-next",   "children"),
    Input("btn-guide",         "n_clicks"),
    Input("btn-guide-back",    "n_clicks"),
    Input("btn-guide-next",    "n_clicks"),
    State("store-guide-step",  "data"),
    prevent_initial_call=True,
)
def cb_guide_nav(n_open, n_back, n_next, step):
    trig = _trigger_id()
    step = int(step or 0)
    last = len(GUIDE_STEPS) - 1
    if trig == "btn-guide":
        step = 0
    elif trig == "btn-guide-back":
        step = max(0, step - 1)
    elif trig == "btn-guide-next":
        if step >= last:                       # "Finish" on the last step
            return False, 0, no_update, no_update, no_update, no_update, no_update
        step += 1
    title, body = GUIDE_STEPS[step]
    progress = f"Step {step + 1} of {len(GUIDE_STEPS)}"
    next_label = "Finish" if step == last else "Next"
    return True, step, title, body, progress, step == 0, next_label


# ── CB-15  Optimizer — risk targets display ───────────────────────────────────
@app.callback(
    Output("div-risk-targets", "children"),
    Input("dd-risk-mode",      "value"),
)
def cb_risk_targets(risk_mode):
    if risk_mode not in RISK_MODES:
        return _no_data_alert("Unknown risk mode.")
    t = RISK_MODES[risk_mode]
    items = [("Target Beta", f"{t['beta']:.2f}"), ("Min Sharpe", f"{t['sharpe']:.2f}"),
             ("Max Drawdown", f"{t['max_dd']*100:.0f}%"), ("Vol Target", f"{t['vol']*100:.0f}%"),
             ("Min Calmar", f"{t['calmar']:.2f}"), ("VaR 95%", f"{t['var95']*100:.1f}%")]
    cols = [dbc.Col(dbc.Card(dbc.CardBody([
        html.P(lab, className="mb-0 text-muted", style={"fontSize":"0.7rem","textTransform":"uppercase"}),
        html.H5(val, style={"color":CLR_BLUE,"fontWeight":"700","margin":"3px 0 0"}),
    ]), style={"backgroundColor":"#1a1a2e","border":"1px solid #333"}), md=2, sm=4, xs=6)
            for lab, val in items]
    return dbc.Row(cols, className="g-2")

# ── CB-16  Optimizer — run efficient frontier + optimization ──────────────────
@app.callback(
    Output("fig-efficient-frontier", "figure"),
    Output("fig-opt-weights",        "figure"),
    Output("div-metrics-before",     "children"),
    Output("div-metrics-after",      "children"),
    Output("store-opt-result",       "data"),
    Input("btn-run-optimizer", "n_clicks"),
    State("dd-risk-mode",   "value"),
    State("dd-objective",   "value"),
    State("inp-min-weight", "value"),
    State("inp-max-weight", "value"),
    State("dd-exclude",     "value"),
    State("store-prices",   "data"),
    State("store-portfolio","data"),
    prevent_initial_call=True,
)
def cb_run_optimizer(n, risk_mode, objective, min_w_pct, max_w_pct, exclude, prices, portfolio_data):
    _ef, _nd = _empty_fig, _no_data_alert
    try:
        if not prices or not portfolio_data:
            m = _nd("Refresh data first, then run optimizer.")
            return _ef("No data"), _ef("No data"), m, m, no_update
        positions = (portfolio_data or {}).get("positions", [])
        if len(positions) < 2:
            m = _nd("Need at least 2 positions to optimize.")
            return _ef("Need >=2 positions"), _ef("Need >=2 positions"), m, m, no_update
        price_dict = {}
        for p in positions:
            s = _prices_to_series(prices, p["ticker"])
            if s is not None and len(s) > 30:
                price_dict[p["ticker"]] = s
        if len(price_dict) < 2:
            m = _nd("Need price data for >=2 positions. Click Refresh All Data.")
            return _ef("Insufficient data"), _ef("Insufficient data"), m, m, no_update
        prices_df  = pd.DataFrame(price_dict).dropna()
        returns_df = prices_df.pct_change().dropna().dropna(axis=1)
        if returns_df.shape[1] < 2 or returns_df.shape[0] < 30:
            m = _nd("Insufficient return history after alignment.")
            return _ef("Insufficient data"), _ef("Insufficient data"), m, m, no_update
        tickers  = list(returns_df.columns)
        R        = returns_df.values
        n_assets = len(tickers)
        rf_ann   = _get_rf_annual(prices)
        lo_b = max(0.0, float(min_w_pct or 1) / 100.0)
        hi_b = min(1.0, float(max_w_pct or 40) / 100.0)
        excl = set(exclude or [])
        bounds = [(0.0, 0.0) if t in excl else (lo_b, hi_b) for t in tickers]
        n_active = sum(1 for lo, hi in bounds if hi > 0)
        if n_active < 2:
            m = _nd("At least 2 non-excluded tickers required.")
            return _ef("Too few assets"), _ef("Too few assets"), m, m, no_update
        if sum(lo for lo, hi in bounds if hi > 0) > 1.0:
            adj = 1.0 / n_active * 0.5
            bounds = [(0.0, 0.0) if hi == 0 else (adj, hi_b) for lo, hi in bounds]
        x0 = np.array([1.0/n_active if b[1] > 0 else 0.0 for b in bounds])
        cons = [{"type":"eq","fun": lambda w: float(w.sum()) - 1.0}]
        # --- Tier-1 robust estimation -------------------------------------------------
        # Ledoit-Wolf shrunk daily covariance + James-Stein shrunk daily means.
        # These stabilise mean-variance optimisation against estimation error.
        Sigma  = shrunk_cov(R)                       # (n_assets, n_assets) daily cov
        mu_vec = shrink_mean(R.mean(axis=0), 0.5)    # (n_assets,) shrunk daily means
        def _ann_stats(w):
            dr = float(mu_vec @ w)
            dv = float(w @ Sigma @ w)
            return (1 + dr) ** 252 - 1, np.sqrt(max(dv, 1e-12) * 252)
        def neg_sharpe(w):
            r, v = _ann_stats(w)
            return -(r - rf_ann)/max(v, 1e-8)
        def min_vol(w):
            return _ann_stats(w)[1]
        def neg_sortino(w):
            r, _ = _ann_stats(w)
            pr = R @ w; d = pr[pr < 0]
            ds = d.std()*np.sqrt(252) if len(d) > 1 else 1e-8
            return -(r - rf_ann)/max(ds, 1e-8)
        obj_fn = {"sharpe":neg_sharpe,"min_vol":min_vol,"sortino":neg_sortino}.get(str(objective or "sharpe"), neg_sharpe)
        res = minimize(obj_fn, x0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter":1000,"ftol":1e-9})
        opt_w = res.x if res.success else x0
        opt_w = np.maximum(opt_w, 0.0); opt_w /= opt_w.sum()
        n_rand = 10_000
        W_rand = np.random.dirichlet(np.ones(n_assets), size=n_rand)
        for i,(lo,hi) in enumerate(bounds):
            W_rand[:, i] = np.clip(W_rand[:, i], lo, hi)
        rs = W_rand.sum(axis=1, keepdims=True); rs[rs<1e-8] = 1.0; W_rand /= rs
        # Cloud stats from the same shrunk estimates the optimiser uses (consistency)
        ann_rets_r = (1 + (W_rand @ mu_vec)) ** 252 - 1
        ann_vols_r = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", W_rand, Sigma, W_rand), 1e-12) * 252)
        sharpes_r  = (ann_rets_r - rf_ann)/np.maximum(ann_vols_r, 1e-8)
        # Current weights from user portfolio (normalized)
        uw = _get_weights(portfolio_data)
        curr_w = np.array([uw.get(t, 0.0) for t in tickers])
        if curr_w.sum() > 0:
            curr_w /= curr_w.sum()
        else:
            curr_w = np.ones(n_assets)/n_assets
        bench_rets = None
        bs = _prices_to_series(prices, BENCHMARK)
        if bs is not None:
            br = bs.pct_change().dropna().reindex(returns_df.index).dropna()
            if len(br) >= 10:
                bench_rets = br
        curr_pr = pd.Series(R @ curr_w, index=returns_df.index)
        opt_pr  = pd.Series(R @ opt_w,  index=returns_df.index)
        curr_m  = compute_metrics(curr_pr, bench_rets, rf_ann)
        opt_m   = compute_metrics(opt_pr,  bench_rets, rf_ann)
        targets = np.linspace(float(np.percentile(ann_rets_r,10)), float(np.percentile(ann_rets_r,90)), 50)
        sv, sr = [], []
        for tr in targets:
            try:
                cs = [{"type":"eq","fun": lambda w: float(w.sum())-1.0},
                      {"type":"ineq","fun": lambda w, tr=tr: float((1+(mu_vec @ w))**252 - 1) - tr}]
                r2 = minimize(min_vol, x0, method="SLSQP", bounds=bounds, constraints=cs,
                              options={"maxiter":200,"ftol":1e-6})
                if r2.success and r2.fun > 0:
                    sv.append(float(r2.fun)*100); sr.append(float(tr)*100)
            except Exception:
                pass
        if sv:
            pairs = sorted(zip(sv, sr)); sv, sr = zip(*pairs)
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=ann_vols_r*100, y=ann_rets_r*100, mode="markers",
            marker=dict(color=sharpes_r, colorscale="Plasma", size=3, opacity=0.45,
                        colorbar=dict(title="Sharpe", thickness=12, len=0.7), showscale=True),
            name="Random Portfolios",
            hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<br>Sharpe: %{marker.color:.3f}<extra></extra>"))
        if sv:
            fig_ef.add_trace(go.Scatter(x=list(sv), y=list(sr), mode="lines",
                line=dict(color=CLR_YELLOW, width=2.5), name="Efficient Frontier",
                hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>"))
        def _pt(w, lbl, col):
            ar, av = _ann_stats(w); ar *= 100; av *= 100
            fig_ef.add_trace(go.Scatter(x=[av], y=[ar], mode="markers",
                marker=dict(size=16, symbol="star", color=col, line=dict(width=1, color="white")),
                name=lbl, hovertemplate=f"{lbl}<br>Vol: {av:.2f}%<br>Ret: {ar:.2f}%<extra></extra>"))
        _pt(curr_w, "Current Portfolio", CLR_YELLOW)
        _pt(opt_w,  "Optimal Portfolio", CLR_GREEN)
        _dark_layout(fig_ef, "Risk / Return - Efficient Frontier (Ledoit-Wolf cov + shrunk means)")
        fig_ef.update_xaxes(title_text="Annualised Volatility (%)")
        fig_ef.update_yaxes(title_text="Annualised Return (%, arithmetic MV convention)")
        fig_w = go.Figure([
            go.Bar(name="Current",   x=tickers, y=(curr_w*100).round(2),
                   marker_color=CLR_YELLOW, opacity=0.75,
                   hovertemplate="%{x}: %{y:.2f}%<extra>Current</extra>"),
            go.Bar(name="Optimized", x=tickers, y=(opt_w*100).round(2),
                   marker_color=CLR_GREEN, opacity=0.85,
                   hovertemplate="%{x}: %{y:.2f}%<extra>Optimized</extra>"),
        ])
        fig_w.update_layout(barmode="group")
        _dark_layout(fig_w, "Current vs Optimized Weights (%)")
        fig_w.update_yaxes(title_text="Weight (%)")
        opt_result = {t: round(float(w), 6) for t, w in zip(tickers, opt_w)}
        return fig_ef, fig_w, _make_metrics_table(curr_m), _make_metrics_table(opt_m), opt_result
    except Exception as e:
        m = dbc.Alert(f"Optimizer error: {e}", color="danger")
        return _ef(f"Error: {e}"), _ef("Error"), m, m, no_update


# ── CB-17  Forecasting — Monte Carlo ──────────────────────────────────────────
@app.callback(
    Output("fig-monte-carlo", "figure"),
    Output("row-mc-stats",    "children"),
    Input("btn-run-forecast", "n_clicks"),
    State("inp-horizon",      "value"),
    State("inp-paths",        "value"),
    State("dd-conf-bands",    "value"),
    State("store-prices",     "data"),
    State("store-portfolio",  "data"),
    State("store-total-value","data"),
    State("store-opt-result", "data"),
    prevent_initial_call=True,
)
def cb_run_forecast(n, horizon, n_paths, conf_band, prices, portfolio_data, total_value, opt_result):
    try:
        if not prices or not portfolio_data:
            return _empty_fig("Load data and set positions first."), []
        positions = (portfolio_data or {}).get("positions", [])
        if not positions:
            return _empty_fig("No positions."), []
        horizon = max(21, min(int(horizon or 252), 1260))
        n_paths = max(10000, min(int(n_paths or 50000), 200000))
        # --- Memory budget: cap the paths matrix to ~720 MB of float32 so the
        # forecast never OOM-crashes the app at the high end (e.g. 200k x 1260).
        # The default 50k x 252 is unaffected; only extreme settings auto-reduce.
        MEM_BUDGET_ELEMS = 180_000_000          # ~720 MB as float32
        eff_paths = min(n_paths, max(10000, MEM_BUDGET_ELEMS // (horizon + 1)))
        capped = eff_paths < n_paths
        lo_p, hi_p = {"5_95":(5,95),"10_90":(10,90),"25_75":(25,75)}.get(conf_band or "5_95", (5,95))
        tv = float(total_value or DEFAULT_VALUE)
        # Choose weights: optimized if present, else current user weights
        weights = None
        if opt_result and isinstance(opt_result, dict) and len(opt_result) > 0:
            avail = [t for t in opt_result if prices and t in prices and opt_result[t] > 0]
            if avail:
                tot = sum(opt_result[t] for t in avail)
                weights = {t: opt_result[t]/tot for t in avail} if tot > 0 else None
        if weights is None:
            uw = _get_weights(portfolio_data)
            avail = [t for t in uw if prices and t in prices]
            if not avail:
                return _empty_fig("No price data for portfolio tickers."), []
            tot = sum(uw[t] for t in avail)
            weights = {t: uw[t]/tot for t in avail} if tot > 0 else {t: 1.0/len(avail) for t in avail}
        price_dict = _series_dict(prices, weights)
        port_rets = compute_portfolio_returns(weights, price_dict)
        if port_rets.empty or len(port_rets) < 30:
            return _empty_fig("Insufficient return history for simulation."), []
        # Drift is anchored to the GEOMETRIC annualized return so the median
        # simulated path compounds at the same rate the Overview tab reports —
        # one growth convention across the whole app (the Ito-consistent one).
        n_obs   = len(port_rets)
        geo_ann = float((1 + port_rets).prod() ** (252.0 / n_obs) - 1)
        m_log   = np.log1p(geo_ann) / 252.0            # daily log drift
        garch_p   = fit_garch_params(port_rets)         # None -> constant-vol fallback
        sigma_ann = fit_garch_vol(port_rets)            # for display / fallback vol
        diffus  = sigma_ann * np.sqrt(1.0/252.0)
        rng = np.random.default_rng()
        _shock_label = f"Student-t({T_DOF}), unit variance"
        _t_scale = np.float32(np.sqrt((T_DOF - 2) / T_DOF))   # var of raw t(v) is v/(v-2)
        m_log_f, diffus_f = np.float32(m_log), np.float32(diffus)
        _vol_label = "constant vol (GBM)"
        if garch_p is not None:
            g_omega, g_alpha, g_beta, g_s2_0 = (np.float32(v) for v in garch_p)
            lr_var = garch_p[0] / (1 - garch_p[1] - garch_p[2])   # long-run daily var
            _vol_label = (f"GARCH(1,1) path: now {np.sqrt(garch_p[3]*252)*100:.1f}%, "
                          f"long-run {np.sqrt(lr_var*252)*100:.1f}%/yr")
        # Allocate the path matrix once and fill it in path-chunks so the random/
        # intermediate buffers stay tiny (peak temp = CHUNK x horizon, not
        # n_paths x horizon). This keeps peak memory ~= the paths array alone.
        paths = np.empty((eff_paths, horizon + 1), dtype=np.float32)
        paths[:, 0] = 100.0
        CHUNK = 20000
        for s in range(0, eff_paths, CHUNK):
            e = min(s + CHUNK, eff_paths)
            rows = e - s
            try:
                z = (rng.standard_t(T_DOF, size=(rows, horizon)).astype(np.float32)
                     * _t_scale)                        # standardized: unit variance
            except Exception:
                z = rng.standard_normal((rows, horizon)).astype(np.float32)
                _shock_label = "Normal"
            if garch_p is not None:
                # Simulate the conditional-variance path per step:
                #   r_t = m + sigma_t * z_t
                #   s2_{t+1} = omega + alpha * (sigma_t z_t)^2 + beta * s2_t
                # Log prices are accumulated into `paths` and exponentiated once.
                s2  = np.full(rows, g_s2_0, dtype=np.float32)
                cum = np.zeros(rows, dtype=np.float32)
                for t_i in range(horizon):
                    sig = np.sqrt(s2)
                    eps = sig * z[:, t_i]
                    cum += m_log_f + eps
                    paths[s:e, t_i + 1] = cum
                    s2 = g_omega + g_alpha * (eps * eps) + g_beta * s2
                np.exp(paths[s:e, 1:], out=paths[s:e, 1:])
                paths[s:e, 1:] *= 100.0
            else:
                z *= diffus_f
                z += m_log_f
                np.cumsum(z, axis=1, out=z)
                np.exp(z, out=z)
                paths[s:e, 1:] = z
                paths[s:e, 1:] *= 100.0
            del z
        # Exact percentiles, computed in time-blocks so np.percentile's internal
        # sort copy is bounded to a slice rather than the whole matrix.
        bands = np.empty((5, horizon + 1), dtype=np.float64)
        qs = [lo_p, 25, 50, 75, hi_p]
        TBLK = 256
        for c in range(0, horizon + 1, TBLK):
            d = min(c + TBLK, horizon + 1)
            bands[:, c:d] = np.percentile(paths[:, c:d], qs, axis=0)
        p_lo, p_25, p_med, p_75, p_hi = bands
        today = datetime.today()
        dates = [today + timedelta(days=int(i*365.0/252)) for i in range(horizon+1)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=p_hi.tolist(), mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=dates, y=p_lo.tolist(), mode="lines", fill="tonexty",
                                 fillcolor="rgba(41,121,255,0.12)", line=dict(width=0),
                                 name=f"{lo_p}-{hi_p}% Band"))
        if lo_p != 25:
            fig.add_trace(go.Scatter(x=dates, y=p_75.tolist(), mode="lines",
                                     line=dict(width=0), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dates, y=p_25.tolist(), mode="lines", fill="tonexty",
                                     fillcolor="rgba(41,121,255,0.25)", line=dict(width=0),
                                     name="25-75% Band"))
        fig.add_trace(go.Scatter(x=dates, y=p_med.tolist(), mode="lines",
                                 line=dict(color=CLR_GREEN, width=2.5), name="Median",
                                 hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Median</extra>"))
        for i in rng.choice(eff_paths, min(20, eff_paths), replace=False):
            fig.add_trace(go.Scatter(x=dates, y=paths[i].tolist(), mode="lines",
                                     line=dict(color="rgba(255,255,255,0.06)", width=0.5),
                                     showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=100, line_dash="dot", line_color="#666",
                      annotation_text="Start (100)", annotation_position="bottom right")
        _cap_note = f"  |  paths: {eff_paths:,}" + (" (capped for memory)" if capped else "")
        _dark_layout(fig, f"Monte Carlo - {horizon} Trading Days  |  {_vol_label}  |  "
                     f"geometric mu={geo_ann*100:.1f}%/yr  |  "
                     f"shocks: {_shock_label}{_cap_note}")
        fig.update_yaxes(title_text="Portfolio Index (Start = 100)")
        fig.update_xaxes(title_text="Date")
        fig.update_layout(hovermode="x unified")
        final_rets  = paths[:, -1]/100.0 - 1.0
        prob_profit = float((final_rets > 0).mean())*100
        def _fc(v): return CLR_GREEN if v >= 0 else CLR_RED
        exp_ret = float(np.mean(final_rets))*100
        med_ret = float(np.median(final_rets))*100
        bull = float(np.percentile(final_rets, hi_p))*100
        bear = float(np.percentile(final_rets, lo_p))*100
        stats = [
            _metric_card("Expected Return", f"{exp_ret:+.2f}%", _fc(exp_ret)),
            _metric_card("Median Return",   f"{med_ret:+.2f}%", _fc(med_ret)),
            _metric_card(f"Bull ({hi_p}th)", f"{bull:+.2f}%", CLR_GREEN),
            _metric_card(f"Bear ({lo_p}th)", f"{bear:+.2f}%", CLR_RED),
            _metric_card("Prob. of Profit", f"{prob_profit:.1f}%", CLR_BLUE),
            _metric_card("Expected Value",  f"${tv*(1+np.mean(final_rets)):,.0f}", CLR_TEXT),
        ]
        return fig, stats
    except Exception as e:
        return _empty_fig(f"Forecast error: {e}"), [dbc.Alert(str(e), color="danger")]


# ── CB-18  Forecasting — scenario analysis ────────────────────────────────────
@app.callback(
    Output("fig-scenario",        "figure"),
    Output("div-scenario-result", "children"),
    Input("btn-run-scenario", "n_clicks"),
    State("dd-scenario",      "value"),
    State("store-portfolio",  "data"),
    prevent_initial_call=True,
)
def cb_run_scenario(n, scenario, portfolio_data):
    try:
        if scenario not in SCENARIOS:
            return _empty_fig("Unknown scenario"), _no_data_alert("Select a valid scenario.")
        positions = (portfolio_data or {}).get("positions", [])
        if not positions:
            return _empty_fig("No positions"), _no_data_alert("Add positions first.")
        start_date, end_date, title = SCENARIOS[scenario]
        tickers     = [p["ticker"] for p in positions]
        all_tickers = list(set(tickers + [BENCHMARK]))
        if USE_CACHE:
            closes = _cache_closes_df(all_tickers, start_date, end_date)
            if closes is None or closes.empty:
                return _empty_fig("No data for this period"), _no_data_alert(f"No data for {title}.")
        else:
            try:
                raw = yf.download(all_tickers, start=start_date, end=end_date,
                                  auto_adjust=True, progress=False)
            except Exception as e:
                return _empty_fig("Fetch failed"), dbc.Alert(f"Download failed: {e}", color="danger")
            if raw.empty:
                return _empty_fig("No data for this period"), _no_data_alert(f"No data for {title}.")
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    closes = raw["Close"]
                else:
                    return _empty_fig("Unexpected format"), _no_data_alert("Unexpected data format.")
            else:
                closes = pd.DataFrame({all_tickers[0]: raw["Close"]}) if "Close" in raw.columns else None
                if closes is None:
                    return _empty_fig("No close column"), _no_data_alert("No Close column.")
        available = [t for t in tickers if t in closes.columns and not closes[t].dropna().empty]
        excluded  = [t for t in tickers if t not in available]
        if not available:
            return _empty_fig("No tickers available"), _no_data_alert(
                f"None of your tickers have data for {title} (they may have listed later).")
        uw = _get_weights(portfolio_data)
        sub = {t: uw.get(t, 0) for t in available}
        tot = sum(sub.values())
        weights_s = {t: v/tot for t, v in sub.items()} if tot > 0 else {t: 1.0/len(available) for t in available}
        cols = available + ([BENCHMARK] if BENCHMARK in closes.columns else [])
        closes = closes[cols].dropna()
        if closes.empty or len(closes) < 3:
            return _empty_fig("Insufficient aligned data"), _no_data_alert("Too few aligned trading days.")
        rets = closes.pct_change().dropna()
        port_rets = sum(rets[t]*weights_s.get(t, 0) for t in available if t in rets.columns)
        port_cum = (1 + port_rets).cumprod(); port_cum = port_cum/port_cum.iloc[0]*100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum.values, mode="lines",
            name="Portfolio", line=dict(color=CLR_GREEN, width=2.5),
            hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>Portfolio</extra>"))
        bench_cum = None
        if BENCHMARK in rets.columns:
            bc = (1 + rets[BENCHMARK]).cumprod(); bench_cum = bc/bc.iloc[0]*100
            fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode="lines",
                name=BENCH_LABEL, line=dict(color=CLR_BLUE, width=2, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}: %{y:.2f}<extra>" + BENCH_LABEL + "</extra>"))
        fig.add_hline(y=100, line_dash="dot", line_color="#666")
        _dark_layout(fig, f"Scenario: {title}")
        fig.update_yaxes(title_text="Index (100 = scenario start)")
        fig.update_xaxes(title_text="Date")
        port_total = float(port_cum.iloc[-1]/100 - 1)*100
        port_dd    = float((port_cum/port_cum.cummax() - 1).min())*100
        bench_total = float(bench_cum.iloc[-1]/100 - 1)*100 if bench_cum is not None else None
        def _fc(v): return CLR_GREEN if v >= 0 else CLR_RED
        cards = [_metric_card("Portfolio Return", f"{port_total:+.2f}%", _fc(port_total)),
                 _metric_card("Max Drawdown", f"{port_dd:.2f}%", CLR_RED)]
        if bench_total is not None:
            cards.append(_metric_card(f"{BENCH_LABEL} Return", f"{bench_total:+.2f}%", _fc(bench_total)))
            ex = port_total - bench_total
            cards.append(_metric_card("Excess Return", f"{ex:+.2f}%", _fc(ex)))
        items = [dbc.Row(cards, className="g-3 mb-3")]
        if excluded:
            items.append(dbc.Alert([html.Strong("Excluded (no data for this period): "),
                                    html.Span(", ".join(excluded))], color="warning",
                                   className="mb-2 py-2"))
        trows = []
        for t in available:
            if t in rets.columns:
                tc = (1 + rets[t]).cumprod(); tr = float(tc.iloc[-1] - 1)*100
                trows.append(html.Tr([
                    html.Td(t, style={"fontFamily":"monospace","fontWeight":"bold"}),
                    html.Td(get_display_name(t), style={"color":"#aaa","fontSize":"11px"}),
                    html.Td(f"{tr:+.2f}%", style={"color":_fc(tr),"fontWeight":"600","fontFamily":"monospace"}),
                ]))
        if trows:
            items.append(dbc.Card(dbc.CardBody([
                html.H6("Individual Ticker Returns During Scenario", className="card-title text-muted mb-2"),
                dbc.Table([html.Thead(html.Tr([html.Th("Ticker"), html.Th("Company"), html.Th("Return")])),
                           html.Tbody(trows)], bordered=False, size="sm", style={"fontSize":"12px"}),
            ]), className="shadow-sm"))
        return fig, html.Div(items)
    except Exception as e:
        return _empty_fig(f"Scenario error: {e}"), dbc.Alert(str(e), color="danger")



# ENTRY POINT  (auto-opens browser)

# In cached mode (cloud), warm the cache from disk/snapshot and start the
# twice-daily refresh. Runs at import so it is active under gunicorn too.
if USE_CACHE:
    bootstrap_cache()
    start_scheduler()

def _open_browser():
    try:
        webbrowser.open_new("http://127.0.0.1:8050/")
    except Exception:
        pass

if __name__ == "__main__":
    Timer(1.25, _open_browser).start()
    # use_reloader=False keeps a single process so the browser opens exactly once
    app.run(debug=True, port=8050, use_reloader=False)
