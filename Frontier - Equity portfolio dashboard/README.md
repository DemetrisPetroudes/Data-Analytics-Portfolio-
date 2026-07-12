# Frontier

An S&P 500 portfolio optimization, risk, and forecasting dashboard built with Plotly Dash.

Frontier is a single-file Python application for building a weighted equity portfolio from
S&P 500 large caps and analyzing its risk, return, optimal allocation, and forward-looking
behavior. The name refers to the efficient frontier at the center of the optimizer.

**Live demo:** https://huggingface.co/spaces/dpetro05/Frontier

<img width="920" height="486" alt="dashboard_demo" src="https://github.com/user-attachments/assets/06a69f73-7933-4a02-a59b-8a9f3c8ec2a6" />

## Overview

Frontier models a portfolio as a set of tickers with percentage weights and a total
notional value. You assemble it on the Builder tab, and three analytics views then compute
exposures and risk, an optimal allocation, and a forward-looking simulation with
historical stress tests.

A built-in Start Guide walks through every tab and explains the models and the intuition
behind them, so the dashboard is usable without prior background in portfolio theory.

## Features

### Builder
- Choose from roughly 200 large-cap US tickers spanning all sectors.
- Assign a percentage weight to each position; an editable table holds the working portfolio.
- Live total-weight indicator and one-click normalization to 100%.
- Set the total portfolio value, save the portfolio, and launch the analytics.

### Overview
- Summary cards for portfolio value, day change, annualized volatility, and position count,
  with a 6M / 1Y / 3Y toggle that switches the backtest window for the cards and the
  cumulative return chart.
- Full risk suite on the 36-month window: Sharpe, Sortino, Calmar, VaR(95%), CVaR(95%),
  Information Ratio, Omega, Blume-adjusted beta, and maximum drawdown.
- Allocation donuts by weight, sector, and country or currency.
- Cumulative return against the S&P 500 Total Return index, normalized to a common base
  of 100.
- Position detail table with trailing 1M, 6M, 1Y and 3Y returns per holding, plus each
  position's annualized volatility and Blume-adjusted beta.

### Optimizer
- Markowitz mean-variance optimization via sequential least-squares (SLSQP), long-only and
  fully invested, with configurable minimum and maximum position weights.
- Objectives: maximize Sharpe, minimize volatility, or maximize Sortino.
- Estimation-noise controls: Ledoit-Wolf shrinkage of the covariance matrix and
  James-Stein-style shrinkage of mean returns toward the cross-sectional average.
- Efficient frontier drawn from 10,000 random long-only portfolios with the optimized
  point overlaid.
- Six risk-profile presets, from Ultra-Defensive to Ultra-Aggressive, each defining target
  beta, Sharpe, drawdown, volatility, Calmar, and VaR.

### Forecasting and Scenario Analysis
- Monte Carlo simulation of portfolio value with a simulated GARCH(1,1) conditional
  variance path, preserving volatility clustering and mean reversion rather than assuming
  constant volatility.
- Student-t(5) shocks, standardized to unit variance, for fat tails without inflating the
  calibrated volatility.
- Configurable horizon (21 to 1260 trading days) and path count (10,000 to 200,000), with
  selectable confidence bands, a fan chart, and terminal-value statistics.
- Historical scenario replay across seven stress windows: the 2008 global financial crisis,
  the 2010 flash crash, the 2011 European debt crisis, the 2015-16 China slowdown, the
  Q4 2018 selloff, the 2020 COVID crash, and the 2022 rate-hike bear market.

## Methodology

**Data window.** All estimates use 36 months of daily closing prices, adjusted for
dividends and splits. Three years balances two failure modes: shorter windows turn tail
metrics such as VaR into statistical noise, while much longer windows average across
regimes that no longer describe the market. The Overview toggle shortens the *performance*
backtest to 6 or 12 months on request, but the risk metrics deliberately remain on the
full window.

**Benchmark.** The benchmark is the S&P 500 Total Return index (`^SP500TR`). Holdings are
dividend-adjusted, so the benchmark must be as well; comparing against the price-only index
would overstate relative performance by roughly the dividend yield every year. The
risk-free rate is taken from the 13-week US Treasury bill (`^IRX`).

**Returns.** Annualized returns are geometric, meaning the constant compound rate that
reproduces the realized end value. Annualizing the arithmetic mean of daily returns
systematically overstates growth under volatility, so the geometric convention is applied
consistently, including as the drift of the Monte Carlo.

**Risk metrics.** Volatility is the annualized standard deviation of daily returns.
Sortino replaces total volatility with downside deviation. Calmar divides annualized
return by the absolute maximum drawdown of the window. VaR(95%) is the empirical 5th
percentile of the daily return distribution, and CVaR(95%) is the mean loss beyond it;
CVaR is the coherent of the two and is reported alongside. Beta is the sample covariance
of portfolio and benchmark returns over benchmark variance, then Blume-adjusted (shrunk one
third of the way toward 1) to counter the noise of daily-frequency betas, with the raw
sample beta reported for transparency.

**Volatility model.** Portfolio conditional variance is modeled with GARCH(1,1) using the
`arch` package. The fitted dynamics (omega, alpha, beta, and the current conditional
variance) are simulated forward inside the Monte Carlo rather than collapsed into a single
point forecast, so the simulation reproduces the volatility clustering and mean reversion
GARCH actually models. Non-stationary or failed fits fall back to constant volatility, and
the chart title states which model produced the paths.

**Monte Carlo.** Daily log returns are simulated as a constant log drift plus a
GARCH-driven stochastic volatility term with standardized Student-t(5) shocks. The drift is
anchored to the realized geometric return, which is the Ito-consistent choice, so the
median simulated path compounds at the same rate the Overview reports. Path generation is
chunked under a fixed memory budget, so large path counts reduce the effective count
rather than exhausting memory.

**Optimization.** The efficient frontier is approximated by sampling random long-only
weight vectors, and the optimal portfolio is found by constrained numerical optimization
for the chosen objective. The optimizer works in standard mean-variance space, which is
defined on arithmetic moments, and the frontier axis is labeled with that convention
explicitly. Its inputs are shrunk (Ledoit-Wolf covariance, James-Stein means) because naive
mean-variance amplifies estimation error, loading weight onto whatever performed best
in-sample.

## Tech stack

Python with Plotly Dash and dash-bootstrap-components for the interface, pandas and NumPy
for data handling, SciPy for optimization, scikit-learn for the Ledoit-Wolf covariance
estimator, `arch` for the GARCH model, and yfinance for market data. The application is a
single file, `app.py`.

## Quick start

```
git clone <your-repo-url>
cd <repo>
python -m venv .venv
# Windows (PowerShell):  .venv\Scripts\Activate.ps1
# macOS / Linux:         source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

The app opens at http://127.0.0.1:8050 automatically and fetches live data from Yahoo
Finance. No accounts or API keys are required. See `RUNNING.md` for full setup steps, a
first-run walkthrough, and troubleshooting.

## Deployment

The dashboard is deployed to Hugging Face Spaces on the free CPU tier using the Docker SDK.

Deployment uses a cached data mode. In this mode the app does not call Yahoo Finance on a page load.
Instead a background scheduler refreshes the whole ticker universe twice per trading day
(12:30 and 16:15 US Eastern) and every visitor is served from an in-memory cache, so pages
load immediately. This matters because yfinance is an unofficial scraper and Yahoo
rate-limits shared cloud IP addresses aggressively, which would otherwise make a public
demo unreliable.

The fallback chain is designed so a visitor never sees an empty page: a blocked refresh
keeps the last good cache, and a cold start loads a price snapshot committed to the
repository.

## Hugging Face Repository contents

- `app.py` - the application
- `build_snapshot.py` - builds the committed cold-start data snapshot
- `data/` - the snapshot (`prices.parquet`, `meta.json`), generated by the script above
- `Dockerfile`, `requirements.txt`, `.dockerignore` - deployment
- `RUNNING.md` - local setup and troubleshooting
- `DEPLOY.md` - Hugging Face Spaces deployment guide

## Limitations

These are stated plainly rather than hidden. several are inherent to building on free data.

- Historical scenario replays apply current index constituents to past windows, so they
  carry survivorship bias and will look more resilient than the index did at the time. The
  interface flags this under the scenario chart.
- Correlations are estimated over the full window and treated as static. In crises,
  cross-asset correlations rise and diversification benefits shrink precisely when they are
  most needed; the risk figures do not capture that regime dependence.
- Historical sample means remain weak forecasts of future returns even after shrinkage,
  which is why institutional practice reaches for equilibrium-implied returns
  (Black-Litterman). That is outside the scope of this build.
- The optimization is in-sample and applies no transaction costs or turnover limits, so the
  optimal weights are illustrative rather than directly tradeable.
- yfinance is an unofficial interface to Yahoo Finance and can rate-limit requests,
  particularly from cloud IP addresses. A `JSONDecodeError` or empty data usually means
  yfinance and curl_cffi are out of date or a temporary rate limit is in effect.
- All figures are historical estimates and the forecast is illustrative. Nothing here is
  investment advice.

## License

MIT
