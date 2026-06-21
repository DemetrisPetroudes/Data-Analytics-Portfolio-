# Frontier

An S&P 500 portfolio optimization, risk, and forecasting dashboard built with Plotly Dash.

Frontier is a single-file Python application for building a weighted equity portfolio from
S&P 500 large caps and analyzing its risk, return, optimal allocation, and forward-looking
behavior on a dark Bootstrap theme. The name refers to the efficient frontier that sits at
the center of the optimizer. All market data is pulled from Yahoo Finance at runtime; no
accounts or API keys are required.

![Frontier dashboard demo](dashboard_demo.gif)

## Overview

Frontier models a portfolio as a set of tickers with percentage weights and a total
notional value. You assemble it on the Builder tab, and four analytics views then
compute exposures, historical performance against the S&P 500, a mean-variance optimal
allocation, and a Monte Carlo forecast with historical stress tests. Everything runs
locally.

## Features

### Builder
- Choose from roughly 200 large-cap US tickers spanning all sectors.
- Assign a percentage weight to each position; an editable table holds the working portfolio.
- Live total-weight indicator and one-click normalization to 100%.
- Set the total portfolio value, save to disk (`portfolio.json`), and launch the analytics.

### Overview
- Summary cards for annualized return and volatility, Sharpe, Sortino, Calmar,
  VaR(95%), CVaR(95%), Information Ratio, Omega, beta, and maximum drawdown.
- Allocation donuts by weight, sector, and country/currency. Sector and country are read
  from per-instrument metadata.
- Cumulative return against the S&P 500, normalized to a common base of 100.
- Per-position detail table.

### Optimizer
- Markowitz mean-variance optimization via sequential least-squares (SLSQP), long-only
  and fully invested.
- Objectives: maximize Sharpe, minimize volatility, or maximize Sortino.
- Efficient frontier drawn from 10,000 random long-only portfolios with the optimized
  frontier overlaid.
- Six risk-profile presets, from Ultra-Defensive to Ultra-Aggressive, each defining a
  target beta, Sharpe, drawdown, volatility, Calmar, and VaR.

### Forecasting and Scenario Analysis
- Monte Carlo simulation of portfolio value under geometric Brownian motion, with
  per-asset volatility seeded from a GARCH(1,1) fit.
- Configurable horizon (21 to 1260 trading days) and path count (10,000 to 200,000),
  with selectable confidence bands.
- Fan chart and terminal-value summary statistics.
- Historical scenario replay across seven stress windows: the 2008 global financial
  crisis, the 2010 flash crash, the 2011 European debt crisis, the 2015-16 China
  slowdown, the Q4 2018 selloff, the 2020 COVID crash, and the 2022 rate-hike bear
  market.

## Methodology

The estimators are kept transparent so the numbers are easy to check against a textbook.

**Returns and risk.** Daily simple returns are annualized on a 252-day convention.
Volatility is the annualized standard deviation of daily returns. Beta is the covariance
of portfolio and benchmark returns over the benchmark variance. Downside deviation, used
for Sortino, considers only negative returns. VaR(95%) is the empirical 5th percentile of
the daily return distribution, and CVaR(95%) is the mean loss beyond it. Calmar is
annualized return over the absolute maximum drawdown. The benchmark is the S&P 500
(`^GSPC`) and the risk-free rate is taken from the 13-week US Treasury bill (`^IRX`).

**Volatility model.** For the forecast, each asset's conditional volatility is estimated
with a GARCH(1,1) model using the `arch` package and annualized for use in the
simulation.

**Monte Carlo.** Paths follow geometric Brownian motion with the Ito drift correction,
so the per-step log drift is `(mu - 0.5 * sigma^2) * dt`. Log returns are accumulated and
exponentiated to produce price paths. Generation is chunked under a fixed memory budget,
so large path counts reduce the effective count rather than exhausting memory.

**Optimization.** The efficient frontier is approximated by sampling random long-only
weight vectors, and the optimal portfolio is found with constrained numerical
optimization (weights non-negative and summing to one) for the chosen objective.

**A note on conventions.** This build uses empirical sample estimators for means and
covariances. That is the standard textbook approach and keeps the logic legible, but
sample means in particular are noisy inputs to optimization. A variant using covariance
and mean shrinkage (Ledoit-Wolf, James-Stein) is maintained separately for a more
institutional treatment.

## Tech stack

Python with Plotly Dash and dash-bootstrap-components for the interface, pandas and NumPy
for data handling, SciPy for optimization, `arch` for the GARCH model, and yfinance for
market data. The application is a single file, `app.py`.

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

The app opens at http://127.0.0.1:8050 automatically. See `RUNNING.md` for full setup
steps, a first-run walkthrough, and troubleshooting.

## Data and persistence

Market data is retrieved at runtime from Yahoo Finance through yfinance. The working
portfolio is saved to `portfolio.json` in the project directory and reloaded on the next
launch.

## Limitations

- yfinance is an unofficial interface to Yahoo Finance and can rate-limit requests. This
  is more likely from shared or cloud IP addresses. A `JSONDecodeError` or empty data
  almost always means yfinance and curl_cffi are out of date or a temporary rate limit is
  in effect.
- The first analytics load fetches per-instrument metadata (sector, country) for every
  position, which is the main source of startup latency.
- Estimates are historical and unconditional. The forecast is illustrative and is not
  investment advice.
- The app runs Dash in debug mode for local development and is not hardened for public
  production hosting as it stands.

## License

Add a license of your choice. MIT is a common default for portfolio projects.
