# Running Frontier

Setup steps, a short tour of the app, and fixes for the common snags. Tested on Windows
with PowerShell; the same steps work on macOS and Linux.

## Prerequisites

- Python 3.10 or newer.
- An internet connection. Market data is fetched from Yahoo Finance at runtime.
- Around 300 MB of free memory for a default run, more if you raise the Monte Carlo path
  count.

## 1. Get the code

```
git clone <your-repo-url>
cd <repo>
```

## 2. Create a virtual environment

A project-local environment keeps these dependencies from clashing with other Python
work.

Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks the activation script with an execution-policy error, allow local
scripts for your user account and try again:
```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
That is a Windows security setting, not a problem with the app.

## 3. Install dependencies

```
pip install -r requirements.txt
```

If you have not added a `requirements.txt` yet, install directly:
```
pip install dash dash-bootstrap-components plotly pandas numpy scipy yfinance arch curl_cffi
```

A minimal `requirements.txt`:
```
dash>=2.14
dash-bootstrap-components>=1.5
plotly>=5.18
pandas>=2.0
numpy>=1.24
scipy>=1.11
yfinance>=0.2.40
arch>=6.2
curl_cffi>=0.6
```
For an exact, reproducible set, run `pip freeze > requirements.txt` once the app is
working on your machine.

## 4. Run the app

```
python app.py
```

The server starts on port 8050 and your default browser opens at
http://127.0.0.1:8050 on its own. Press Ctrl+C in the terminal to stop it.

Run the file as a script, not inside the Python interpreter. If you start `python` first
and then paste the file, or use an editor command that sends the whole file to the
terminal line by line, Python evaluates each line on its own and reports errors such as
`SyntaxError: invalid syntax`, `IndentationError: unexpected indent`, or `unmatched ')'`,
with `File "<stdin>"` in the traceback. Those come from how the code was handed to Python,
not from the code itself. The fix is to run the whole file in one command, `python app.py`.
A sign that you are in the interpreter is the `>>>` prompt; type `exit()` to leave it.

## First run: a short tour

1. The app opens on the Builder tab. Pick a ticker, enter a weight in percent, and click
   Add. Repeat for a few names.
2. Click Normalize to scale the weights to 100%, or edit them directly in the table.
3. Set a total portfolio value if you want dollar figures.
4. Click Launch, or simply open the Overview tab, to compute the analytics.
5. Overview shows risk metrics, allocation breakdowns, and performance against the
   S&P 500. Optimizer computes an optimal allocation and the efficient frontier.
   Forecasting runs the Monte Carlo simulation and the historical scenario replays.

The first analytics load is the slowest, because the app downloads price history and
per-instrument metadata for each position. Later interactions are faster.

## Troubleshooting

**JSONDecodeError, empty charts, or "rate limited" messages.** Almost always an
out-of-date data layer or a temporary Yahoo rate limit. Update the packages and retry:
```
pip install --upgrade yfinance curl_cffi
```
If it persists you are likely being rate-limited. Wait a few minutes and avoid reloading
repeatedly. Rate limits are more aggressive from cloud or VPN addresses than from a home
connection.

**Buttons do nothing when clicked.** First confirm you launched with `python app.py` and
not in the interpreter (see above). If the app is running correctly and Add, Clear, or
Launch still have no effect, it is a known interaction between this version's duplicate
callback outputs and certain Dash releases. Pin Dash to a version where it works (record
yours with `pip freeze`), or switch to the single-owner callback build if you have it.

**Port 8050 already in use.** Stop the other process, or change the port in the
`app.run(...)` call at the bottom of `app.py` to, for example, 8051.

**The browser does not open by itself.** Open http://127.0.0.1:8050 manually. The server
is still running in the terminal.

**Slow startup or high memory use.** Startup latency is dominated by metadata fetches and
is expected. For the Monte Carlo, very high path counts use more memory; the app caps the
work under a memory budget, but lowering the path count keeps runs quick on modest
machines.

## Data and files

- `portfolio.json` is written to the project directory the first time you save and is
  reloaded on the next launch. Delete it to start from an empty portfolio.
- No accounts, keys, or paid data feeds are needed.

## Notes

- The app starts Dash in debug mode, which is convenient locally (clear error pages, no
  caching) but is not meant for public deployment as it stands.
- All figures are historical estimates and the forecast is illustrative. Nothing here is
  investment advice.
