# falsify

[![CI](https://github.com/RAJUSHANIGARAPU/falsify/actions/workflows/ci.yml/badge.svg)](https://github.com/RAJUSHANIGARAPU/falsify/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![dependencies](https://img.shields.io/badge/runtime%20deps-none-brightgreen)

**Gate a backtested trading edge before you trust it.**

A green backtest is easy to produce and almost always overfits. `falsify` is a
small, **pure-stdlib** toolkit (no numpy, no pandas, no install-time
dependencies) that subjects a candidate strategy to the three ways a good-looking
backtest most often fools you — and returns a blunt `PASS` / `WEAK` / `FAIL`.

> The point isn't to make the backtest look good. It's to find out whether the
> edge is real *before* you risk anything on it.

---

## The three gates

| Gate | What it catches | Method |
|---|---|---|
| **Multiple testing** | The best of N tried variants looks good *by selection*, not skill. | Probabilistic & **Deflated Sharpe Ratio** (Bailey & López de Prado) — deflate by the expected maximum under the null. |
| **Cost floor** | A gross edge that sits below `round-trip cost × turnover` is not an edge. | Net-of-cost per-trade survival. Most "anomalies" live entirely beneath this line. |
| **Autocorrelation** | A naive t-stat on overlapping / serially-correlated returns *overstates* significance. | **Newey-West (HAC)** t-statistic with the automatic Bartlett lag rule. |

`PASS` requires surviving **both** the multiple-testing haircut **and** the cost floor.

---

## Install

The PyPI distribution name is **`falsify-edge`**; the import name stays `falsify`.
The name `falsify` on PyPI belongs to an unrelated project, so `pip install falsify`
installs a different package. Until the first `falsify-edge` release is out, install
from source:

```bash
pip install git+https://github.com/RAJUSHANIGARAPU/falsify
```

or, from a clone (editable, for development):

```bash
git clone https://github.com/RAJUSHANIGARAPU/falsify
cd falsify
pip install -e .
```

Python ≥ 3.9. Runtime dependencies: **none**.

Not to be confused with [`falsify-quant`](https://pypi.org/project/falsify-quant/),
a separate AGPL-licensed CLI (numpy/scipy) that scores a strategy file 0–100 across
seven checks; this project is an MIT, dependency-free library of three gates you call
on return series you already have.

---

## Quickstart

The recommended entry points take **raw per-period returns** and compute every
statistic internally, so you can't mismatch units (see *Design* below):

```python
from falsify import discovery_verdict_from_returns, newey_west_tstat

# best  = per-trade (or per-period) return series of your chosen strategy
# trials = one return series per variant you tried (best is one of them)
verdict = discovery_verdict_from_returns(
    best, trials,
    gross_edge_per_trade=0.0040,   # mean gross edge per trade
    round_trip_cost=0.0008,        # all-in round-trip cost
    turnover_per_year=12,
)
print(verdict["verdict"])          # "PASS" | "WEAK" | "FAIL"
print(verdict["deflated_sharpe"])  # multiple-testing-adjusted P(true Sharpe > 0)

# Always sanity-check significance under autocorrelation:
nw = newey_west_tstat(best)
print(nw["t_stat"], "vs naive", nw["iid_t"])   # HAC t can be much smaller
```

Run the worked example (no data needed):

```bash
python examples/synthetic_demo.py
```

```
candidate                verdict    DSR  net/trade    NW t
------------------------------------------------------------
clean low-noise edge     PASS     1.000    +0.327%  +20.40
noisy 'edge'             WEAK     0.928    +0.592%   +2.61
```

Both candidates have the same mean return per trade. The clean one passes. The noisy
one clears the cost floor, but its deflated Sharpe (0.928) sits between the `WEAK`
threshold (0.90) and the `PASS` threshold (0.95), so it is held at `WEAK` and not
promoted. It is not outright `FAIL`ed; that needs a deflated Sharpe below 0.90 or a
cost-floor failure.

---

## Design — why "from returns"

The most common foot-gun in this kind of analysis is a **unit mismatch**: feeding
an *annualized* Sharpe into a per-period test with `n_obs` set to a sample/event
count silently saturates the deflated Sharpe to ~1.0 — a false `PASS`. The
`*_from_returns` functions remove the choice entirely: you hand over raw return
arrays and the library computes Sharpes, skew, and kurtosis itself, in one
consistent frame. The lower-level functions remain available, but they raise on
degenerate input (`n_obs < 2`, non-positive variance) instead of masking it.

---

## API

| Function | Use |
|---|---|
| `discovery_verdict_from_returns(best, trials, *, gross_edge_per_trade, round_trip_cost, turnover_per_year, …)` | **Start here.** Combined gate from raw returns. |
| `deflated_sharpe_from_returns(best, trials)` | Multiple-testing-adjusted P(Sharpe > 0) from raw returns. |
| `newey_west_tstat(returns, lags=None)` | HAC t-stat (and the naive `iid_t` for comparison). |
| `discovery_verdict(best_sr, trial_sharpes, n_obs, …)` | Lower-level gate (per-period Sharpe inputs). |
| `deflated_sharpe_ratio` / `probabilistic_sharpe_ratio` / `expected_max_sharpe` | Building blocks. |
| `cost_floor_net` / `min_track_record_length` | Cost survival and the sample size needed to confirm an edge. |

---

## The discipline it encodes

The functions are only half of it; the workflow is the other half:

1. **Screen freely, test rarely.** Generating ideas is free and raises no
   statistical haircut. *Running a real test does* — so ration tests to ideas
   with a plausible structural mechanism.
2. **Pre-register.** Freeze the rule, the variant grid, the benchmark, and the
   kill-criteria *before* pulling data. No goal-post moving after seeing results.
3. **A clean kill is a success.** The job is a trustworthy verdict, not a green
   one.

---

## Field notes

This was extracted from a personal quantitative research platform where it was
used in anger as the gate on every candidate strategy. Across ~30 candidates —
forecasting, calendar/flow effects, cross-asset signals, carry, event-driven —
**none survived** hostile out-of-sample testing versus simply holding the index
after costs. The closest call (a pre-FOMC drift effect) was statistically real
in-sample yet still failed out-of-sample. In the course of that work the harness
also caught a real unit-mismatch bug **in its own significance gate** — which is
why the current API is built to be misuse-proof. A tool that can falsify its own
results is the only kind worth trusting.

Full case study of the platform it came from:
**https://rajushanigarapu.github.io/autonomous-investor-writeup/**

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Releasing

Publishing to PyPI is automated via GitHub Actions using
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) — no API
token is stored in the repo. Every push builds and `twine check`s the distribution in
CI, so `main` is always release-ready.

To cut a release:

1. **One-time:** on PyPI, add a pending Trusted Publisher for project `falsify-edge` pointing at
   this repo, workflow `publish.yml`, and environment `pypi`.
2. Bump `version` in `pyproject.toml`, commit, and tag (`git tag v0.1.1 && git push --tags`).
3. Publish a GitHub Release for that tag — the `Publish to PyPI` workflow builds and
   uploads automatically.

## License

MIT — see [LICENSE](LICENSE).
