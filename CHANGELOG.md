# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.1.0] - 2026-06-23

Initial release: a pure-standard-library toolkit for gating a backtested trading
edge against the three ways backtests overfit.

### Added
- **Deflated Sharpe Ratio** (Bailey & López de Prado) with the expected-maximum-Sharpe
  haircut, to correct for multiple testing across strategy candidates.
- **Cost-floor check** — net-of-cost per-trade survival, so an edge that only exists
  gross of transaction costs is rejected.
- **Autocorrelation check** — Newey-West / HAC t-statistic with the NW(1994) automatic
  lag-selection rule.
- Combined `PASS` / `WEAK` / `FAIL` verdict over the three gates.
- Misuse-resistant `*_from_returns` entry points that compute Sharpe internally, so the
  classic annualized-Sharpe-with-event-count footgun is structurally impossible;
  low-level functions raise on degenerate input instead of masking it with `NaN`.
- Acklam inverse-normal approximation (~1e-9 accuracy) and a Bartlett-kernel
  Newey-West implementation, from scratch.
- Zero runtime dependencies; supported on Python 3.9–3.13.
- Synthetic demo (`examples/synthetic_demo.py`) and a test suite.

[Unreleased]: https://github.com/RAJUSHANIGARAPU/falsify/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/RAJUSHANIGARAPU/falsify/releases/tag/v0.1.0
