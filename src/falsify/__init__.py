"""falsify — gate a backtested edge before you trust it.

Pure-stdlib toolkit that subjects a candidate strategy to the three ways a
green backtest most often fools you: multiple-testing (deflated Sharpe), the
cost floor, and autocorrelation (Newey-West). See ``falsify.core`` for full
docstrings.
"""
from .core import (
    cost_floor_net,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    discovery_verdict,
    discovery_verdict_from_returns,
    expected_max_sharpe,
    min_track_record_length,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
)

__all__ = [
    # recommended (misuse-proof: pass raw per-period returns)
    "discovery_verdict_from_returns",
    "deflated_sharpe_from_returns",
    "newey_west_tstat",
    # lower-level (per-period Sharpe inputs)
    "discovery_verdict",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "min_track_record_length",
    "cost_floor_net",
]

__version__ = "0.1.0"
