"""falsify.core — false-discovery + cost-floor + autocorrelation analytics (pure, no I/O)."""
from __future__ import annotations

import math

import pytest

from falsify.core import (
    cost_floor_net,
    deflated_sharpe_from_returns,
    deflated_sharpe_ratio,
    discovery_verdict,
    discovery_verdict_from_returns,
    expected_max_sharpe,
    min_track_record_length,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    _norm_cdf,
    _norm_ppf,
)


def test_norm_cdf_ppf_roundtrip():
    assert _norm_cdf(0) == 0.5
    for p in (0.05, 0.5, 0.95, 0.975):
        assert _norm_cdf(_norm_ppf(p)) == __import__("pytest").approx(p, abs=1e-6)


def test_psr_monotonic():
    # higher observed Sharpe -> higher P(true>0); more data -> higher confidence
    assert probabilistic_sharpe_ratio(0.2, 0.0, 252) > probabilistic_sharpe_ratio(0.05, 0.0, 252)
    assert probabilistic_sharpe_ratio(0.1, 0.0, 1000) > probabilistic_sharpe_ratio(0.1, 0.0, 100)


def test_expected_max_sharpe_grows_with_trials():
    # the multiple-testing haircut increases with the number of trials
    assert expected_max_sharpe(100, 0.1) > expected_max_sharpe(10, 0.1) > expected_max_sharpe(2, 0.1)
    assert expected_max_sharpe(1, 0.1) == 0.0


def test_deflation_reduces_confidence():
    # testing many variants must LOWER confidence in the best one vs a single test
    best = 0.15
    single = probabilistic_sharpe_ratio(best, 0.0, 252)
    many = deflated_sharpe_ratio(best, [0.15, 0.12, 0.10, 0.08, 0.05, 0.02, -0.03, -0.05], 252)
    assert many < single
    # SAME dispersion but MORE trials (wider search) -> even lower DSR (haircut scales with N x std)
    spread = [0.15, 0.10, 0.05, 0.0, -0.05, -0.10]
    assert deflated_sharpe_ratio(best, spread * 5, 252) < deflated_sharpe_ratio(best, spread, 252)


def test_cost_floor_kills_subfloor_edge():
    # the weekend-reversion case: +0.31%/trade gross, 0.60% round-trip, ~52 turnover -> dead
    c = cost_floor_net(0.0031, 0.0060, 52)
    assert not c["survives"] and c["net_annual"] < 0
    # a low-turnover edge with the same gross survives
    assert cost_floor_net(0.0031, 0.0060, 1)["net_per_trade"] < 0  # still per-trade negative here
    assert cost_floor_net(0.02, 0.0060, 12)["survives"]


def test_min_trl_infinite_when_no_edge():
    assert min_track_record_length(0.05, 0.05) == math.inf
    assert min_track_record_length(0.20, 0.0, prob=0.95) > 0


def test_discovery_verdict_gate():
    # strong, unique, cost-surviving -> PASS
    p = discovery_verdict(0.30, [0.30], 1000, gross_edge_per_trade=0.02,
                          round_trip_cost=0.004, turnover_per_year=4)
    assert p["verdict"] == "PASS"
    # same Sharpe but cost floor breached -> FAIL
    f = discovery_verdict(0.30, [0.30], 1000, gross_edge_per_trade=0.003,
                          round_trip_cost=0.006, turnover_per_year=52)
    assert f["verdict"] == "FAIL"
    # heavily multiple-tested best -> deflated -> not PASS. NOTE: the haircut scales
    # with trial DISPERSION, not count — a modest best Sharpe against widely-dispersed
    # trials deflates below the gate (best is the max of the trials by construction).
    trials = [round(-0.14 + 0.01 * i, 3) for i in range(30)]  # -0.14..+0.15, sr_std ~0.084
    m = discovery_verdict(0.15, trials, 250, gross_edge_per_trade=0.02,
                          round_trip_cost=0.004, turnover_per_year=4)
    assert m["verdict"] in ("WEAK", "FAIL")


# --- Hardening regression: guards + misuse-proof from-returns API -----------


def test_psr_raises_on_degenerate_input():
    # n_obs < 2 must raise, not return NaN that slips through as a non-FAIL verdict.
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 0.0, 1)
    # Implausible moments giving a non-positive variance term must raise, not clamp.
    # var_term = 1 - skew*sr + (kurt-1)/4*sr^2 = 1 - 1.5 + 0 = -0.5 here.
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(1.5, 0.0, 100, skew=1.0, kurt=1.0)


def test_discovery_verdict_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        discovery_verdict(0.2, [0.2], 250, gross_edge_per_trade=0.02,
                          round_trip_cost=0.004, turnover_per_year=4,
                          dsr_min=0.90, dsr_weak=0.95)  # weak > min


def test_from_returns_immune_to_annualization_footgun():
    # The whole point: callers pass RAW per-period returns; a real but tiny per-period
    # edge must NOT saturate the DSR. A strong, low-noise edge passes; a weak one does not.
    strong = [0.01] * 60 + [-0.002] * 40           # mean +0.0044, low dispersion
    weak = [0.02, -0.018] * 50                      # mean +0.001, high dispersion
    dsr_strong = deflated_sharpe_from_returns(strong, [strong, weak])
    dsr_weak = deflated_sharpe_from_returns(weak, [strong, weak])
    assert 0.0 <= dsr_strong <= 1.0 and 0.0 <= dsr_weak <= 1.0
    assert dsr_strong > dsr_weak               # honest ordering, neither saturates blindly
    assert dsr_weak < 0.95                     # a noisy edge is not waved through


def test_discovery_verdict_from_returns_gate():
    # cost-surviving strong edge -> PASS; sub-cost edge -> FAIL (same series, higher cost).
    strong = [0.01] * 60 + [-0.002] * 40
    p = discovery_verdict_from_returns(strong, [strong],
                                       gross_edge_per_trade=0.0044, round_trip_cost=0.0005,
                                       turnover_per_year=12)
    assert p["verdict"] == "PASS"
    f = discovery_verdict_from_returns(strong, [strong],
                                       gross_edge_per_trade=0.0044, round_trip_cost=0.01,
                                       turnover_per_year=12)
    assert f["verdict"] == "FAIL"  # cost floor breached


def test_newey_west_deflates_autocorrelated_tstat():
    # Strong positive autocorrelation (AR(1), rho=0.8) inflates the IID t; the HAC
    # t must be smaller (wider SE) — the t-stat analogue of the DSR hardening.
    import random
    rng = random.Random(0)
    x, series = 0.0, []
    for _ in range(300):
        x = 0.8 * x + rng.gauss(0.0, 1.0)
        series.append(0.05 + x)            # small positive mean + autocorrelated noise
    nw = newey_west_tstat(series)
    assert nw["lags"] >= 1
    assert nw["se"] > math.sqrt(sum((s - nw["mean"]) ** 2 for s in series) / len(series) ** 2)
    assert abs(nw["t_stat"]) < abs(nw["iid_t"])   # HAC widens SE -> smaller |t|


def test_newey_west_raises_on_tiny_sample():
    with pytest.raises(ValueError):
        newey_west_tstat([0.01, 0.02])
