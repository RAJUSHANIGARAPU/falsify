"""falsify — gate a backtested edge before you trust it.

A backtest is easy to make look good and almost always overfits. This is a
small, pure-stdlib toolkit (no numpy/pandas, no install-time dependencies) that
subjects a candidate strategy to the three ways a green backtest most often
fools you:

  1. MULTIPLE TESTING — Probabilistic & Deflated Sharpe Ratio (Bailey &
     Lopez de Prado). When you try N variants, the best one's Sharpe is inflated
     by selection; the DSR deflates it by the expected maximum under the null and
     returns P(true Sharpe > 0).
  2. COST FLOOR — a gross edge below round-trip-cost x turnover is not an edge.
     Most "anomalies" exist only beneath this line.
  3. AUTOCORRELATION — Newey-West (HAC) t-statistics, because a naive t-stat on
     overlapping or serially-correlated returns silently overstates significance.

Recommended entry points take RAW per-period returns and compute the statistics
internally, so units can't be mismatched (a classic foot-gun — feeding an
annualized Sharpe with an observation count saturates the DSR to ~1.0):

    >>> from falsify import discovery_verdict_from_returns
    >>> v = discovery_verdict_from_returns(
    ...         best_returns, trial_returns,
    ...         gross_edge_per_trade=0.0040, round_trip_cost=0.0008,
    ...         turnover_per_year=12)
    >>> v["verdict"]   # "PASS" | "WEAK" | "FAIL"

PASS requires surviving BOTH the multiple-testing haircut AND the cost floor.
Screen freely; ration real tests to gate-clearers (every extra test raises the
haircut); pre-register the rule and kill-criteria before pulling data.
"""
from __future__ import annotations

import math

_GAMMA = 0.5772156649015329  # Euler-Mascheroni


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's algorithm, ~1e-9 accuracy)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def probabilistic_sharpe_ratio(sr: float, sr_benchmark: float, n_obs: int,
                               skew: float = 0.0, kurt: float = 3.0) -> float:
    """P(true Sharpe > sr_benchmark).

    UNIT CONTRACT: ``sr`` and ``sr_benchmark`` are PER-PERIOD (NON-annualized)
    Sharpes over exactly ``n_obs`` observations. Feeding an *annualized* Sharpe
    with ``n_obs`` = a sample/event count silently saturates the result to ~1.0
    (this is the bug that propped prior PASS verdicts). Prefer the misuse-proof
    ``*_from_returns`` entry points below, which compute Sharpes internally.

    ``kurt`` is RAW (non-excess) kurtosis: normal == 3.0 (pass excess+3). Raises
    on degenerate input rather than masking it with a NaN/clamp that would slip
    through ``discovery_verdict`` as a non-FAIL.
    """
    if n_obs < 2:
        raise ValueError(f"probabilistic_sharpe_ratio: n_obs must be >= 2, got {n_obs}")
    var_term = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if var_term <= 0.0:
        raise ValueError(
            f"probabilistic_sharpe_ratio: non-positive variance term ({var_term:.4g}) "
            f"from implausible moments (sr={sr}, skew={skew}, kurt={kurt})"
        )
    return _norm_cdf(((sr - sr_benchmark) * math.sqrt(n_obs - 1.0)) / math.sqrt(var_term))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """Expected MAX Sharpe under the null across n_trials independent tests (the haircut)."""
    if n_trials < 1 or sr_std <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n_trials)
    z2 = _norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return sr_std * ((1.0 - _GAMMA) * z1 + _GAMMA * z2)


def deflated_sharpe_ratio(best_sr: float, trial_sharpes, n_obs: int,
                          skew: float = 0.0, kurt: float = 3.0) -> float:
    """DSR = P(best strategy's true Sharpe > 0) after deflating for multiple testing.
    trial_sharpes: the Sharpes of ALL variants tried (its length = N, its std = cross-trial dispersion)."""
    vals = [s for s in trial_sharpes if s is not None]
    n = len(vals)
    if n <= 1:
        return probabilistic_sharpe_ratio(best_sr, 0.0, n_obs, skew, kurt)
    mean = sum(vals) / n
    sr_std = math.sqrt(sum((s - mean) ** 2 for s in vals) / (n - 1))
    sr_star = expected_max_sharpe(n, sr_std)
    return probabilistic_sharpe_ratio(best_sr, sr_star, n_obs, skew, kurt)


def min_track_record_length(sr: float, sr_benchmark: float, n_obs_freq_per_year: int = 252,
                            skew: float = 0.0, kurt: float = 3.0, prob: float = 0.95) -> float:
    """Min observations to confirm sr > sr_benchmark at `prob` confidence (same per-period units)."""
    if sr <= sr_benchmark:
        return float("inf")
    var = 1.0 - skew*sr + ((kurt - 1.0) / 4.0) * sr*sr
    return 1.0 + var * (_norm_ppf(prob) / (sr - sr_benchmark)) ** 2


def cost_floor_net(gross_edge_per_trade: float, round_trip_cost: float,
                   turnover_per_year: float) -> dict:
    """A gross edge below round_trip_cost is not an edge. Returns net per-trade + annualized."""
    net_per_trade = gross_edge_per_trade - round_trip_cost
    return {
        "net_per_trade": net_per_trade,
        "net_annual": net_per_trade * turnover_per_year,
        "cost_floor_annual": round_trip_cost * turnover_per_year,
        "survives": net_per_trade > 0,
    }


def discovery_verdict(best_sr: float, trial_sharpes, n_obs: int,
                      gross_edge_per_trade: float, round_trip_cost: float, turnover_per_year: float,
                      skew: float = 0.0, kurt: float = 3.0, dsr_min: float = 0.95,
                      dsr_weak: float = 0.90) -> dict:
    """Combined gate: PASS only if it survives BOTH the multiple-testing haircut AND the cost floor.

    LEGACY low-level entry: ``best_sr`` and ``trial_sharpes`` must be PER-PERIOD
    Sharpes over ``n_obs`` (see ``probabilistic_sharpe_ratio``). For new code use
    ``discovery_verdict_from_returns`` — it takes raw per-period returns and so
    cannot be fed mismatched units.
    """
    if dsr_weak > dsr_min:
        raise ValueError(f"dsr_weak ({dsr_weak}) must be <= dsr_min ({dsr_min})")
    dsr = deflated_sharpe_ratio(best_sr, trial_sharpes, n_obs, skew, kurt)
    cost = cost_floor_net(gross_edge_per_trade, round_trip_cost, turnover_per_year)
    if dsr >= dsr_min and cost["survives"]:
        verdict = "PASS"
    elif dsr < dsr_weak or not cost["survives"]:
        verdict = "FAIL"
    else:
        verdict = "WEAK"
    return {"verdict": verdict, "deflated_sharpe": dsr, "n_trials": len([s for s in trial_sharpes if s is not None]),
            **cost}


# ---------------------------------------------------------------------------
# Misuse-proof entry points: pass RAW per-period returns, never Sharpes.
# Sharpes (and skew/kurt) are computed internally in ONE consistent frame, so
# the annualized-Sharpe-with-event-count footgun is structurally impossible.
# ---------------------------------------------------------------------------


def _sharpe_from_returns(returns, rf_per_period: float = 0.0) -> tuple[float, int]:
    """Per-period Sharpe of an excess-return series: mean/std(ddof=1). Raises on degenerate input."""
    xs = [float(r) - rf_per_period for r in returns]
    n = len(xs)
    if n < 2:
        raise ValueError(f"_sharpe_from_returns: need >= 2 returns, got {n}")
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0.0:
        raise ValueError("_sharpe_from_returns: zero variance in return series")
    return mean / math.sqrt(var), n


def _skew_kurt_raw(returns, rf_per_period: float = 0.0) -> tuple[float, float]:
    """Sample skewness and RAW (non-excess) kurtosis (normal == 3.0). Falls back to (0, 3) if flat."""
    xs = [float(r) - rf_per_period for r in returns]
    n = len(xs)
    if n < 2:
        return 0.0, 3.0
    mean = sum(xs) / n
    m2 = sum((x - mean) ** 2 for x in xs) / n
    if m2 <= 0.0:
        return 0.0, 3.0
    sd = math.sqrt(m2)
    skew = (sum((x - mean) ** 3 for x in xs) / n) / sd ** 3
    kurt = (sum((x - mean) ** 4 for x in xs) / n) / sd ** 4
    return skew, kurt


def deflated_sharpe_from_returns(best_returns, trial_returns, rf_per_period: float = 0.0) -> float:
    """DSR from RAW per-period returns. ``trial_returns`` is one return array per variant tried.

    Every Sharpe (best + trials) is computed with the SAME definition over its own
    ``n_obs``, and skew/kurtosis are estimated from ``best_returns`` — callers cannot
    mismatch units. This is the recommended path.
    """
    best_sr, n_obs = _sharpe_from_returns(best_returns, rf_per_period)
    trial_sharpes = [_sharpe_from_returns(tr, rf_per_period)[0] for tr in trial_returns]
    skew, kurt = _skew_kurt_raw(best_returns, rf_per_period)
    return deflated_sharpe_ratio(best_sr, trial_sharpes, n_obs, skew, kurt)


def discovery_verdict_from_returns(best_returns, trial_returns, *,
                                   gross_edge_per_trade: float, round_trip_cost: float,
                                   turnover_per_year: float, rf_per_period: float = 0.0,
                                   dsr_min: float = 0.95, dsr_weak: float = 0.90) -> dict:
    """Combined PASS/WEAK/FAIL gate from RAW per-period returns + cost inputs (misuse-proof units)."""
    if dsr_weak > dsr_min:
        raise ValueError(f"dsr_weak ({dsr_weak}) must be <= dsr_min ({dsr_min})")
    dsr = deflated_sharpe_from_returns(best_returns, trial_returns, rf_per_period)
    cost = cost_floor_net(gross_edge_per_trade, round_trip_cost, turnover_per_year)
    if dsr >= dsr_min and cost["survives"]:
        verdict = "PASS"
    elif dsr < dsr_weak or not cost["survives"]:
        verdict = "FAIL"
    else:
        verdict = "WEAK"
    best_sr, n_obs = _sharpe_from_returns(best_returns, rf_per_period)
    return {"verdict": verdict, "deflated_sharpe": dsr, "best_sr_per_period": best_sr,
            "n_obs": n_obs, "n_trials": len(list(trial_returns)), **cost}


def newey_west_tstat(returns, lags: int | None = None) -> dict:
    """HAC (Newey-West, Bartlett-kernel) t-stat for H0: mean(returns) == 0.

    The IID t = mean / (std/sqrt(n)) OVERSTATES significance when returns are
    serially correlated or come from OVERLAPPING holding windows (e.g. multi-day
    event holds, monthly-rebalanced daily series). This is the t-stat analogue of
    the annualized-Sharpe→DSR footgun: a falsely-confident significance number.
    Use this for any per-period return series before trusting a "t > 2" claim.

    ``lags`` defaults to the Newey-West (1994) automatic rule
    ``floor(4*(n/100)**(2/9))``. Returns {t_stat, mean, se, lags, n, iid_t}.
    """
    xs = [float(r) for r in returns]
    n = len(xs)
    if n < 3:
        raise ValueError(f"newey_west_tstat: need >= 3 observations, got {n}")
    mean = sum(xs) / n
    resid = [x - mean for x in xs]
    if lags is None:
        lags = int(4.0 * (n / 100.0) ** (2.0 / 9.0))
    lags = max(0, min(int(lags), n - 1))
    gamma0 = sum(r * r for r in resid) / n
    # Bartlett-kernel long-run variance (guaranteed >= 0).
    lrv = gamma0
    for k in range(1, lags + 1):
        weight = 1.0 - k / (lags + 1.0)
        cov_k = sum(resid[t] * resid[t - k] for t in range(k, n)) / n
        lrv += 2.0 * weight * cov_k
    if lrv <= 0.0:
        lrv = gamma0  # finite-sample safety net (Bartlett should keep lrv >= 0)
    se = math.sqrt(lrv / n)
    iid_se = math.sqrt(gamma0 / n)
    return {
        "t_stat": (mean / se) if se > 0 else 0.0,
        "mean": mean,
        "se": se,
        "lags": lags,
        "n": n,
        "iid_t": (mean / iid_se) if iid_se > 0 else 0.0,
    }
