"""Worked example — kill a noisy 'edge', confirm a clean one. No external data.

    python examples/synthetic_demo.py   (after: pip install -e .)

Two candidates with the SAME positive mean per-trade return: one with low noise
(a genuine small edge) and one with 12x the dispersion (noise dressed up as an
edge). Each is judged against eight weaker variants you "also tried". The gate
passes the clean one and refuses the noisy one — and Newey-West shows how much
of a naive t-stat is real.
"""
import random

from falsify import discovery_verdict_from_returns, newey_west_tstat

rng = random.Random(7)
N = 400


def series(mean, sd):
    return [mean + rng.gauss(0.0, sd) for _ in range(N)]


clean = series(0.0040, 0.004)                       # tiny edge, low noise
noisy = series(0.0040, 0.050)                        # same mean, 12x dispersion
trials_clean = [clean] + [series(0.0010, 0.004) for _ in range(8)]
trials_noisy = [noisy] + [series(0.0010, 0.050) for _ in range(8)]

print(f"{'candidate':24s} {'verdict':7s} {'DSR':>6s} {'net/trade':>10s} {'NW t':>7s}")
print("-" * 60)
for label, best, trials in (("clean low-noise edge", clean, trials_clean),
                            ("noisy 'edge'", noisy, trials_noisy)):
    v = discovery_verdict_from_returns(
        best, trials,
        gross_edge_per_trade=sum(best) / len(best),
        round_trip_cost=0.0008, turnover_per_year=50,
    )
    nw = newey_west_tstat(best)
    print(f"{label:24s} {v['verdict']:7s} {v['deflated_sharpe']:6.3f} "
          f"{v['net_per_trade']*100:+9.3f}% {nw['t_stat']:+7.2f}")
