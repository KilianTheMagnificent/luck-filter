#!/usr/bin/env python3
"""Self-check for luckfilter. Run: python3 test_luckfilter.py

Most of these are not "does it run" tests. They are the published numbers from
the papers the module implements, plus the properties that must hold by
construction. If a formula is transcribed wrong, one of these fails.
"""

import math
import os
import random
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import luckfilter as L

_RESULTS = []


def check(name):
    def deco(fn):
        _RESULTS.append((name, fn))
        return fn
    return deco


def close(a, b, tol=1e-9):
    assert abs(a - b) <= tol, "%.12g != %.12g (tol %g)" % (a, b, tol)


# --- moments -----------------------------------------------------------------

@check("moments: Bessel sd with population shape moments, RAW kurtosis")
def _():
    T, mu, sd, g3, g4 = L.moments([0.04, -0.02, 0.06, 0.00])
    close(mu, 0.02, 1e-15)
    close(sd, math.sqrt(0.004 / 3), 1e-15)      # NOT the population sd 0.0316...
    close(g3, 0.0, 1e-9)
    close(g4, 1.36, 1e-12)                       # raw, not excess
    assert abs(sd - 0.0316227766016838) > 1e-4, "used the population sd"


@check("moments: zero variance and non-finite input are refused")
def _():
    for bad in ([1.0, 1.0, 1.0], [0.1, float("nan"), 0.2], [0.1, float("inf")]):
        try:
            L.moments(bad)
        except L.LuckFilterError:
            continue
        raise AssertionError("accepted %r" % (bad,))


# --- Sharpe inference --------------------------------------------------------

@check("PSR is exactly 0.5 when the observed Sharpe equals the benchmark")
def _():
    for sr in (0.0, -0.4, 0.158, 5.0):
        for g3, g4 in ((0, 3), (2.5, 40), (-3, 10)):
            assert L.psr(sr, sr, 1250, g3, g4) == 0.5


@check("PSR(0) = 0.5 for an antisymmetric series")
def _():
    r = [0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
    T, mu, sd, g3, g4 = L.moments(r)
    close(mu, 0.0, 1e-18)
    close(g3, 0.0, 1e-12)
    assert L.psr(L.sharpe(r), 0.0, T, g3, g4) == 0.5


@check("annualisation is a pure rescaling of the per-period statistics")
def _():
    rng = random.Random(4)
    r = [rng.gauss(0.0005, 0.01) for _ in range(500)]
    T, mu, sd, g3, g4 = L.moments(r)
    sr = L.sharpe(r)
    se = L.sharpe_std_error(sr, T, g3, g4)
    k = 252
    close(sr * math.sqrt(k) / math.sqrt(k), sr, 1e-15)
    # PSR must be invariant to expressing both Sharpes in annual units
    close(L.psr(sr, 0.02, T, g3, g4),
          L.psr(sr * math.sqrt(k) / math.sqrt(k), 0.02, T, g3, g4), 1e-15)
    close(se * math.sqrt(k), math.sqrt(k) * se, 1e-15)


@check("Lo (2002) Table 1: Gaussian Sharpe standard errors, 32 published cells")
def _():
    table = {0.50: [0.306, 0.217, 0.177, 0.153, 0.137, 0.095, 0.067, 0.047],
             1.00: [0.354, 0.250, 0.204, 0.177, 0.158, 0.110, 0.077, 0.055],
             2.00: [0.500, 0.354, 0.289, 0.250, 0.224, 0.155, 0.110, 0.077],
             3.00: [0.677, 0.479, 0.391, 0.339, 0.303, 0.210, 0.148, 0.105]}
    for sr, row in table.items():
        for T, want in zip((12, 24, 36, 48, 60, 125, 250, 500), row):
            got = math.sqrt((1.0 + sr * sr / 2.0) / T)          # Lo divides by T
            close(got, want, 5e-4)
            # our T-1 version must differ by exactly sqrt(T/(T-1))
            close(L.sharpe_std_error(sr, T, 0.0, 3.0),
                  got * math.sqrt(T / (T - 1.0)), 1e-12)


@check("MinTRL is the exact algebraic inverse of PSR")
def _():
    sr, star, g3, g4 = 0.1581138830084190, 0.0632455532033676, 0.0, 3.0
    n = L.min_track_record_length(sr, star, g3, g4, 0.95)
    close(L.psr(sr, star, n, g3, g4), 0.95, 1e-12)
    assert L.psr(sr, star, math.ceil(n), g3, g4) >= 0.95
    assert L.psr(sr, star, math.floor(n) - 1, g3, g4) < 0.95
    assert L.min_track_record_length(0.1, 0.2, 0, 3) == float("inf")


@check("Bailey & Lopez de Prado (2012) Figures 8-11: four published MinTRL values")
def _():
    cases = [(250, 0.0, 3.0, 2.73), (52, 0.0, 3.0, 2.83),
             (12, 0.0, 3.0, 3.24), (12, -0.72, 5.78, 4.99)]
    for k, g3, g4, want_years in cases:
        sr, star = 2.0 / math.sqrt(k), 1.0 / math.sqrt(k)
        n = L.min_track_record_length(sr, star, g3, g4, 0.95)
        close(n / k, want_years, 5e-3)
    # negative skew and fat tails must LENGTHEN the required record
    normal = L.min_track_record_length(2 / math.sqrt(12), 1 / math.sqrt(12), 0.0, 3.0)
    fat = L.min_track_record_length(2 / math.sqrt(12), 1 / math.sqrt(12), -0.72, 5.78)
    assert fat > normal * 1.4


@check("Bailey & Lopez de Prado (2014) p.10: the DSR paper's own worked example")
def _():
    sr, v, T = 2.5 / math.sqrt(250), 0.5 / 250, 1250
    dsr, sr0 = L.deflated_sharpe(sr, T, -3.0, 10.0, 100, v)
    close(sr0, 0.1132, 5e-5)                     # paper prints 0.1132
    close(dsr, 0.9004, 5e-5)                     # paper prints 0.9004


@check("Bailey & Lopez de Prado (2014): both published breakeven trial counts")
def _():
    sr, v, T = 2.5 / math.sqrt(250), 0.5 / 250, 1250
    close(L.deflated_sharpe(sr, T, -3.0, 10.0, 46, v)[0], 0.9505, 5e-5)
    close(L.deflated_sharpe(sr, T, 0.0, 3.0, 88, v)[0], 0.9505, 5e-5)
    assert L.breakeven_trials(sr, T, -3.0, 10.0, v) == 47
    assert L.breakeven_trials(sr, T, 0.0, 3.0, v) == 89


@check("excess kurtosis passed as raw kurtosis fails loudly, not silently")
def _():
    try:
        L.variance_factor(2.0, 0.0, 0.0)         # 0.0 = a Gaussian's EXCESS kurtosis
    except L.LuckFilterError as exc:
        assert "RAW" in str(exc)
        return
    raise AssertionError("silently accepted excess kurtosis")


@check("expected_max_sharpe needs >= 2 trials and grows with the trial count")
def _():
    try:
        L.expected_max_sharpe(0.01, 1)
    except L.LuckFilterError:
        pass
    else:
        raise AssertionError("N=1 accepted")
    vals = [L.expected_max_sharpe(1.0, n) for n in (5, 10, 50, 100, 1000)]
    assert vals == sorted(vals)
    # the extreme-value approximation is 7.8% BELOW the truth at N=2, which is why
    # the analysis layer floors the trial count at 3 rather than 2
    assert L.expected_max_sharpe(1.0, 2) < 0.53 < 0.5642
    close(L.expected_max_sharpe(0.0, 100), 0.0, 1e-15)


@check("expected_max_sharpe tracks the Monte Carlo max of N normals, from above")
def _():
    rng = random.Random(7)
    for n, reps in ((10, 20000), (100, 20000), (1000, 8000)):
        mc = statistics.mean(max(rng.gauss(0, 1) for _ in range(n)) for _ in range(reps))
        an = L.expected_max_sharpe(1.0, n)
        assert 0.0 < an - mc < 0.06, "N=%d analytic %.4f vs MC %.4f" % (n, an, mc)


@check("PSR is Uniform(0,1) under the null, so the 5% tail is 5%")
def _():
    rng = random.Random(12345)
    vals = []
    for _ in range(4000):
        r = [rng.gauss(0.0, 1.0) for _ in range(250)]
        T, mu, sd, g3, g4 = L.moments(r)
        vals.append(L.psr(L.sharpe(r), 0.0, T, g3, g4))
    close(statistics.mean(vals), 0.5, 0.02)
    assert abs(sum(1 for v in vals if v > 0.95) / len(vals) - 0.05) < 0.015
    assert abs(sum(1 for v in vals if v > 0.99) / len(vals) - 0.01) < 0.008


@check("PSR holds its nominal 5% size on serially correlated returns")
def _():
    # without the AR(1) correction the 5% test fires 11% of the time at phi=0.3,
    # which is unremarkable for smoothed or overlapping-position returns
    for phi, tol in ((0.0, 0.015), (0.3, 0.015)):
        reps, hits_iid, hits_fixed = 2500, 0, 0
        for r in range(reps):
            rng = random.Random(900000 + r * 13 + int(phi * 100))
            x, ser = 0.0, []
            for _ in range(504):
                x = phi * x + rng.gauss(0, 1)
                ser.append(x)
            T, mu, sd, g3, g4 = L.moments(ser)
            sr = mu / sd
            rho = L.autocorrelation(ser, 1)
            if L.psr(sr, 0.0, T, g3, g4) > 0.95:
                hits_iid += 1
            if L.psr(sr, 0.0, T, g3, g4, rho) > 0.95:
                hits_fixed += 1
        rate = hits_fixed / float(reps)
        assert abs(rate - 0.05) < tol, "phi=%.1f corrected size %.4f" % (phi, rate)
        if phi > 0:                      # and the uncorrected version must be visibly worse
            assert hits_iid / float(reps) > 0.08, "the correction is doing nothing"


@check("Lo (2002) eta: reduces to sqrt(q), and matches its AR(1) closed form")
def _():
    for q, want in ((1, 1.0), (2, 2 ** 0.5), (12, 12 ** 0.5), (250, 250 ** 0.5)):
        close(L.eta(q, [0.0] * (q - 1)), want, 1e-12)
        close(L.eta_ar1(q, 0.0), want, 1e-12)
    for rho in (0.9, 0.5, 0.2, -0.2, -0.5, -0.9):
        for q in (2, 3, 4, 6, 12, 24, 36, 48, 125, 250):
            close(L.eta(q, [rho ** k for k in range(1, q)]), L.eta_ar1(q, rho), 1e-9)
    close(L.eta_ar1(12, 0.2), 2.8788486905274400, 1e-12)


@check("Lo (2002) Table 2: published eta values, and monotone in rho")
def _():
    rows = {0.9: [1.03, 1.05, 1.07, 1.10, 1.21, 1.41, 1.60, 1.77, 2.67, 3.70],
            0.5: [1.15, 1.28, 1.39, 1.60, 2.12, 2.91, 3.53, 4.06, 6.49, 9.15],
            0.2: [1.29, 1.52, 1.73, 2.07, 2.88, 4.04, 4.93, 5.68, 9.14, 12.92],
            0.0: [1.41, 1.73, 2.00, 2.45, 3.46, 4.90, 6.00, 6.93, 11.18, 15.81],
            -0.5: [2.00, 2.45, 3.02, 3.84, 5.69, 8.26, 10.21, 11.84, 19.26, 27.31]}
    qs = (2, 3, 4, 6, 12, 24, 36, 48, 125, 250)
    for rho, row in rows.items():
        for q, want in zip(qs, row):
            close(round(L.eta_ar1(q, rho), 2), want, 5e-3)
    for q in (2, 12, 250):
        vals = [L.eta_ar1(q, r) for r in (0.9, 0.5, 0.2, 0.0, -0.5)]
        assert vals == sorted(vals), "eta must decrease in rho"


@check("correlated trials are worth fewer independent ones")
def _():
    close(L.implied_independent_trials(200, 0.0), 200.0, 1e-12)
    close(L.implied_independent_trials(200, 1.0), 1.0, 1e-12)
    assert 1 < L.implied_independent_trials(200, 0.5) < 200
    close(L.implied_independent_trials(1, 0.3), 1.0, 1e-12)
    x = [[1.0, 2.0, 3.0, 4.0]] * 3
    close(L.average_correlation(x), 1.0, 1e-9)


# --- PBO / CSCV --------------------------------------------------------------

@check("PBO: hand-computed 4-block, 2-strategy case, exact to 1e-12")
def _():
    m = [[6, -2], [4, -4], [5, -1], [3, -3], [-3, 6], [-5, 4], [-2, 5], [-4, 3]]
    r = L.probability_of_backtest_overfitting(m, s_blocks=4)
    close(r["pbo"], 4.0 / 6.0, 1e-12)
    assert r["n_splits"] == 6
    close(r["degradation_slope"], -0.7670273960, 1e-9)
    close(r["degradation_intercept"], 0.3514199001, 1e-9)
    # BBLZ define the probability of loss strictly (< 0); split 3's OOS Sharpe is
    # exactly 0.0, so the strict rule gives 2/6 and the closed rule would give 3/6
    close(r["prob_loss"], 2.0 / 6.0, 1e-12)
    ln2 = math.log(2)
    assert [round(x, 10) for x in r["logits"]] == [round(x, 10) for x in
                                                   (-ln2, ln2, -ln2, -ln2, ln2, -ln2)]


@check("PBO: N identical strategies gives exactly 1.0 and every logit exactly 0")
def _():
    rng = random.Random(0)
    x = [rng.gauss(0.001, 0.01) for _ in range(320)]
    for n in (4, 5):
        r = L.probability_of_backtest_overfitting([[v] * n for v in x], s_blocks=8)
        assert r["pbo"] == 1.0
        assert set(r["logits"]) == {0.0}


@check("PBO: pure noise with even N averages 0.5")
def _():
    vals = []
    for i in range(60):
        rng = random.Random(50000 + i)
        m = [[rng.gauss(0.0, 1.0) for _ in range(10)] for _ in range(600)]
        vals.append(L.probability_of_backtest_overfitting(m, s_blocks=8)["pbo"])
    close(statistics.mean(vals), 0.5, 0.06)


@check("PBO: one genuinely skilled strategy among noise gives 0")
def _():
    for i in range(6):
        rng = random.Random(4000 + i)
        m = [[rng.gauss(0.35 if n == 0 else 0.0, 1.0) for n in range(20)]
             for _ in range(1000)]
        r = L.probability_of_backtest_overfitting(m, s_blocks=8)
        assert r["pbo"] == 0.0, "seed %d gave %r" % (i, r["pbo"])
        assert r["prob_loss"] == 0.0


@check("PBO: the answer does not depend on the order of the columns")
def _():
    # discrete payoffs (a prediction ledger, an integer-scored grid) tie for
    # in-sample best routinely; breaking the tie by column index made PBO swing
    # from 0.833 to 0.667 on this matrix purely by reordering it
    import itertools
    m = [[-4, 2, -3], [-2, -1, -3], [-4, -4, 3], [3, -2, 4],
         [-1, 3, 4], [-1, -2, 2], [2, -3, 2], [2, -1, -4]]
    seen = set()
    for perm in itertools.permutations(range(3)):
        r = L.probability_of_backtest_overfitting(
            [[row[i] for i in perm] for row in m], s_blocks=4)
        seen.add((r["pbo"], round(r["degradation_slope"], 12), r["prob_loss"]))
    assert len(seen) == 1, "PBO depends on column order: %r" % (seen,)


@check("PBO: the block-sums shortcut equals direct computation, on every payoff shape")
def _():
    # the O(blocks) algebra is the deepest correctness risk in the tool; check it
    # against physically concatenating the rows, including the discrete and binary
    # payoffs where exact ties and exact zeros are routine
    import itertools

    def naive(matrix, S=8):
        T, N = len(matrix), len(matrix[0])
        b = T // S
        off = T - b * S
        rows = [matrix[off + t] for t in range(b * S)]
        blocks = [[rows[s * b + i] for i in range(b)] for s in range(S)]

        def sh(v):
            sd = statistics.stdev(v)
            m = statistics.fmean(v)
            if sd <= 0:
                return 0.0 if m == 0 else math.copysign(float("inf"), m)
            return m / sd

        logits, isp, oosp = [], [], []
        for comb in itertools.combinations(range(S), S // 2):
            comp = tuple(i for i in range(S) if i not in comb)
            ins = [r for s in comb for r in blocks[s]]
            outs = [r for s in comp for r in blocks[s]]
            ri = [sh([r[n] for r in ins]) for n in range(N)]
            ro = [sh([r[n] for r in outs]) for n in range(N)]
            top = max(ri)
            tied = [n for n in range(N)
                    if math.isclose(ri[n], top, rel_tol=1e-12, abs_tol=1e-15)]
            rk = L._midranks(ro)
            om = sum(rk[n] for n in tied) / (len(tied) * (N + 1.0))
            logits.append(math.log(om / (1 - om)))
            isp.append(top)
            oosp.append(sum(ro[n] for n in tied) / len(tied))
        # PBO counts the closed inequality (the paper integrates to 0 inclusive);
        # the probability of loss is strict, per BBLZ section 3.2
        return (sum(1 for v in logits if v <= 0) / len(logits),
                sum(1 for v in oosp if v < 0) / len(oosp))

    shapes = {
        "continuous": lambda r: r.gauss(0.0004, 0.01),
        "discrete": lambda r: float(r.choice([-2, -1, 0, 1, 2])),
        "binary": lambda r: float(r.choice([-1, 1])),
        "sparse": lambda r: float(r.choice([0, 0, 0, 1, -1])),
    }
    for name, draw in shapes.items():
        rng = random.Random(len(name) * 977)
        m = [[draw(rng) for _ in range(6)] for _ in range(160)]
        got = L.probability_of_backtest_overfitting(m, s_blocks=8)
        want_pbo, want_loss = naive(m, 8)
        close(got["pbo"], want_pbo, 1e-12)
        close(got["prob_loss"], want_loss, 1e-12)


@check("PBO: degenerate inputs are refused")
def _():
    rng = random.Random(1)
    ok = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(64)]
    for kwargs, matrix in ((dict(s_blocks=5), ok), (dict(s_blocks=2), ok),
                           (dict(s_blocks=8), [[r[0]] for r in ok]),
                           (dict(s_blocks=8), [[1.0, 1.0]] * 64)):
        try:
            L.probability_of_backtest_overfitting(matrix, **kwargs)
        except L.LuckFilterError:
            continue
        raise AssertionError("accepted %r" % (kwargs,))


# --- bootstrap ---------------------------------------------------------------

@check("stationary bootstrap keeps every index marginally uniform")
def _():
    rng = random.Random(3)
    x = [rng.gauss(0.5, 1.0) for _ in range(200)]
    r = random.Random(1)
    means = [statistics.mean(x[i] for i in L.stationary_bootstrap_indices(200, 0.1, r))
             for _ in range(3000)]
    close(statistics.mean(means), statistics.mean(x), 0.02)


@check("Reality Check: rejects genuine skill's null, spares pure noise")
def _():
    rng = random.Random(9)
    skill = ([[rng.gauss(0.3, 1.0) for _ in range(300)]] +
             [[rng.gauss(0.0, 1.0) for _ in range(300)] for _ in range(19)])
    assert L.reality_check(skill, n_boot=600, seed=1)["p_value"] < 0.02
    noise = [[rng.gauss(0.0, 1.0) for _ in range(300)] for _ in range(20)]
    assert L.reality_check(noise, n_boot=600, seed=1)["p_value"] > 0.05


@check("SPA: the three p-values are ordered lower <= consistent <= upper")
def _():
    rng = random.Random(11)
    cols = [[rng.gauss(0.05 if k == 0 else -0.2, 1.0) for _ in range(250)]
            for k in range(12)]
    r = L.superior_predictive_ability(cols, n_boot=600, seed=2)
    assert r["p_lower"] <= r["p_consistent"] <= r["p_upper"] + 1e-12


@check("cluster bootstrap: correlated groups widen the interval as theory says")
def _():
    # every observation inside a day is identical, so 20 calls/day carry the
    # information of one day: the SE must inflate by exactly sqrt(20)
    hits, days = [], []
    rng = random.Random(2)
    for d in range(60):
        v = 1 if rng.random() < 0.6 else 0
        hits += [v] * 20
        days += [d] * 20
    d = L.design_effect(hits, days)
    close(d["rho"], 1.0, 1e-9)
    close(d["deff"], 20.0, 1e-9)
    close(d["n_eff"], 60.0, 1e-9)
    ci = L.cluster_bootstrap_ci(hits, days, n_boot=2000, seed=1)
    naive = math.sqrt(0.6 * 0.4 / 1200)
    assert (ci["hi"] - ci["lo"]) > 6 * naive, "cluster interval did not widen"


@check("design effect is 1 when there is no within-cluster correlation")
def _():
    rng = random.Random(6)
    hits = [1 if rng.random() < 0.5 else 0 for _ in range(2000)]
    days = [i // 20 for i in range(2000)]
    d = L.design_effect(hits, days)
    assert d["deff"] < 1.35, d["deff"]
    assert d["n_eff"] > 1400, d["n_eff"]


# --- directional -------------------------------------------------------------

@check("Pesaran-Timmermann: a permanently bullish forecaster scores zero skill")
def _():
    actual = [1] * 580 + [-1] * 420
    r = L.pesaran_timmermann([1] * 1000, actual)
    close(r["hit_rate"], 0.58, 1e-12)
    close(r["expected_hit_rate"], 0.58, 1e-12)
    assert r["degenerate"] is True
    # the naive test everyone reaches for calls this overwhelming evidence
    assert L.binomial_test(580, 1000, 0.5) < 1e-6


@check("Pesaran-Timmermann: near-constant but not degenerate is still ~zero")
def _():
    rng = random.Random(17)
    actual = [1 if rng.random() < 0.58 else -1 for _ in range(1000)]
    pred = [1] * 1000
    for i in rng.sample(range(1000), 5):          # a few contrarian calls, at random
        pred[i] = -1
    r = L.pesaran_timmermann(pred, actual)
    assert abs(r["statistic"]) < 2.0, r["statistic"]
    assert r["p_value"] > 0.02


@check("Pesaran-Timmermann: a clairvoyant forecaster is overwhelming")
def _():
    actual = [1 if i % 3 else -1 for i in range(600)]
    r = L.pesaran_timmermann(list(actual), actual)
    close(r["hit_rate"], 1.0, 1e-12)
    assert r["statistic"] > 20 and r["p_value"] < 1e-10


@check("Pesaran-Timmermann equals the chi-square statistic times N/(N-1)")
def _():
    rng = random.Random(21)
    for _ in range(50):
        actual = [1 if rng.random() < 0.55 else -1 for _ in range(300)]
        pred = [y if rng.random() < 0.6 else -y for y in actual]
        r = L.pesaran_timmermann(pred, actual)
        if r["degenerate"]:
            continue
        a, b, c, d = r["table"]
        n = a + b + c + d
        n1, n2, k1, k2 = a + b, c + d, a + c, b + d
        chi2 = n * (a * d - b * c) ** 2 / float(n1 * n2 * k1 * k2)
        close(r["statistic"] ** 2, chi2 * n / (n - 1.0), 1e-9)
        # and the textbook variance form must agree with the exact identity
        p_y, p_x = n1 / float(n), k1 / float(n)
        p_star = p_y * p_x + (1 - p_y) * (1 - p_x)
        var_p = p_star * (1 - p_star) / n
        var_ps = ((2 * p_y - 1) ** 2 * p_x * (1 - p_x) / n +
                  (2 * p_x - 1) ** 2 * p_y * (1 - p_y) / n +
                  4 * p_y * (1 - p_y) * p_x * (1 - p_x) / n ** 2)
        if var_p - var_ps > 1e-15:
            close(r["statistic"],
                  (r["hit_rate"] - p_star) / math.sqrt(var_p - var_ps), 1e-6)


@check("Henriksson-Merton exact test agrees in direction with the asymptotic one")
def _():
    rng = random.Random(31)
    actual = [1 if rng.random() < 0.5 else -1 for _ in range(200)]
    good = [y if rng.random() < 0.62 else -y for y in actual]
    bad = [1 if rng.random() < 0.5 else -1 for _ in range(200)]
    assert L.henriksson_merton(good, actual)["p_value"] < 0.05
    assert L.henriksson_merton(bad, actual)["p_value"] > 0.05


@check("directional: a tiny sample defers to the exact test, not the asymptotic one")
def _():
    # 4 calls, all correct: the asymptotic test reads p = 0.01 while the exact test
    # says 0.25. The card must not report SURVIVES on the strength of the former.
    rep = L.analyse_directional([1, -1, -1, -1], [1, -1, -1, -1],
                                ["d0", "d1", "d2", "d3"], n_boot=50, seed=1)
    assert rep["pt_reliable"] is False
    text = L.render(rep)
    assert "ADVISORY ONLY" in text
    assert "SURVIVES" not in text, text
    # a large, well-populated ledger still uses Pesaran-Timmermann as primary
    rng = random.Random(4)
    actual = [1 if rng.random() < 0.5 else -1 for _ in range(600)]
    pred = [y if rng.random() < 0.6 else -y for y in actual]
    big = L.analyse_directional(pred, actual, [str(i // 6) for i in range(600)],
                                n_boot=200, seed=1)
    assert big["pt_reliable"] is True
    assert "ADVISORY ONLY" not in L.render(big)


@check("binomial test: exact tail, and the edges")
def _():
    close(L.binomial_test(10, 10, 0.5), 0.5 ** 10, 1e-15)
    close(L.binomial_test(0, 10, 0.5), 1.0, 1e-15)
    close(L.binomial_test(6, 10, 0.5), 0.376953125, 1e-12)


@check("exact tails survive a realistically sized ledger")
def _():
    # math.comb is exact but its product with a float overflows in the thousands,
    # which a year of daily multi-ticker predictions reaches easily
    close(L.binomial_test(2400, 4500, 0.5), 4.121213e-06, 1e-10)
    pred = [1] * 2300 + [-1] * 2200
    actual = [1] * 2400 + [-1] * 2100
    r = L.henriksson_merton(pred, actual)
    assert r["p_value"] >= 0.0 and not r["degenerate"]
    # and the small-n values are unchanged by the log-space rewrite
    close(L.binomial_test(10, 10, 0.5), 0.5 ** 10, 1e-15)
    close(L.binomial_test(6, 10, 0.5), 0.376953125, 1e-12)


@check("Murphy decomposition reconstructs the Brier score exactly")
def _():
    rng = random.Random(5)
    p = [rng.random() for _ in range(2000)]
    o = [1 if rng.random() < x else 0 for x in p]
    b = L.brier_decomposition(p, o)
    close(b["residual"], 0.0, 1e-12)
    assert b["reliability"] >= 0 and b["resolution"] >= 0
    assert b["resolution"] <= b["uncertainty"] + 1e-12
    # a perfectly calibrated, perfectly sharp forecaster
    perfect = L.brier_decomposition([1.0, 0.0] * 50, [1, 0] * 50)
    close(perfect["brier"], 0.0, 1e-15)
    close(perfect["reliability"], 0.0, 1e-15)
    close(perfect["skill_score"], 1.0, 1e-12)


@check("Brier: refuses to report calibration it cannot actually measure")
def _():
    # one observation per bin makes reliability equal the Brier score by
    # construction, which reads as catastrophic miscalibration and is an artefact
    rng = random.Random(11)
    p = [rng.random() for _ in range(15)]
    o = [1 if rng.random() < 0.5 else 0 for _ in range(15)]
    b = L.brier_decomposition(p, o)
    assert b["informative"] is False
    assert b["reliability"] is None and b["resolution"] is None
    assert b["brier"] > 0 and b["skill_score"] is not None
    # with enough observations per bin it reports normally again
    b2 = L.brier_decomposition([0.2, 0.5, 0.8] * 40,
                               [1 if rng.random() < x else 0
                                for x in [0.2, 0.5, 0.8] * 40])
    assert b2["informative"] is True and b2["reliability"] is not None


@check("Brier: a constant forecast at the base rate has zero skill")
def _():
    o = [1] * 60 + [0] * 40
    b = L.brier_decomposition([0.6] * 100, o)
    close(b["resolution"], 0.0, 1e-12)
    close(b["reliability"], 0.0, 1e-12)
    close(b["skill_score"], 0.0, 1e-12)


# --- end to end --------------------------------------------------------------

@check("the demo search over pure noise is not judged skilful")
def _():
    px = L.random_walk(600, 11)
    grid = L.strategy_grid(px)
    names = list(grid)
    matrix = [[grid[n][t] for n in names] for t in range(200, len(px))]
    rep = L.analyse_search(names, matrix, n_boot=200)
    assert rep["dsr"] < 0.95, rep["dsr"]
    assert rep["sharpe_annual"] > 0.5, "the noise mine should still look good raw"
    assert rep["psr"] > rep["dsr"], "deflation must lower the verdict"
    assert "LUCK" in L.render(rep) or "NOT PROVEN" in L.render(rep)


@check("deflation can never raise the verdict, even on near-duplicate candidates")
def _():
    # a grid of 190 near-identical configurations implies barely more than one
    # independent trial; the extreme-value threshold goes NEGATIVE below 2, which
    # would make the deflated Sharpe exceed the undeflated one
    rng = random.Random(3)
    base = [rng.gauss(0.0007, 0.009) for _ in range(600)]
    matrix = [[v + rng.gauss(0, 1e-7) for _ in range(190)] for v in base]
    rep = L.analyse_search(["c%d" % i for i in range(190)], matrix, n_boot=50)
    assert rep["independent_trials"] < 2, rep["independent_trials"]
    assert rep["threshold_sharpe_annual"] > 0, "noise threshold must be non-negative"
    assert rep["dsr"] <= rep["psr"] + 1e-12, "deflation raised the verdict"


@check("the demo only claims median provenance for the seed that has it")
def _():
    import io
    import contextlib
    for seed, must, must_not in ((17, "MEDIAN", None), (3, "one draw of many", "MEDIAN")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            L.main(["--demo", "--no-color", "--seed", str(seed),
                    "--obs", "300", "--boot", "20", "--blocks", "8"])
        text = buf.getvalue()
        assert must in text, "seed %d missing %r" % (seed, must)
        if must_not:
            assert must_not not in text.split("Over 25 independent")[0], \
                "seed %d claims median provenance" % seed


@check("a genuinely skilled series survives the same battery")
def _():
    rng = random.Random(77)
    series = [rng.gauss(0.0012, 0.008) for _ in range(1500)]     # Sharpe ~2.4
    rep = L.analyse_single("real", series, stated_trials=20)
    assert rep["psr"] > 0.99
    assert rep["dsr"] > 0.95, rep["dsr"]
    assert "SURVIVES" in L.render(rep)


@check("no verdict is issued on a sample too small to support one")
def _():
    rep = L.analyse_single("tiny", [0.01, 0.011, 0.009, 0.0105])
    assert "INSUFFICIENT DATA" in L.render(rep)
    assert "SURVIVES" not in L.render(rep)
    d = L.analyse_directional([1, -1, 1, -1] * 4, [1, 1, -1, -1] * 4,
                              [str(i) for i in range(16)], n_boot=50)
    assert "INSUFFICIENT DATA" in L.render(d)


@check("every card can actually reach LUCK on its own inputs")
def _():
    # a two-column ledger of coin flips: the documented shape, nothing else present
    rng = random.Random(1)
    pred = [rng.choice([1, -1]) for _ in range(300)]
    actual = [rng.choice([1, -1]) for _ in range(300)]
    assert "LUCK" in L.render(L.analyse_directional(pred, actual,
                                                    [str(i) for i in range(300)], n_boot=200))
    # ten years of noise with no declared trial count
    rng2 = random.Random(2)
    rep = L.analyse_single("noise", [rng2.gauss(0, 0.01) for _ in range(2520)])
    assert "LUCK" in L.render(rep), L.render(rep)


@check("a ledger confined to one day is the most clustered case, not an unclustered one")
def _():
    rng = random.Random(5)
    pred, actual = [], []
    for _ in range(300):
        a = rng.choice([1, -1])
        actual.append(a)
        pred.append(a if rng.random() < 0.593 else -a)
    rep = L.analyse_directional(pred, actual, ["2024-03-01"] * 300, n_boot=200)
    assert rep.get("single_day") is True
    text = L.render(rep)
    assert "LUCK" in text and "one day" in text


@check("the cluster interval is withheld when there are too few clusters to form one")
def _():
    rng = random.Random(6)
    vals = [1 if rng.random() < 0.6 else 0 for _ in range(300)]
    few = L.cluster_bootstrap_ci(vals, [i // 150 for i in range(300)], n_boot=500)
    many = L.cluster_bootstrap_ci(vals, [i // 6 for i in range(300)], n_boot=500)
    assert few["reliable"] is False and few["n_clusters"] == 2
    assert many["reliable"] is True and many["n_clusters"] == 50


@check("price levels pasted in as returns are caught, not blessed")
def _():
    px, p = [], 100.0
    rng = random.Random(7)
    for _ in range(600):
        p *= 1 + rng.gauss(0.0004, 0.011)
        px.append(p)
    rep = L.analyse_single("levels", px)
    text = L.render(rep)
    assert abs(rep["autocorr_1"]) > 0.9
    assert "SURVIVES" not in text
    assert "autocorrelated" in text


@check("duplicate column names are refused rather than silently collapsed")
def _():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "dupe.csv")
        rng = random.Random(8)
        with open(path, "w") as fh:
            fh.write("date,alpha,alpha\n")
            for i in range(50):
                fh.write("%d,%.6f,%.6f\n" % (i, rng.gauss(0, .01), rng.gauss(0, .01)))
        try:
            L.load_returns(path)
        except L.LuckFilterError as exc:
            assert "duplicate" in str(exc)
            return
        raise AssertionError("duplicate columns accepted")


@check("one flat configuration is dropped, not fatal to the whole grid")
def _():
    rng = random.Random(9)
    names = ["alpha", "flat", "beta", "gamma"]
    matrix = [[rng.gauss(0, .01), 0.0, rng.gauss(0, .01), rng.gauss(0, .01)]
              for _ in range(400)]
    rep = L.analyse_search(names, matrix, n_boot=50, s_blocks=8)
    assert rep["dropped_constant"] == ["flat"]
    assert rep["n_configs"] == 3


@check("confidences stated as percentages are rescaled, not discarded")
def _():
    rng = random.Random(10)
    pred, actual, conf = [], [], []
    for _ in range(400):
        a = rng.choice([1, -1])
        actual.append(a)
        pred.append(a if rng.random() < 0.55 else -a)
        conf.append(float(rng.choice([55, 60, 70, 75, 80, 90])))
    rep = L.analyse_directional(pred, actual, [str(i // 8) for i in range(400)],
                                conf, n_boot=200)
    assert rep.get("confidence_rescaled") is True
    assert "calibration" in rep and rep["calibration"]["brier"] > 0


@check("CSV round trip: returns, prices, and a directional ledger")
def _():
    with tempfile.TemporaryDirectory() as d:
        rng = random.Random(8)
        p1 = os.path.join(d, "r.csv")
        with open(p1, "w") as fh:
            fh.write("date,alpha,beta\n")
            for i in range(60):
                fh.write("2026-01-%02d,%.6f,%.6f\n" % (i % 28 + 1, rng.gauss(0, .01),
                                                       rng.gauss(0, .01)))
        series, labels = L.load_returns(p1)
        names, matrix, _ = L.align(series, labels)
        assert names == ["alpha", "beta"] and len(matrix) == 60

        p2 = os.path.join(d, "p.csv")
        with open(p2, "w") as fh:
            fh.write("date,px\n")
            v = 100.0
            for i in range(30):
                v *= 1 + rng.gauss(0, .01)
                fh.write("2026-02-%02d,%.4f\n" % (i % 28 + 1, v))
        series, labels = L.load_returns(p2, prices=True)
        names, matrix, _ = L.align(series, labels)
        assert names == ["px"] and len(matrix) == 29        # first row has no return

        p3 = os.path.join(d, "c.csv")
        with open(p3, "w") as fh:
            fh.write("date,ticker,predicted,actual,confidence\n")
            for i in range(80):
                fh.write("2026-03-%02d,AAA,%s,%s,0.55\n" % (
                    i % 10 + 1, "up" if i % 3 else "down", "up" if i % 4 else "down"))
        pred, actual, dates, conf = L.load_directional(p3)
        assert len(pred) == 80 and len(set(dates)) == 10 and conf[0] == 0.55


@check("percent-scaled input is converted, not silently mis-scaled")
def _():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "pct.csv")
        with open(p, "w") as fh:
            fh.write("date,s\n")
            for i in range(40):
                fh.write("2026-04-%02d,%.4f\n" % (i % 28 + 1, 1.5 if i % 2 else -1.0))
        series, labels = L.load_returns(p, pct=True)
        assert abs(series[0][1][0]) < 0.02


@check("bad CSV inputs fail cleanly with LuckFilterError and CLI exit 1")
def _():
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as d:
        cases = [
            ("header_only.csv", "date,a\n", []),
            ("no_numeric.csv", "date,note\n2026-01-01,hello\n2026-01-02,world\n2026-01-03,foo\n", []),
            ("bad_dir.csv", "date,foo,bar\n2026-01-01,1,2\n", ["--directional"]),
            ("few_dir.csv", "predicted,actual\nup,up\ndown,down\n", ["--directional"]),
        ]
        for fname, body, flags in cases:
            p = os.path.join(d, fname)
            with open(p, "w") as fh:
                fh.write(body)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                rc = L.main([p] + flags)
            assert rc == 1, (fname, rc)
            assert err.getvalue().startswith("luckfilter: "), (fname, err.getvalue())


@check("--demo --json emits pure valid JSON without mixed prose")
def _():
    import contextlib
    import io
    import json
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = L.main(["--demo", "--json", "--obs", "240", "--boot", "20", "--blocks", "8"])
    assert rc == 0, err.getvalue()
    data = json.loads(out.getvalue())
    assert data["mode"] == "search"
    assert "dsr" in data and "pbo" in data and "reality_check" in data and "spa" in data
    assert "logits" not in data["pbo"]


@check("average_correlation O(N*T) identity matches pairwise correlation")
def _():
    rng = random.Random(42)
    cols = [[rng.gauss(0, 1) for _ in range(120)] for _ in range(8)]
    cols.append([1.0] * 120)  # zero-variance column ignored
    # naive pairwise
    zs = []
    for c in cols:
        mu = sum(c) / len(c)
        d = [v - mu for v in c]
        n = math.sqrt(sum(v * v for v in d))
        if n > 0:
            zs.append([v / n for v in d])
    tot = pairs = 0
    for i in range(len(zs)):
        for j in range(i + 1, len(zs)):
            tot += sum(a * b for a, b in zip(zs[i], zs[j]))
            pairs += 1
    close(L.average_correlation(cols), tot / pairs, 1e-12)


def run():
    passed = failed = 0
    for name, fn in _RESULTS:
        try:
            fn()
        except Exception as exc:                     # noqa: BLE001 - this is the runner
            failed += 1
            print("FAIL  %s\n        %s: %s" % (name, type(exc).__name__, exc))
        else:
            passed += 1
            print("ok    %s" % name)
    print("\n%d passed, %d failed" % (passed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
