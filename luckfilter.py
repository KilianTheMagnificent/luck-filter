#!/usr/bin/env python3
"""Luck Filter - tell the difference between a strategy that works and one you found.

A backtest that looks good proves almost nothing on its own. If you tried 200
parameter combinations, the best of them looks good even when the data contains
no signal at all. This module implements the standard statistics for saying how
much of a result survives that fact:

    Probabilistic Sharpe Ratio      Bailey & Lopez de Prado (2012)
    Deflated Sharpe Ratio           Bailey & Lopez de Prado (2014)
    Minimum Track Record Length     Bailey & Lopez de Prado (2012)
    Probability of Backtest
      Overfitting (CSCV)            Bailey, Borwein, Lopez de Prado & Zhu (2016)
    Reality Check                   White (2000)
    Superior Predictive Ability     Hansen (2005)
    Stationary bootstrap            Politis & Romano (1994)
    Directional accuracy            Pesaran & Timmermann (1992)
    Market timing (exact)           Henriksson & Merton (1981)
    Brier decomposition             Murphy (1973)

Three deliberate departures from the cited papers, each marked in place: PBO averages
over tied in-sample winners rather than picking one, so the answer cannot depend on the
order the columns were passed in; the clustered directional statistic divides by
sqrt(design effect), an approximation not present in Pesaran & Timmermann; and the
autocorrelation adjustment extrapolates a single lag-1 estimate through an AR(1),
because eta at q=252 would otherwise need 251 autocorrelations.

Standard library only. No numpy, no scipy, no pandas, no install step.

Usage:
    python3 luckfilter.py --demo
    python3 luckfilter.py returns.csv --trials 200
    python3 luckfilter.py grid.csv                 # many columns = many candidates
    python3 luckfilter.py calls.csv --directional
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import statistics
import sys
from statistics import NormalDist

__version__ = "1.0.0"

EULER_GAMMA = 0.5772156649015329
_ND = NormalDist()

Z95 = 1.959963984540054          # two-sided 95%


class LuckFilterError(ValueError):
    """Raised when the input cannot support the statistic being asked for."""


# ---------------------------------------------------------------------------
# 1. Moments and Sharpe-ratio inference
# ---------------------------------------------------------------------------

def moments(r):
    """(T, mean, sample sd, skewness, RAW kurtosis) of a return series.

    The asymmetry is deliberate and matches the source papers: the standard
    deviation is Bessel-corrected (1/(T-1)) because the Sharpe ratio is defined
    from the sample sd, while the shape moments are population moments (1/T).
    Kurtosis is RAW, not excess: a Gaussian gives 3.0, not 0.0.
    """
    r = list(r)
    T = len(r)
    if T < 2:
        raise LuckFilterError("need at least 2 observations, got %d" % T)
    for i, v in enumerate(r):
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise LuckFilterError("observation %d is not a finite number: %r" % (i, v))
    if max(r) == min(r):
        raise LuckFilterError("returns have zero variance; the Sharpe ratio is undefined")
    mu = math.fsum(r) / T
    d = [x - mu for x in r]
    s2 = math.fsum(v * v for v in d)
    s3 = math.fsum(v * v * v for v in d)
    s4 = math.fsum(v * v * v * v for v in d)
    m2 = s2 / T
    if m2 <= 0 or m2 <= (1e-15 * mu) ** 2:
        raise LuckFilterError("returns have zero variance; the Sharpe ratio is undefined")
    return (T, mu, math.sqrt(s2 / (T - 1)), (s3 / T) / m2 ** 1.5, (s4 / T) / (m2 * m2))


def sharpe(r, rf_per_period=0.0):
    """Per-period Sharpe ratio of an excess-return series."""
    x = [v - rf_per_period for v in r]
    T, mu, sd, _g3, _g4 = moments(x)
    return mu / sd


def variance_factor(sr, g3, g4, ar1=0.0):
    """The bracket shared by the SE, PSR, DSR and MinTRL formulas.

        1 - g3*SR + ((g4 - 1)/4)*SR^2

    SR must be PER-PERIOD and g4 must be RAW kurtosis. Passing an annualised
    Sharpe or an excess kurtosis here is the single most common way to get a
    confidently wrong answer out of this family of statistics, so the failure
    is loud rather than silent.
    """
    f = 1.0 - g3 * sr + ((g4 - 1.0) / 4.0) * sr * sr
    if ar1:
        # the formula assumes independent observations. Smoothed, illiquid or
        # overlapping-position returns are not, and without this the nominal 5%
        # test fires 11% of the time at a first-order autocorrelation of 0.3.
        r = min(0.95, max(-0.95, ar1))
        f *= (1.0 + r) / (1.0 - r)
    if f <= 0.0:
        raise LuckFilterError(
            "variance factor is non-positive (%.6g). Check that kurtosis is RAW "
            "(3.0 for a Gaussian, not 0.0) and that the Sharpe ratio is "
            "per-period, not annualised." % f)
    return f


def sharpe_std_error(sr, T, g3, g4, ar1=0.0):
    """Standard error of the per-period Sharpe ratio, adjusted for skew and fat
    tails (Mertens 2002; Lo 2002). Reduces to sqrt((1+SR^2/2)/(T-1)) for a Gaussian.

    Note the denominator: Lo divides by T, Bailey & Lopez de Prado by T-1. We follow
    the latter so this standard error and the PSR share one convention; the difference
    is a factor of sqrt(T/(T-1)).
    """
    return math.sqrt(variance_factor(sr, g3, g4, ar1) / (T - 1))


def psr(sr, sr_star, T, g3, g4, ar1=0.0):
    """Probabilistic Sharpe Ratio: P(true Sharpe > sr_star) given the observed
    moments. Everything per-period."""
    if sr == sr_star:
        return 0.5
    z = (sr - sr_star) * math.sqrt(T - 1) / math.sqrt(variance_factor(sr, g3, g4, ar1))
    return _ND.cdf(z)


def min_track_record_length(sr, sr_star, g3, g4, alpha=0.95, ar1=0.0):
    """Observations needed before PSR(sr_star) would reach `alpha`, holding the
    observed moments fixed. Infinite when the point estimate is already at or
    below the benchmark."""
    if not 0.5 < alpha < 1.0:
        raise LuckFilterError(
            "alpha must lie strictly between 0.5 and 1; the inversion squares "
            "Z(alpha), so alpha=0.3 would silently return the alpha=0.7 answer")
    if sr <= sr_star:
        return float("inf")
    return (1.0 + variance_factor(sr, g3, g4, ar1)
            * (_ND.inv_cdf(alpha) / (sr - sr_star)) ** 2)


def expected_max_sharpe(v, n_trials, mean_sr=0.0):
    """Expected best-of-N Sharpe ratio when NONE of the N trials has any skill.

    `v` is the variance of Sharpe ratios ACROSS trials (per-period units), not
    the sampling variance of one strategy's Sharpe. This is the number an
    honest backtest has to beat.
    """
    if v < 0:
        raise LuckFilterError("variance across trials cannot be negative")
    if n_trials <= 1:
        raise LuckFilterError(
            "n_trials must exceed 1; n_trials=1 asserts a single pre-registered "
            "hypothesis, for which the right statistic is psr(sr_star=0)")
    if v == 0:
        return mean_sr
    a = _ND.inv_cdf(1.0 - 1.0 / n_trials)
    b = _ND.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return mean_sr + math.sqrt(v) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(sr, T, g3, g4, n_trials, v, ar1=0.0):
    """Deflated Sharpe Ratio: the PSR with the benchmark raised from zero to the
    Sharpe you would expect from the best of `n_trials` worthless strategies.
    Returns (dsr, sr_threshold)."""
    sr0 = expected_max_sharpe(v, n_trials)
    return psr(sr, sr0, T, g3, g4, ar1), sr0


def breakeven_trials(sr, T, g3, g4, v, alpha=0.95, hi=1000000, ar1=0.0):
    """Smallest number of independent trials at which the DSR drops below
    `alpha` - i.e. how much searching it would take to explain this result away.
    None if even 2 trials already sink it, or if no amount of searching does."""
    if deflated_sharpe(sr, T, g3, g4, 2, v, ar1)[0] < alpha:
        return 2
    if deflated_sharpe(sr, T, g3, g4, hi, v, ar1)[0] >= alpha:
        return None
    lo = 2
    while lo + 1 < hi:                                   # bisect on an integer
        mid = (lo + hi) // 2
        if deflated_sharpe(sr, T, g3, g4, mid, v, ar1)[0] >= alpha:
            lo = mid
        else:
            hi = mid
    return hi


def autocorrelation(r, lag):
    """Sample autocorrelation at `lag`, biased (1/T) normalisation so the
    implied autocovariance sequence stays positive semi-definite."""
    T = len(r)
    mu = math.fsum(r) / T
    d = [v - mu for v in r]
    den = math.fsum(v * v for v in d)
    if den <= 0:
        raise LuckFilterError("returns have zero variance")
    return math.fsum(d[t] * d[t - lag] for t in range(lag, T)) / den


def eta(q, rhos):
    """Lo (2002) time-aggregation factor. Replaces the naive sqrt(q) when
    returns are serially correlated: positive autocorrelation makes sqrt(q)
    overstate the annualised Sharpe ratio."""
    if q < 1 or int(q) != q:
        raise LuckFilterError("q must be a positive integer")
    q = int(q)
    if q == 1:
        return 1.0
    if len(rhos) < q - 1:
        raise LuckFilterError("need %d autocorrelations for q=%d" % (q - 1, q))
    s = q + 2.0 * math.fsum((q - k) * rhos[k - 1] for k in range(1, q))
    if s <= 0:
        raise LuckFilterError("implied multi-period variance is non-positive")
    return q / math.sqrt(s)


def eta_ar1(q, rho):
    """Closed form of `eta` when returns follow an AR(1)."""
    if not -1.0 < rho < 1.0:
        raise LuckFilterError("rho must lie strictly between -1 and 1")
    if rho == 0.0:
        return math.sqrt(q)
    inner = 1.0 + (2.0 * rho / (1.0 - rho)) * (1.0 - (1.0 - rho ** q) / (q * (1.0 - rho)))
    if inner <= 0:
        raise LuckFilterError("implied multi-period variance is non-positive")
    return math.sqrt(q) * inner ** -0.5


def average_correlation(columns):
    """Mean pairwise correlation between candidate strategies' return series."""
    m = len(columns)
    if m < 2:
        return 0.0
    z = []
    for col in columns:
        mu = math.fsum(col) / len(col)
        d = [v - mu for v in col]
        n = math.sqrt(math.fsum(v * v for v in d))
        z.append([v / n for v in d] if n > 0 else None)
    total, pairs = 0.0, 0
    for i in range(m):
        if z[i] is None:
            continue
        for j in range(i + 1, m):
            if z[j] is None:
                continue
            total += math.fsum(a * b for a, b in zip(z[i], z[j]))
            pairs += 1
    return total / pairs if pairs else 0.0


def implied_independent_trials(m_trials, rho):
    """Bailey & Lopez de Prado (2014) eq. 9: how many genuinely independent
    trials `m_trials` correlated ones are worth. Neighbouring cells of a
    parameter grid are near-duplicates, so the honest N is well below the
    nominal count."""
    if m_trials <= 1:
        return 1.0
    lo = -1.0 / (m_trials - 1)
    rho = min(1.0, max(lo, rho))
    return min(float(m_trials), max(1.0, rho + (1.0 - rho) * m_trials))


# ---------------------------------------------------------------------------
# 2. Probability of Backtest Overfitting, via combinatorially symmetric
#    cross-validation (Bailey, Borwein, Lopez de Prado & Zhu)
# ---------------------------------------------------------------------------

def _midranks(values):
    """Ascending ranks 1..N, averaging ties. Ties must be averaged or the answer
    would depend on the order the caller happened to pass the columns in."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # matched with a tolerance so the grouping cannot depend on the last bit
        # of an intermediate sum, which differs between algebraically identical routes
        while j + 1 < n and math.isclose(values[order[j + 1]], values[order[i]],
                                         rel_tol=1e-12, abs_tol=1e-15):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _sharpe_from_sums(n_obs, a, q, shift, ddof):
    """Sharpe ratio of a block set, recovered from that set's sum of centred
    returns `a` and sum of squared centred returns `q`. Exact, and lets every
    split cost O(blocks) instead of O(observations)."""
    mean = a / n_obs + shift
    ss = q - a * a / n_obs
    if ss < 0.0:
        ss = 0.0
    var = ss / (n_obs - ddof)
    sd = math.sqrt(var) if var > 0.0 else 0.0
    # Undoing the shift leaves a rounding residue of order 1e-17 where the true
    # mean is exactly zero. On discrete payoffs - a prediction ledger, an
    # integer-scored grid - that is common, and it flips the "is this block set
    # under water" test on pure noise. Snap it.
    if abs(mean) <= 1e-13 * (abs(shift) + sd):
        mean = 0.0
    if var <= 0.0:
        return 0.0 if mean == 0 else math.copysign(float("inf"), mean)
    return mean / sd


def probability_of_backtest_overfitting(matrix, s_blocks=16, ddof=1, loss_threshold=0.0):
    """How often the configuration that looked best in-sample lands in the
    bottom half out-of-sample.

    `matrix` is rows = time, columns = candidate configurations. Splits the
    sample into `s_blocks` blocks and walks every balanced train/test split.
    0.5 means the selection carried no information at all - a coin flip.

    Returns a dict with the PBO, the performance-degradation regression, and
    the probability that the selected configuration loses money out of sample.
    """
    if not matrix or not matrix[0]:
        raise LuckFilterError("matrix must be a non-empty T x N grid, rows = time")
    T, N = len(matrix), len(matrix[0])
    if N < 2:
        raise LuckFilterError(
            "PBO needs at least 2 configurations; with one there is no selection to overfit")
    if s_blocks % 2 or s_blocks < 4:
        raise LuckFilterError("s_blocks must be an even integer >= 4")
    for t, row in enumerate(matrix):
        if len(row) != N:
            raise LuckFilterError("row %d has %d columns, expected %d" % (t, len(row), N))
        for n, v in enumerate(row):
            if not math.isfinite(v):
                raise LuckFilterError("matrix[%d][%d] is not finite" % (t, n))

    block = T // s_blocks
    if block < 1:
        raise LuckFilterError("T=%d is smaller than s_blocks=%d" % (T, s_blocks))
    used = block * s_blocks
    offset = T - used                                     # drop the OLDEST rows

    shift = []
    for n in range(N):
        col = [matrix[offset + t][n] for t in range(used)]
        if max(col) == min(col):
            raise LuckFilterError("column %d is constant; it is not a strategy" % n)
        shift.append(math.fsum(col) / used)

    a_blocks, q_blocks = [], []
    for s in range(s_blocks):
        arow, qrow = [], []
        for n in range(N):
            ys = [matrix[offset + s * block + i][n] - shift[n] for i in range(block)]
            arow.append(math.fsum(ys))
            qrow.append(math.fsum(y * y for y in ys))
        a_blocks.append(arow)
        q_blocks.append(qrow)
    a_total = [math.fsum(a_blocks[s][n] for s in range(s_blocks)) for n in range(N)]
    q_total = [math.fsum(q_blocks[s][n] for s in range(s_blocks)) for n in range(N)]

    half = block * (s_blocks // 2)
    logits, is_perf, oos_perf = [], [], []
    for comb in itertools.combinations(range(s_blocks), s_blocks // 2):
        a_is = [math.fsum(a_blocks[s][n] for s in comb) for n in range(N)]
        q_is = [math.fsum(q_blocks[s][n] for s in comb) for n in range(N)]
        r_is, r_oos = [], []
        for n in range(N):
            r_is.append(_sharpe_from_sums(half, a_is[n], q_is[n], shift[n], ddof))
            r_oos.append(_sharpe_from_sums(half, a_total[n] - a_is[n],
                                           q_total[n] - q_is[n], shift[n], ddof))
        # Ties for in-sample best are routine once payoffs are discrete (a
        # prediction ledger, an integer-scored grid), and breaking them by
        # column index would make the answer depend on the order the caller
        # happened to pass the configurations in. Average over the tied set.
        top = max(r_is)
        # Exact float equality would make the tied set depend on the last bit of
        # an intermediate sum, so two algebraically identical routes disagree on
        # binary or integer payoffs. Compare with a tolerance instead.
        tied = [n for n in range(N)
                if math.isclose(r_is[n], top, rel_tol=1e-12, abs_tol=1e-15)]
        ranks = _midranks(r_oos)
        omega = math.fsum(ranks[n] for n in tied) / (len(tied) * (N + 1.0))
        logits.append(math.log(omega / (1.0 - omega)))
        is_perf.append(top)
        oos_perf.append(math.fsum(r_oos[n] for n in tied) / len(tied))

    n_splits = len(logits)
    pairs = [(x, y) for x, y in zip(is_perf, oos_perf) if math.isfinite(x) and math.isfinite(y)]
    slope = intercept = r2 = float("nan")
    if len(pairs) >= 2:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        xb, yb = math.fsum(xs) / len(xs), math.fsum(ys) / len(ys)
        sxx = math.fsum((x - xb) ** 2 for x in xs)
        sxy = math.fsum((x - xb) * (y - yb) for x, y in pairs)
        syy = math.fsum((y - yb) ** 2 for y in ys)
        if sxx > 0:
            slope = sxy / sxx
            intercept = yb - slope * xb
            r2 = (slope * slope * sxx / syy) if syy > 0 else float("nan")

    return {
        "pbo": sum(1 for x in logits if x <= 0.0) / n_splits,
        "n_splits": n_splits,
        "n_strategies": N,
        "n_blocks": s_blocks,
        "block_size": block,
        "n_rows_dropped": offset,
        "logits": logits,
        "is_perf": is_perf,
        "oos_perf": oos_perf,
        "degradation_slope": slope,
        "degradation_intercept": intercept,
        "degradation_r2": r2,
        "prob_loss": sum(1 for v in oos_perf if v < loss_threshold) / n_splits,
    }


# ---------------------------------------------------------------------------
# 3. Bootstrap tests for data snooping
# ---------------------------------------------------------------------------

def stationary_bootstrap_indices(T, p, rng):
    """Politis & Romano (1994). Resamples geometric-length blocks with circular
    wrapping, so the resample keeps the serial dependence of the original and
    every index stays marginally uniform."""
    if not 0.0 < p <= 1.0:
        raise LuckFilterError("p must lie in (0, 1]")
    idx = [rng.randrange(T)]
    for _ in range(T - 1):
        idx.append(rng.randrange(T) if rng.random() < p else (idx[-1] + 1) % T)
    return idx


def _autocov(d, den_T):
    n = len(d)
    return [math.fsum(d[t] * d[t + i] for t in range(n - i)) / den_T for i in range(n)]


def optimal_block_length(x):
    """Politis & White (2004) with the Patton, Politis & White (2009)
    correction. Returns the stationary-bootstrap parameter p = 1/b; p = 1 means
    no detectable serial dependence, i.e. a plain iid bootstrap."""
    T = len(x)
    if T < 8:
        return 1.0
    mu = math.fsum(x) / T
    d = [v - mu for v in x]
    g = _autocov(d, T)
    if g[0] <= 0:
        return 1.0
    rho = [v / g[0] for v in g]

    k_t = max(5, int(math.ceil(math.sqrt(math.log10(T)))))
    crit = 2.0 * math.sqrt(math.log10(T) / T)
    m_hat = 0
    for k in range(1, T):
        if k + k_t >= T:
            break
        if all(abs(rho[k + j]) < crit for j in range(1, k_t + 1)):
            m_hat = k
            break
    big_m = min(2 * max(m_hat, 1), int(math.ceil(math.sqrt(T))) + k_t, T - 1)

    def flat_top(t):
        return min(1.0, max(0.0, 2.0 * (1.0 - abs(t))))

    g_hat = 2.0 * math.fsum(flat_top(k / big_m) * k * g[k] for k in range(1, big_m + 1))
    s_inf = g[0] + 2.0 * math.fsum(flat_top(k / big_m) * g[k] for k in range(1, big_m + 1))
    if g_hat == 0.0 or s_inf <= 0.0:
        return 1.0
    b = (2.0 * g_hat * g_hat / (2.0 * s_inf * s_inf)) ** (1.0 / 3.0) * T ** (1.0 / 3.0)
    b = min(b, math.ceil(min(3.0 * math.sqrt(T), T / 3.0)))
    return 1.0 / max(1.0, b)


def stationary_bootstrap_variance(x, p):
    """Long-run variance of sqrt(T) * mean(x) under the stationary bootstrap.
    This is the studentiser in Hansen's SPA."""
    T = len(x)
    mu = math.fsum(x) / T
    d = [v - mu for v in x]
    g = _autocov(d, T)
    total = g[0]
    for i in range(1, T):
        kappa = (1.0 - i / T) * (1.0 - p) ** i + (i / T) * (1.0 - p) ** (T - i)
        total += 2.0 * kappa * g[i]
    return total


def reality_check(columns, n_boot=2000, p=None, seed=0):
    """White's (2000) Reality Check.

    `columns` are the candidate strategies' per-period performance RELATIVE to
    the benchmark. Tests the null that the best of them is no better than the
    benchmark, using the whole search as the unit of analysis rather than the
    winner alone. The p-value answers: if none of these worked, how often would
    the best of them look this good?
    """
    N = len(columns)
    if N < 1:
        raise LuckFilterError("need at least one candidate strategy")
    T = len(columns[0])
    if any(len(c) != T for c in columns):
        raise LuckFilterError("all candidate columns must have the same length")
    if p is None:
        # the longest block any candidate demands: the winner may look iid while
        # its siblings do not, and too short a block narrows the bootstrap null
        p = min(optimal_block_length(c) for c in columns)
    rng = random.Random(seed)
    means = [math.fsum(c) / T for c in columns]
    root_t = math.sqrt(T)
    v_obs = max(root_t * m for m in means)

    hits = 0
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(T, p, rng)
        v_star = max(root_t * (math.fsum(c[i] for i in idx) / T - m)
                     for c, m in zip(columns, means))
        if v_star >= v_obs:
            hits += 1
    return {"p_value": hits / n_boot, "statistic": v_obs, "n_boot": n_boot,
            "block_p": p, "best": max(range(N), key=lambda k: means[k])}


def superior_predictive_ability(columns, n_boot=2000, p=None, seed=0):
    """Hansen's (2005) SPA test - the Reality Check's successor.

    Studentises each candidate and, in the `consistent` variant, stops letting
    hopeless candidates depress the null distribution. That matters because
    padding a search with junk makes White's Reality Check easier to pass. A
    wide gap between the lower and upper p-values means the verdict depends on
    how much junk was in the pool.
    """
    N = len(columns)
    if N < 1:
        raise LuckFilterError("need at least one candidate strategy")
    T = len(columns[0])
    if any(len(c) != T for c in columns):
        raise LuckFilterError("all candidate columns must have the same length")
    if p is None:
        # the longest block any candidate demands: the winner may look iid while
        # its siblings do not, and too short a block narrows the bootstrap null
        p = min(optimal_block_length(c) for c in columns)
    rng = random.Random(seed)
    means = [math.fsum(c) / T for c in columns]
    omega = []
    for c in columns:
        v = stationary_bootstrap_variance(c, p)
        omega.append(math.sqrt(v) if v > 0 else 0.0)
    root_t = math.sqrt(T)
    stat = max(0.0, max((root_t * m / w if w > 0 else 0.0) for m, w in zip(means, omega)))

    a_t = math.sqrt(2.0 * math.log(math.log(T))) if T > math.e else 0.0
    g_lower = [max(m, 0.0) for m in means]
    g_cons = [m if (w > 0 and root_t * m / w >= -a_t) else (m if w == 0 else 0.0)
              for m, w in zip(means, omega)]
    g_upper = list(means)

    counts = [0, 0, 0]
    for _ in range(n_boot):
        idx = stationary_bootstrap_indices(T, p, rng)
        boot = [math.fsum(c[i] for i in idx) / T for c in columns]
        for slot, g in enumerate((g_lower, g_cons, g_upper)):
            t_star = 0.0
            for k in range(N):
                if omega[k] > 0:
                    val = root_t * (boot[k] - g[k]) / omega[k]
                    if val > t_star:
                        t_star = val
            if t_star >= stat:
                counts[slot] += 1
    return {"statistic": stat, "p_lower": counts[0] / n_boot,
            "p_consistent": counts[1] / n_boot, "p_upper": counts[2] / n_boot,
            "n_boot": n_boot, "block_p": p}


def cluster_bootstrap_ci(values, clusters, n_boot=4000, alpha=0.05, seed=0):
    """Confidence interval for a mean when observations come in correlated
    groups - 40 predictions made on the same day are not 40 independent facts.

    Resamples whole clusters. Reports a BCa interval, falling back to the
    percentile interval when the acceleration is undefined.
    """
    if len(values) != len(clusters):
        raise LuckFilterError("values and clusters must be the same length")
    groups = {}
    for v, g in zip(values, clusters):
        s, n = groups.get(g, (0.0, 0))
        groups[g] = (s + v, n + 1)
    keys = list(groups)
    G = len(keys)
    if G < 2:
        raise LuckFilterError("need at least 2 clusters")
    sums = [groups[k][0] for k in keys]
    sizes = [groups[k][1] for k in keys]
    total, n_all = math.fsum(sums), sum(sizes)
    theta = total / n_all

    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = n = 0.0
        for _ in range(G):
            j = rng.randrange(G)
            s += sums[j]
            n += sizes[j]
        boots.append(s / n)
    boots.sort()

    def quantile(q):
        if q <= 0:
            return boots[0]
        if q >= 1:
            return boots[-1]
        pos = q * (len(boots) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(boots) - 1)
        return boots[lo] + (pos - lo) * (boots[hi] - boots[lo])

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    method = "percentile"
    below = sum(1 for b in boots if b < theta)
    if 0 < below < n_boot:
        z0 = _ND.inv_cdf(below / n_boot)
        jack = [(total - sums[g]) / (n_all - sizes[g]) for g in range(G)
                if n_all - sizes[g] > 0]
        if len(jack) == G:
            jbar = math.fsum(jack) / G
            u = [jbar - v for v in jack]
            d2 = math.fsum(v * v for v in u)
            d3 = math.fsum(v ** 3 for v in u)
            if d2 > 0:
                acc = d3 / (6.0 * d2 ** 1.5)
                out = []
                for z in (_ND.inv_cdf(alpha / 2.0), _ND.inv_cdf(1.0 - alpha / 2.0)):
                    denom = 1.0 - acc * (z0 + z)
                    if denom == 0:
                        out = []
                        break
                    out.append(_ND.cdf(z0 + (z0 + z) / denom))
                if len(out) == 2 and all(0.0 < v < 1.0 for v in out):
                    lo_q, hi_q, method = out[0], out[1], "BCa"
    return {"estimate": theta, "lo": quantile(lo_q), "hi": quantile(hi_q),
            "method": method, "n_clusters": G, "n_obs": n_all,
            # measured coverage of the nominal 95% interval: 0.39 at G=2, 0.85 at
            # G=5, 0.92 at G=10, 0.96 at G=50. Below ~10 it is not an interval.
            "reliable": G >= 10, "marginal": G < 25}


# ---------------------------------------------------------------------------
# 4. Directional calls and probability forecasts
# ---------------------------------------------------------------------------

def contingency(pred, actual):
    """2x2 table of directional calls. Returns (a, b, c, d) =
    (up/up, up-actual & down-call, down-actual & up-call, down/down)."""
    if len(pred) != len(actual):
        raise LuckFilterError("pred and actual must be the same length")
    a = b = c = d = 0
    for p, y in zip(pred, actual):
        if y > 0:
            a, b = (a + 1, b) if p > 0 else (a, b + 1)
        else:
            c, d = (c + 1, d) if p > 0 else (c, d + 1)
    return a, b, c, d


def pesaran_timmermann(pred, actual):
    """Pesaran & Timmermann (1992) test of directional accuracy.

    A raw hit rate is not evidence. In a market that rises 58% of the time, a
    forecaster who says "up" every single day is right 58% of the time while
    carrying no information whatsoever. This test compares the hit rate against
    the one your own directional bias earns for free, so a constant forecaster
    scores exactly zero.
    """
    a, b, c, d = contingency(pred, actual)
    n = a + b + c + d
    n1, n2 = a + b, c + d                      # actual up / actual down
    k1, k2 = a + c, b + d                      # called up / called down
    hit = (a + d) / n
    if min(n1, n2, k1, k2) == 0:
        return {"n": n, "hit_rate": hit, "expected_hit_rate": hit, "statistic": None,
                "p_value": None, "table": (a, b, c, d), "degenerate": True,
                "note": "one margin is empty - the forecast or the outcome never varied, "
                        "so no directional information can exist"}
    p_star = (n1 * k1 + n2 * k2) / float(n * n)
    s = n * (a * d - b * c) / math.sqrt(float((n - 1)) * n1 * n2 * k1 * k2)
    return {"n": n, "hit_rate": hit, "expected_hit_rate": p_star, "statistic": s,
            "p_value": 1.0 - _ND.cdf(s), "table": (a, b, c, d), "degenerate": False}


def _log_comb(n, k):
    """log C(n, k). math.comb is exact but its result overflows float conversion for
    n in the thousands, which a real prediction ledger reaches easily."""
    if k < 0 or k > n:
        return float("-inf")
    return (math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0))


def _sum_exp(logs):
    """Sum probabilities given as logs, skipping terms that underflow to zero."""
    return math.fsum(math.exp(v) for v in logs if v > -745.0)


def henriksson_merton(pred, actual):
    """Exact market-timing test (Henriksson & Merton 1981), which is Fisher's
    exact test on the same table. Use it when the sample is small or a margin
    is thin, where the asymptotic test is unreliable."""
    a, b, c, d = contingency(pred, actual)
    n = a + b + c + d
    n1, n2, k1 = a + b, c + d, a + c
    if min(n1, n2, k1, b + d) == 0:
        return {"p_value": None, "degenerate": True, "table": (a, b, c, d)}
    hi = min(n1, k1)
    log_den = _log_comb(n, k1)
    p = _sum_exp([_log_comb(n1, v) + _log_comb(n2, k1 - v) - log_den
                  for v in range(a, hi + 1)])
    return {"p_value": min(1.0, p), "degenerate": False, "table": (a, b, c, d),
            "expected_a": k1 * n1 / n}


def binomial_test(successes, n, p0):
    """Exact one-sided binomial p-value, P(X >= successes) under rate p0."""
    if not 0.0 <= p0 <= 1.0:
        raise LuckFilterError("p0 must lie in [0, 1]")
    if p0 == 0.0:
        return 0.0 if successes > 0 else 1.0
    if p0 == 1.0:
        return 1.0 if successes >= n else 0.0
    lp, lq = math.log(p0), math.log1p(-p0)
    return min(1.0, _sum_exp([_log_comb(n, k) + k * lp + (n - k) * lq
                              for k in range(successes, n + 1)]))


def design_effect(hits, groups):
    """How much less information a clustered sample carries.

    40 calls made on one morning share a market. When it rallies, most "up"
    calls are right at once, so those 40 rows are worth far fewer than 40
    independent observations. Returns the intracluster correlation, the design
    effect, and the effective sample size that should replace the raw count in
    every standard error.
    """
    if len(hits) != len(groups):
        raise LuckFilterError("hits and groups must be the same length")
    by = {}
    for h, g in zip(hits, groups):
        by.setdefault(g, []).append(float(h))
    k = len(by)
    sizes = [len(v) for v in by.values()]
    n = sum(sizes)
    if k < 2:
        raise LuckFilterError("need at least 2 clusters to estimate clustering")
    grand = math.fsum(math.fsum(v) for v in by.values()) / n
    ssb = math.fsum(len(v) * (math.fsum(v) / len(v) - grand) ** 2 for v in by.values())
    ssw = math.fsum(math.fsum((x - math.fsum(v) / len(v)) ** 2 for x in v) for v in by.values())
    msb = ssb / (k - 1)
    msw = ssw / (n - k) if n > k else 0.0
    m0 = (n - math.fsum(s * s for s in sizes) / n) / (k - 1)
    denom = msb + (m0 - 1.0) * msw
    rho = 0.0 if denom <= 0 else (msb - msw) / denom
    rho = min(1.0, max(0.0, rho))
    m_star = math.fsum(s * s for s in sizes) / n
    deff = 1.0 + (m_star - 1.0) * rho
    return {"rho": rho, "deff": deff, "n_eff": n / deff, "n_obs": n, "n_clusters": k,
            "mean_cluster_size": n / k, "kish_cluster_size": m_star}


def brier_decomposition(probs, outcomes, n_bins=10):
    """Brier score split into calibration, resolution and irreducible
    uncertainty (Murphy 1973), plus the two within-bin terms that make the
    identity exact for continuous forecasts.

    Reliability is a penalty - small is good, and it measures whether "70%"
    actually happens 70% of the time. Resolution is a credit - large is good,
    and it measures whether the forecasts separate outcomes at all. Uncertainty
    belongs to the data, not the forecaster.
    """
    if len(probs) != len(outcomes):
        raise LuckFilterError("probs and outcomes must be the same length")
    n = len(probs)
    if n == 0:
        raise LuckFilterError("no forecasts to score")
    for p in probs:
        if not 0.0 <= p <= 1.0:
            raise LuckFilterError("forecast probabilities must lie in [0, 1]")
    o = [1.0 if v else 0.0 for v in outcomes]
    bs = math.fsum((p - y) ** 2 for p, y in zip(probs, o)) / n
    obar = math.fsum(o) / n

    # Binning by unique value is exact, but only informative when each value
    # recurs. With one observation per bin the observed frequency IS the
    # outcome, so reliability collapses onto the Brier score and resolution
    # onto the uncertainty - a catastrophic-miscalibration reading that is a
    # pure artefact of the binning.
    counts = {}
    for pr in probs:
        r = round(pr, 12)
        counts[r] = counts.get(r, 0) + 1
    if len(counts) <= 20 and min(counts.values()) >= 5:
        key = lambda p: round(p, 12)
    else:
        n_bins = n_bins if n >= 100 else 5
        key = lambda p: min(n_bins - 1, int(p * n_bins))
    bins = {}
    for p, y in zip(probs, o):
        bins.setdefault(key(p), []).append((p, y))
    informative = min(len(v) for v in bins.values()) >= 5

    rel = res = wbv = wbc = 0.0
    for members in bins.values():
        nk = len(members)
        fk = math.fsum(p for p, _ in members) / nk
        ok = math.fsum(y for _, y in members) / nk
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - obar) ** 2
        wbv += math.fsum((p - fk) ** 2 for p, _ in members)
        wbc += 2.0 * math.fsum((p - fk) * (y - ok) for p, y in members)
    rel, res, wbv, wbc = rel / n, res / n, wbv / n, wbc / n
    unc = obar * (1.0 - obar)
    return {"brier": bs,
            "reliability": rel if informative else None,
            "resolution": res if informative else None,
            "uncertainty": unc,
            "within_bin_variance": wbv, "within_bin_covariance": wbc,
            "residual": bs - (rel - res + unc + wbv - wbc),
            "skill_score": (1.0 - bs / unc) if unc > 0 else None,
            "n_bins": len(bins), "informative": informative,
            "note": None if informative else
                    "too few forecasts per probability bin to separate calibration "
                    "from resolution; only the Brier score and skill score are reported"}
# ---------------------------------------------------------------------------
# Synthetic search space: a realistic parameter grid over data with NO signal.
# ---------------------------------------------------------------------------

def random_walk(n, seed, drift=0.0, vol=0.011):
    """Geometric random walk. drift=0.0 means there is no signal to find."""
    rng = random.Random(seed)
    px, p = [], 100.0
    for _ in range(n):
        p *= 1.0 + rng.gauss(drift, vol)
        px.append(p)
    return px


def _sma(px, n):
    out, s = [None] * len(px), 0.0
    for i, p in enumerate(px):
        s += p
        if i >= n:
            s -= px[i - n]
        if i >= n - 1:
            out[i] = s / n
    return out


def _rolling_sd(px, n):
    out = [None] * len(px)
    for i in range(n - 1, len(px)):
        out[i] = statistics.pstdev(px[i - n + 1:i + 1])
    return out


def _apply(rets, pos):
    """Position at t-1 earns the return at t. No look-ahead."""
    return [0.0] + [rets[i] * pos[i - 1] for i in range(1, len(rets))]


def strategy_grid(px):
    """The kind of parameter sweep a researcher actually runs: three strategy
    families, long-short and long-only, ~190 configurations in total."""
    rets = [0.0] + [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))]
    out = {}
    for ls in (True, False):
        tag = "ls" if ls else "lo"
        for fast in (3, 5, 8, 10, 13, 15, 20, 25):
            for slow in (30, 40, 50, 60, 80, 100, 120, 150):
                a, b = _sma(px, fast), _sma(px, slow)
                pos = [0.0] * len(px)
                for i in range(len(px)):
                    if a[i] is not None and b[i] is not None:
                        pos[i] = 1.0 if a[i] > b[i] else (-1.0 if ls else 0.0)
                out["cross_%d_%d_%s" % (fast, slow, tag)] = _apply(rets, pos)
        for win in (10, 15, 20, 30, 45, 60):
            m, sd = _sma(px, win), _rolling_sd(px, win)
            for k in (0.5, 1.0, 1.5, 2.0):
                pos = [0.0] * len(px)
                for i in range(len(px)):
                    if m[i] is None or not sd[i]:
                        continue
                    z = (px[i] - m[i]) / sd[i]
                    if z > k:
                        pos[i] = -1.0 if ls else 0.0
                    elif z < -k:
                        pos[i] = 1.0
                out["revert_%d_%.1f_%s" % (win, k, tag)] = _apply(rets, pos)
        for look in (5, 10, 20, 40, 60, 90, 120):
            pos = [0.0] * len(px)
            for i in range(look, len(px)):
                up = px[i] > px[i - look]
                pos[i] = 1.0 if up else (-1.0 if ls else 0.0)
            out["mom_%d_%s" % (look, tag)] = _apply(rets, pos)
    return out
# ---------------------------------------------------------------------------
# Loading real data
# ---------------------------------------------------------------------------

_DATE_NAMES = ("date", "session_date", "day", "dt", "timestamp", "time", "period")
_PRED_NAMES = ("predicted", "pred", "forecast", "direction", "signal", "call")
_ACTUAL_NAMES = ("actual", "realized", "realised", "outcome", "truth", "actual_dir")
# deliberately NOT "prob"/"p_up": those mean P(market up), which for a bearish call
# is the complement of the confidence that the call itself is right
_CONF_NAMES = ("confidence", "conf", "score")


def _norm(s):
    return s.strip().lower().replace(" ", "_")


def _pick(fields, names):
    for want in names:
        for f in fields:
            if _norm(f) == want:
                return f
    for want in names:
        for f in fields:
            if want in _norm(f):
                return f
    return None


def _to_float(v):
    v = (v or "").strip().replace(",", "").replace("%", "")
    if v in ("", "na", "NA", "nan", "NaN", "null", "None", "-"):
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def read_table(path):
    """Read a CSV/TSV into (fieldnames, list of row dicts). Tolerates tabs and
    semicolons, blank lines, and a trailing empty column."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(fh, dialect)
        try:
            raw_header = next(reader)
        except StopIteration:
            raise LuckFilterError("%s is empty" % path)
        seen = [h.strip() for h in raw_header if h.strip()]
        if len(seen) != len(set(seen)):
            dupes = sorted(set(h for h in seen if seen.count(h) > 1))
            raise LuckFilterError(
                "%s has duplicate column names (%s); they would silently collapse into "
                "one and the others would vanish from the search" % (path, ", ".join(dupes)))
        fh.seek(0)
        rows = [r for r in csv.DictReader(fh, dialect=dialect) if any((v or "").strip() for v in r.values())]
    if dialect.delimiter == ";":
        # A ';'-delimited file is the European export format, where ',' is the DECIMAL
        # separator and '.' groups thousands. _to_float strips ',' as a thousands mark,
        # which would turn "0,0123" into 123.0 -- and by a different factor per row, so
        # it does not cancel out of a Sharpe ratio. Normalise before it gets there, but
        # only for cells that actually carry a ',' so US-style values in a ';' file
        # ("0.0123") are left alone.
        rows = [{k: (v.replace(".", "").replace(",", ".") if isinstance(v, str) and "," in v else v)
                 for k, v in r.items()} for r in rows]
    if not rows:
        raise ValueError("%s has no data rows" % path)
    fields = [f for f in rows[0].keys() if f is not None]
    return fields, rows


def load_returns(path, pct=False, prices=False):
    """Load one or many return series.

    Returns (series, labels) where series is an ordered dict-like list of
    (name, [float, ...]) and labels is the per-observation index (dates if a
    date column was found, else 1..T). Every numeric column that is not the
    date column is treated as one candidate strategy.
    """
    fields, rows = read_table(path)
    date_col = _pick(fields, _DATE_NAMES)
    labels = [r[date_col] for r in rows] if date_col else [str(i + 1) for i in range(len(rows))]

    series = []
    for f in fields:
        if f == date_col:
            continue
        col = [_to_float(r.get(f)) for r in rows]
        if sum(1 for v in col if v is not None) < 3:
            continue                                   # not a numeric column
        series.append((f, col))
    if not series:
        raise ValueError("no numeric columns found in %s" % path)

    if prices:
        conv = []
        for name, col in series:
            r = [None]
            for i in range(1, len(col)):
                a, b = col[i - 1], col[i]
                r.append(b / a - 1.0 if (a and b is not None and a > 0) else None)
            conv.append((name, r))
        series = conv
        labels = labels

    if pct:
        series = [(n, [None if v is None else v / 100.0 for v in c]) for n, c in series]

    return series, labels


def align(series, labels):
    """Drop observations where any series is missing, so every strategy is
    scored on identical rows. Returns (names, matrix, labels) with matrix[t][k]."""
    names = [n for n, _ in series]
    cols = [c for _, c in series]
    keep = [i for i in range(len(cols[0])) if all(c[i] is not None for c in cols)]
    matrix = [[c[i] for c in cols] for i in keep]
    return names, matrix, [labels[i] for i in keep]


def load_directional(path):
    """Load a prediction ledger: predicted direction, actual direction, and
    optionally the date the call was made on and a confidence in [0, 1]."""
    fields, rows = read_table(path)
    pcol = _pick(fields, _PRED_NAMES)
    acol = _pick(fields, _ACTUAL_NAMES)
    if not pcol or not acol:
        raise ValueError(
            "directional mode needs a predicted column (one of %s) and an actual "
            "column (one of %s); found %s" % (", ".join(_PRED_NAMES[:3]), ", ".join(_ACTUAL_NAMES[:3]), fields))
    dcol = _pick(fields, _DATE_NAMES)
    ccol = _pick(fields, _CONF_NAMES)

    def sign(v):
        x = _to_float(v)
        if x is None:
            t = _norm(v or "")
            if t in ("up", "u", "long", "buy", "true", "yes", "1"):
                return 1
            if t in ("down", "d", "short", "sell", "false", "no", "-1", "0"):
                return -1
            return None
        return 1 if x > 0 else (-1 if x < 0 else None)

    pred, actual, dates, conf = [], [], [], []
    for r in rows:
        p, a = sign(r.get(pcol)), sign(r.get(acol))
        if p is None or a is None:
            continue                                    # flat / abstained / unscored
        pred.append(p)
        actual.append(a)
        dates.append((r.get(dcol) or "").strip() if dcol else str(len(pred)))
        conf.append(_to_float(r.get(ccol)) if ccol else None)
    if len(pred) < 10:
        raise ValueError("only %d scorable predictions in %s" % (len(pred), path))
    return pred, actual, dates, (conf if ccol and any(c is not None for c in conf) else None)


# ---------------------------------------------------------------------------
# 5. Putting it together
# ---------------------------------------------------------------------------

def _annual(sr, k):
    return sr * math.sqrt(k)


def analyse_search(names, matrix, periods_per_year=252, n_boot=2000, s_blocks=16,
                   seed=0, stated_trials=None, label=None):
    """Full battery for a search over many candidate configurations."""
    T, N = len(matrix), len(matrix[0])
    columns = [[matrix[t][n] for t in range(T)] for n in range(N)]
    # a configuration that never trades is routine in a real sweep; drop it rather
    # than killing the run, and say how many went
    live = [i for i, c in enumerate(columns) if max(c) != min(c)]
    dropped = [names[i] for i in range(N) if i not in live]
    if len(live) < 2:
        raise LuckFilterError(
            "only %d of %d columns vary at all; there is no search to analyse" % (len(live), N))
    if dropped:
        names = [names[i] for i in live]
        columns = [columns[i] for i in live]
        matrix = [[row[i] for i in live] for row in matrix]
        N = len(live)
    sharpes = [sharpe(c) for c in columns]
    best = max(range(N), key=lambda n: sharpes[n])
    col = columns[best]
    _T, _mu, _sd, g3, g4 = moments(col)
    sr = sharpes[best]
    rho1 = autocorrelation(col, 1)
    se = sharpe_std_error(sr, T, g3, g4, rho1)

    nominal = stated_trials if stated_trials else N
    rho = average_correlation(columns) if N > 1 else 0.0
    n_indep = implied_independent_trials(nominal, rho)
    v_trials = statistics.variance(sharpes) if N > 1 else se * se
    v_source = "observed across the %d configurations" % N

    out = {
        "mode": "search", "label": label, "n_obs": T, "n_configs": N,
        "dropped_constant": dropped,
        "periods_per_year": periods_per_year, "years": T / float(periods_per_year),
        "best_name": names[best], "best_index": best,
        "sharpe": sr, "sharpe_annual": _annual(sr, periods_per_year),
        "sharpe_se_annual": _annual(se, periods_per_year),
        "ci_annual": (_annual(sr - Z95 * se, periods_per_year),
                      _annual(sr + Z95 * se, periods_per_year)),
        "skew": g3, "kurtosis": g4,
        "psr": psr(sr, 0.0, T, g3, g4, rho1),
        "median_sharpe_annual": _annual(statistics.median(sharpes), periods_per_year),
        "nominal_trials": nominal, "avg_correlation": rho,
        "independent_trials": n_indep, "trial_sharpe_sd": math.sqrt(v_trials),
        "v_source": v_source,
    }
    # below 2 the extreme-value approximation returns a NEGATIVE threshold, which
    # would let deflation raise the verdict instead of lowering it
    # the effective-trial haircut is NOT applied here: the cross-trial variance has
    # already absorbed the correlation, and haircutting on top lowers the threshold,
    # which measures as anti-conservative (8% rejection at rho=0.9 versus 5% nominal)
    dsr, sr0 = deflated_sharpe(sr, T, g3, g4, max(3.0, float(nominal)), v_trials, rho1)
    out["dsr"] = dsr
    out["threshold_sharpe_annual"] = _annual(sr0, periods_per_year)
    out["breakeven_trials"] = breakeven_trials(sr, T, g3, g4, v_trials, ar1=rho1)
    out["mintrl_years"] = min_track_record_length(
        sr, 0.0, g3, g4, ar1=rho1) / float(periods_per_year)
    out["autocorr_1"] = rho1
    if abs(rho1) > 0.05:
        naive = math.sqrt(periods_per_year)
        adj = eta_ar1(int(periods_per_year), rho1)
        out["autocorr_adjusted_sharpe_annual"] = sr * adj
        out["autocorr_naive_factor"] = naive

    if N >= 2 and T >= s_blocks * 2:
        out["pbo"] = probability_of_backtest_overfitting(matrix, s_blocks=s_blocks)
    out["reality_check"] = reality_check(columns, n_boot=n_boot, seed=seed)
    out["spa"] = superior_predictive_ability(columns, n_boot=n_boot, seed=seed)
    return out


def analyse_single(name, series, periods_per_year=252, stated_trials=None,
                   trial_sr_sd_annual=None):
    """Battery for one return series, where the search that produced it is
    something the user has to declare rather than something we can see."""
    T = len(series)
    _T, _mu, _sd, g3, g4 = moments(series)
    sr = sharpe(series)
    rho1 = autocorrelation(series, 1)
    se = sharpe_std_error(sr, T, g3, g4, rho1)
    out = {
        "mode": "single", "label": name, "n_obs": T,
        "periods_per_year": periods_per_year, "years": T / float(periods_per_year),
        "best_name": name, "sharpe": sr, "sharpe_annual": _annual(sr, periods_per_year),
        "sharpe_se_annual": _annual(se, periods_per_year),
        "ci_annual": (_annual(sr - Z95 * se, periods_per_year),
                      _annual(sr + Z95 * se, periods_per_year)),
        "skew": g3, "kurtosis": g4, "psr": psr(sr, 0.0, T, g3, g4, rho1),
        "mintrl_years": min_track_record_length(
            sr, 0.0, g3, g4, ar1=rho1) / float(periods_per_year),
        "nominal_trials": stated_trials, "autocorr_1": rho1,
    }
    if trial_sr_sd_annual is not None:
        v = (trial_sr_sd_annual ** 2) / periods_per_year
        out["v_source"] = "stated dispersion across trials"
    else:
        v = se * se
        out["v_source"] = ("no trial dispersion given, so the smallest defensible value "
                           "was used - this makes the deflated Sharpe OPTIMISTIC")
    out["trial_sharpe_sd"] = math.sqrt(v)
    out["breakeven_trials"] = breakeven_trials(sr, T, g3, g4, v, ar1=rho1)
    if stated_trials and stated_trials >= 2:
        out["independent_trials"] = stated_trials
        dsr, sr0 = deflated_sharpe(sr, T, g3, g4, max(3, int(stated_trials)), v, rho1)
        out["dsr"] = dsr
        out["threshold_sharpe_annual"] = _annual(sr0, periods_per_year)
    if abs(out["autocorr_1"]) > 0.05:
        out["autocorr_adjusted_sharpe_annual"] = sr * eta_ar1(int(periods_per_year),
                                                              out["autocorr_1"])
        out["autocorr_naive_factor"] = math.sqrt(periods_per_year)
    return out


def analyse_directional(pred, actual, dates, conf=None, seed=0, n_boot=4000):
    """Battery for a ledger of up/down calls."""
    hits = [1 if p == y else 0 for p, y in zip(pred, actual)]
    n = len(hits)
    up_rate = sum(1 for y in actual if y > 0) / float(n)
    base = max(up_rate, 1.0 - up_rate)
    out = {
        "mode": "directional", "n": n, "hit_rate": sum(hits) / float(n),
        "market_up_rate": up_rate, "always_one_way_rate": base,
        "pt": pesaran_timmermann(pred, actual),
        "hm": henriksson_merton(pred, actual),
        "binomial_p_vs_50": binomial_test(sum(hits), n, 0.5),
        "binomial_p_vs_base": binomial_test(sum(hits), n, base),
    }
    a, b, c, d = out["pt"]["table"]
    n1, n2, k1, k2 = a + b, c + d, a + c, b + d
    thin = min(n1, n2, k1, k2) == 0
    out["min_expected_cell"] = (0.0 if thin else
                                min(n1 * k1, n1 * k2, n2 * k1, n2 * k2) / float(n))
    # Pesaran-Timmermann is asymptotic. Below these thresholds it is not the test to
    # read, and the exact Henriksson-Merton result becomes primary instead.
    out["pt_reliable"] = (not thin) and n >= 50 and out["min_expected_cell"] >= 5.0
    out["n_days"] = len(set(dates))
    if out["n_days"] == 1:
        out["single_day"] = True
    elif len(set(dates)) < n:
        deff = design_effect(hits, dates)
        out["clustering"] = deff
        out["clustered_ci"] = cluster_bootstrap_ci(hits, dates, n_boot=n_boot, seed=seed)
        if out["pt"]["statistic"] is not None:
            out["pt_clustered_stat"] = out["pt"]["statistic"] / math.sqrt(deff["deff"])
            out["pt_clustered_p"] = 1.0 - _ND.cdf(out["pt_clustered_stat"])
    if conf and any(c is not None for c in conf):
        vals = [c for c in conf if c is not None]
        scale = 1.0
        if vals and max(vals) > 1.0 and max(vals) <= 100.0 and min(vals) >= 0.0:
            scale = 100.0                       # stated as percentages
            out["confidence_rescaled"] = True
        pairs = [(c / scale, 1 if p == y else 0) for c, p, y in zip(conf, pred, actual)
                 if c is not None and 0.0 <= c / scale <= 1.0]
        out["confidence_dropped"] = len(vals) - len(pairs)
        if len(pairs) >= 20:
            out["calibration"] = brier_decomposition([p for p, _ in pairs],
                                                     [h for _, h in pairs])
    return out


# ---------------------------------------------------------------------------
# 6. The verdict card
# ---------------------------------------------------------------------------

_USE_COLOR = False
_C = {"pass": "\033[32m", "fail": "\033[31m", "warn": "\033[33m",
      "dim": "\033[2m", "bold": "\033[1m", "off": "\033[0m"}


def _c(text, kind):
    return (_C[kind] + text + _C["off"]) if _USE_COLOR else text


LABEL_W = 34


def _row(out, label, value, mark=None, note=None):
    line = "    %-*s %s" % (LABEL_W, label, value)
    if mark:
        line += "  " + _c({"pass": "OK", "fail": "FAIL", "warn": "!"}[mark], mark)
    out.append(line)
    for n in ([note] if isinstance(note, str) else (note or [])):
        out.append("    %-*s %s" % (LABEL_W, "", _c(n, "dim")))


def _fmt_tiny(p):
    """An exact p-value that underflows to 0.0 is not zero, it is below what a float
    can carry."""
    return "< 1e-300" if p == 0.0 else "%.3g" % p


def _fmt_p(p, n_boot):
    """A bootstrap p-value of 0 is not zero, it is below what B draws can resolve."""
    return ("< %.3f" % (1.0 / n_boot)) if p == 0.0 else "%.3f" % p


def _fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not math.isfinite(x):
        return "never" if x > 0 else "-inf"
    return ("%%.%df" % nd) % x


def _head(out, title):
    out.append("")
    out.append("  " + _c(title, "bold"))


def render(rep):
    """The whole point of the tool: one screen that says whether to believe it."""
    out = []
    evidence = []                                   # (weight, text) against the result

    if rep["mode"] == "directional":
        return _render_directional(rep)

    k = rep["periods_per_year"]
    header = "LUCK FILTER   %s" % (rep.get("label") or rep["best_name"])
    out.append(_c(header, "bold"))
    detail = "%d observations (%.1f years)" % (rep["n_obs"], rep["years"])
    if rep["mode"] == "search":
        detail = "%d configurations searched  ·  " % rep["n_configs"] + detail
    out.append(_c("  " + detail, "dim"))

    _head(out, "WHAT THE SEARCH FOUND" if rep["mode"] == "search" else "THE HEADLINE NUMBER")
    if rep["mode"] == "search":
        _row(out, "Best configuration", rep["best_name"])
    _row(out, "Sharpe ratio (annualised)", _fmt(rep["sharpe_annual"]))
    _row(out, "95% confidence interval", "%s to %s" % (_fmt(rep["ci_annual"][0]),
                                                        _fmt(rep["ci_annual"][1])))
    if rep["mode"] == "search":
        _row(out, "Median configuration", _fmt(rep["median_sharpe_annual"]))
    _row(out, "Skew / kurtosis", "%s / %s" % (_fmt(rep["skew"]), _fmt(rep["kurtosis"])))
    _row(out, "P(true Sharpe > 0), one look", _fmt(rep["psr"], 3),
         None, "this is the number that flatters you: it ignores the search")
    if rep["ci_annual"][0] <= 0:
        evidence.append((1, "the 95% confidence interval includes zero"))

    if "autocorr_adjusted_sharpe_annual" in rep:
        _head(out, "SERIAL CORRELATION")
        _row(out, "First-order autocorrelation", _fmt(rep["autocorr_1"], 3))
        _row(out, "Sharpe after Lo adjustment",
             _fmt(rep["autocorr_adjusted_sharpe_annual"]),
             "warn" if rep["autocorr_1"] > 0.05 else None,
             "sqrt(%d) scaling is not valid here; the adjustment assumes an AR(1)" % k)
        if abs(rep["autocorr_1"]) > 0.3:
            evidence.append((5, "these returns are %.2f autocorrelated - they are "
                                "probably price levels rather than returns (try "
                                "--prices), and until that is resolved every number "
                                "above is void" % rep["autocorr_1"]))

    _head(out, "WHAT THE SEARCH COST")
    if rep.get("nominal_trials"):
        _row(out, "Configurations tried", "%d" % rep["nominal_trials"])
    if rep["mode"] == "search":
        _row(out, "Average pairwise correlation", _fmt(rep["avg_correlation"], 3))
        _row(out, "Independent trials", "%.0f" % rep["independent_trials"],
             None, "correlated configurations are not separate attempts")
    if "threshold_sharpe_annual" in rep:
        _row(out, "Best Sharpe from pure noise", _fmt(rep["threshold_sharpe_annual"]),
             None, ["what the best of these trials scores with no skill at all; "
                    "clearing it is necessary, not sufficient",
                    "spread assumed across trials: " + rep.get("v_source", "unstated")])
        _row(out, "Deflated Sharpe Ratio", _fmt(rep["dsr"], 3),
             "pass" if rep["dsr"] >= 0.95 else "fail",
             "P(the true Sharpe beats that threshold), given the trial count you declared")
        if rep["dsr"] < 0.95:
            evidence.append((3, "the deflated Sharpe is %.2f, not the %.2f the raw "
                                "number suggested" % (rep["dsr"], rep["psr"])))
    _row(out, "Trials that would explain it",
         "%s" % (rep["breakeven_trials"] if rep["breakeven_trials"] else "more than 1e6"),
         None, "searching this many configurations is enough to produce it by chance")
    _row(out, "Record needed to beat zero",
         ("no length is enough" if not math.isfinite(rep["mintrl_years"])
          else "%s years" % _fmt(rep["mintrl_years"], 1)),
         "warn" if rep["mintrl_years"] > rep["years"] else "pass",
         "to be 95% sure the Sharpe exceeds zero" +
         (" - a far lower bar than the noise threshold above"
          if "threshold_sharpe_annual" in rep else
          ", the easiest bar there is; declare --trials for the one that matters"))
    if not math.isfinite(rep["mintrl_years"]):
        evidence.append((5, "the Sharpe ratio does not beat zero at all, so no length "
                            "of track record would establish it"))
    elif rep["mintrl_years"] > rep["years"]:
        short_by = rep["mintrl_years"] / rep["years"]
        evidence.append((5 if short_by > 5 else 2,
                         "the record is %.1f years and would need %.1f, %.0fx longer"
                         % (rep["years"], rep["mintrl_years"], short_by)))

    pbo = rep.get("pbo")
    if pbo:
        _head(out, "DOES THE CHOICE GENERALISE")
        _row(out, "Prob. of backtest overfitting", _fmt(pbo["pbo"], 3),
             "pass" if pbo["pbo"] < 0.5 else "fail",
             "how often the in-sample winner lands in the bottom half out of sample")
        _row(out, "Performance degradation", _fmt(pbo["degradation_slope"], 3),
             "pass" if pbo["degradation_slope"] > 0 else "fail",
             "out-of-sample Sharpe per unit of in-sample Sharpe; negative is perverse")
        _row(out, "Prob. of out-of-sample loss", _fmt(pbo["prob_loss"], 3),
             "pass" if pbo["prob_loss"] < 0.5 else "fail")
        _row(out, "Splits examined", "%d" % pbo["n_splits"])
        if pbo["pbo"] >= 0.5:
            evidence.append((3, "picking the in-sample best is a coin flip out of sample "
                                "(PBO %.2f)" % pbo["pbo"]))

    rc, spa = rep.get("reality_check"), rep.get("spa")
    if rc:
        _head(out, "IS THE BEST ONE BETTER THAN NOTHING")
        strong = rep.get("autocorr_1", 0.0) > 0.1
        _row(out, "White's Reality Check p", _fmt_p(rc["p_value"], rc["n_boot"]),
             ("pass" if rc["p_value"] < 0.05 else "fail") if not strong else None,
             ["P(a search this wide produces a winner this good by chance)"] +
             ([] if not strong else
              ["these returns are %.2f autocorrelated; the block bootstrap only "
               "partly absorbs that, so both p-values here are optimistic - measured "
               "8.5%% rejection against a nominal 5%% at this level of dependence"
               % rep["autocorr_1"]]))
        _row(out, "Hansen SPA p (consistent)", _fmt_p(spa["p_consistent"], spa["n_boot"]),
             ("pass" if spa["p_consistent"] < 0.05 else "fail") if not strong else None)
        _row(out, "SPA p (lower / upper bound)",
             "%s / %s" % (_fmt_p(spa["p_lower"], spa["n_boot"]),
                          _fmt_p(spa["p_upper"], spa["n_boot"])))
        if strong:
            evidence.append((1, "the bootstrap tests are unreliable at this level of "
                                "serial correlation, so they neither support nor "
                                "condemn the result"))
        elif rc["p_value"] >= 0.05:
            evidence.append((2, "the Reality Check cannot reject chance (p = %.2f)"
                             % rc["p_value"]))

    weight = sum(w for w, _ in evidence)
    if rep["n_obs"] < 30:
        out.append("")
        out.append("  " + _c("VERDICT: INSUFFICIENT DATA", "warn") + "   " +
                   _c("%d observations cannot support any of these statistics; every "
                      "number above is noise." % rep["n_obs"], "dim"))
        out.append("")
        return "\n".join(out)
    if weight == 0:
        verdict, kind = "SURVIVES", "pass"
        blurb = "Nothing here rules out real skill. That is not the same as proof."
    elif weight >= 5:
        verdict, kind = "LUCK", "fail"
        blurb = "This is what an overfitted backtest looks like."
    else:
        verdict, kind = "NOT PROVEN", "warn"
        blurb = "Some evidence survives, but not enough to trade on."
    out.append("")
    out.append("  " + _c("VERDICT: " + verdict, kind) + "   " + _c(blurb, "dim"))
    for _w, text in sorted(evidence, reverse=True):
        out.append("    - " + text)
    out.append("")
    return "\n".join(out)


def _render_directional(rep):
    out = [_c("LUCK FILTER   directional prediction ledger", "bold"),
           _c("  %d scored calls" % rep["n"], "dim")]
    evidence = []
    pt, hm = rep["pt"], rep["hm"]

    _head(out, "THE HEADLINE NUMBER")
    _row(out, "Hit rate", _fmt(rep["hit_rate"], 3))
    _row(out, "Market went up", _fmt(rep["market_up_rate"], 3), None,
         "a forecaster who always said the same thing would score %s"
         % _fmt(rep["always_one_way_rate"], 3))
    _row(out, "Binomial p vs a coin flip", _fmt_tiny(rep["binomial_p_vs_50"]), None,
         "the number that fools people: it assumes independence and a 50% base rate")

    _head(out, "IS THERE ANY INFORMATION IN IT")
    if pt["degenerate"]:
        _row(out, "Pesaran-Timmermann", "no test possible", "fail", pt["note"])
        evidence.append((5, "the forecast never varied, so it carries no information"))
    else:
        trust = rep.get("pt_reliable", True)
        _row(out, "Hit rate expected by accident", _fmt(pt["expected_hit_rate"], 3), None,
             "from your own directional bias alone, with zero skill")
        _row(out, "Pesaran-Timmermann statistic", _fmt(pt["statistic"], 2))
        _row(out, "Pesaran-Timmermann p", _fmt_tiny(pt["p_value"]),
             ("pass" if pt["p_value"] < 0.05 else "fail") if trust else None,
             ["not corrected for how many rules or forecasters you tried before this "
              "one - a ledger cannot see your search the way a grid of returns can"] +
             ([] if trust else
              ["ADVISORY ONLY: %d calls with a smallest expected cell of %.1f is below "
               "what this asymptotic test needs; read the exact test below instead"
               % (rep["n"], rep.get("min_expected_cell", 0.0))]))
        if trust and pt["p_value"] >= 0.05:
            evidence.append((5, "directional accuracy is indistinguishable from chance"))
    if not hm["degenerate"]:
        primary = not rep.get("pt_reliable", True)
        _row(out, "Henriksson-Merton exact p", _fmt_tiny(hm["p_value"]),
             "pass" if hm["p_value"] < 0.05 else "fail",
             "exact, valid at any sample size" +
             (" - and the primary result here" if primary else ""))
        if primary and hm["p_value"] >= 0.05:
            evidence.append((4, "the exact test cannot distinguish these calls from "
                                "chance at this sample size"))

    if rep.get("single_day"):
        _head(out, "HOW MUCH EVIDENCE IS ACTUALLY THERE")
        _row(out, "Calls / distinct days", "%d / 1" % rep["n"], "fail",
             "every call was made on one day, so this is a single draw from the market "
             "however many rows it has; no clustering correction can rescue it")
        evidence.append((5, "all %d calls share a single day - that is one observation, "
                            "not %d" % (rep["n"], rep["n"])))
    cl = rep.get("clustering")
    if cl:
        _head(out, "HOW MUCH EVIDENCE IS ACTUALLY THERE")
        _row(out, "Calls / distinct days", "%d / %d" % (cl["n_obs"], cl["n_clusters"]))
        _row(out, "Within-day correlation", _fmt(cl["rho"], 3), None,
             "calls made the same morning share one market")
        _row(out, "Design effect", _fmt(cl["deff"], 2))
        shrunk = cl["n_eff"] < cl["n_obs"] * 0.95
        _row(out, "Effective sample size", "%.0f" % cl["n_eff"],
             "warn" if cl["n_eff"] < cl["n_obs"] / 2 else None,
             ("not %d - that is the number every naive p-value above used" % cl["n_obs"])
             if shrunk else
             "the hits themselves are not correlated within a day, so the raw count "
             "stands even though the outcomes share a market")
        ci = rep["clustered_ci"]
        if ci["reliable"]:
            _row(out, "Hit rate 95% interval",
                 "%s to %s" % (_fmt(ci["lo"], 3), _fmt(ci["hi"], 3)), None,
                 "cluster bootstrap, %s%s" % (ci["method"],
                 "; only %d clusters, so treat the width as indicative" % ci["n_clusters"]
                 if ci["marginal"] else ""))
        else:
            _row(out, "Hit rate 95% interval", "not reported", "warn",
                 "resampling %d clusters carries almost no information about the "
                 "sampling variance - the interval it produces is far too narrow "
                 "rather than too wide" % ci["n_clusters"])
        if "pt_clustered_p" in rep:
            _row(out, "Clustered Pesaran-Timmermann p", _fmt_tiny(rep["pt_clustered_p"]),
                 "pass" if rep["pt_clustered_p"] < 0.05 else "fail")
            if rep["pt_clustered_p"] >= 0.05:
                evidence.append((2, "once same-day calls stop counting as independent, "
                                    "the signal does not survive"))

    cal = rep.get("calibration")
    if cal:
        _head(out, "ARE THE CONFIDENCES HONEST")
        _row(out, "Brier score", _fmt(cal["brier"], 4), None, "lower is better")
        if cal["informative"]:
            _row(out, "Reliability (calibration error)", _fmt(cal["reliability"], 4),
                 "pass" if cal["reliability"] < 0.01 else "warn", "0 means 70% means 70%")
            _row(out, "Resolution", _fmt(cal["resolution"], 4), None,
                 "how far the forecasts move off the base rate; larger is better")
        else:
            _row(out, "Reliability / resolution", "not reported", None, cal["note"])
        _row(out, "Skill vs always saying base rate", _fmt(cal["skill_score"], 4),
             "pass" if (cal["skill_score"] or 0) > 0 else "fail")
        if (cal["skill_score"] or 0) <= 0:
            evidence.append((2, "the probabilities score worse than always quoting the "
                                "base rate"))

    weight = sum(w for w, _ in evidence)
    if rep["n"] < 30:
        out.append("")
        out.append("  " + _c("VERDICT: INSUFFICIENT DATA", "warn") + "   " +
                   _c("%d calls cannot support any of these statistics." % rep["n"], "dim"))
        out.append("")
        return "\n".join(out)
    if weight == 0:
        verdict, kind, blurb = ("SURVIVES", "pass",
                                "Nothing here rules out directional skill. "
                                "That is not the same as proof.")
    elif weight >= 5:
        verdict, kind, blurb = "LUCK", "fail", "No demonstrated directional skill."
    else:
        verdict, kind, blurb = "NOT PROVEN", "warn", "Suggestive, not established."
    out.append("")
    out.append("  " + _c("VERDICT: " + verdict, kind) + "   " + _c(blurb, "dim"))
    for _w, text in sorted(evidence, reverse=True):
        out.append("    - " + text)
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 7. Command line
# ---------------------------------------------------------------------------

DEMO_NOTE = """\
  The data above is a GEOMETRIC RANDOM WALK WITH ZERO DRIFT. There is no signal
  in it. None. Every configuration is trading noise, and the winner is simply
  the best of 190 coin-flip sequences.

%s

  Over 25 independent sweeps of pure noise, seeds 1-25:

      median best-of-190 Sharpe        1.14      half of all noise beats this
      median deflated Sharpe           0.496     the 0.5 the null predicts
      median overfitting probability   0.582
      noise searches clearing DSR>0.95 0 of 25

  Run --seed N to see another draw. A raw Sharpe near 1 proves nothing on its
  own; that is the entire point."""


def _demo(args):
    warmup = 200                                      # longest lookback in the grid
    px = random_walk(args.obs + warmup, args.seed)
    grid = strategy_grid(px)
    names = list(grid)
    matrix = [[grid[n][t] for n in names] for t in range(warmup, len(px))]
    rep = analyse_search(names, matrix, periods_per_year=args.periods,
                         n_boot=args.boot, s_blocks=args.blocks, seed=args.seed,
                         label="demo: a parameter sweep over pure noise")
    print(render(rep))
    if args.seed == 17:
        provenance = ("  Seed 17 is not cherry-picked - across seeds 1-25 it is the MEDIAN\n"
                      "  draw, which is the only honest way to pick a default in a tool\n"
                      "  about selection bias.")
    else:
        provenance = ("  Seed %d is one draw of many and its outcome is not representative\n"
                      "  of anything. Seed 17 is the median of seeds 1-25." % args.seed)
    print(_c(DEMO_NOTE % provenance, "dim"))
    print()
    return rep


def main(argv=None):
    global _USE_COLOR
    ap = argparse.ArgumentParser(
        prog="luckfilter",
        description="Tell the difference between a strategy that works and one you found.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  luckfilter.py --demo                        mine pure noise and watch it get killed
  luckfilter.py returns.csv --trials 200      one series; declare what you searched
  luckfilter.py grid.csv                      many columns = many candidates
  luckfilter.py calls.csv --directional       a ledger of up/down calls

csv shapes:
  returns   optional date column, then one column per candidate strategy
  prices    same, with --prices to difference them into returns first
  calls     predicted + actual columns; optional date (clustering) and
            confidence (calibration) columns
""")
    ap.add_argument("csv", nargs="?", help="path to a CSV of returns or predictions")
    ap.add_argument("--demo", action="store_true",
                    help="run the built-in demonstration on data with no signal")
    ap.add_argument("--directional", action="store_true",
                    help="treat the CSV as a ledger of up/down calls")
    ap.add_argument("--prices", action="store_true",
                    help="columns are price levels; difference them into returns")
    ap.add_argument("--pct", action="store_true",
                    help="values are percentages (1.5 means 1.5%%), not fractions")
    ap.add_argument("--trials", type=int, default=None,
                    help="how many configurations you tried before keeping this one")
    ap.add_argument("--trial-sd", type=float, default=None, dest="trial_sd",
                    help="annualised standard deviation of Sharpe across those trials")
    ap.add_argument("--periods", type=int, default=252,
                    help="observations per year (252 daily, 52 weekly, 12 monthly)")
    ap.add_argument("--rf", type=float, default=0.0,
                    help="risk-free rate per period, subtracted before anything else")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap replications")
    ap.add_argument("--blocks", type=int, default=16, help="CSCV blocks (even, >= 4)")
    ap.add_argument("--obs", type=int, default=756, help="demo: observations to generate")
    ap.add_argument("--seed", type=int, default=17, help="demo and bootstrap seed")
    ap.add_argument("--json", action="store_true", help="emit the raw numbers as JSON")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    args = ap.parse_args(argv)

    _USE_COLOR = sys.stdout.isatty() and not args.no_color and not args.json

    if not args.demo and not args.csv:
        ap.print_help()
        return 2
    try:
        if args.demo:
            rep = _demo(args)
        elif args.directional:
            pred, actual, dates, conf = load_directional(args.csv)
            rep = analyse_directional(pred, actual, dates, conf, seed=args.seed,
                                      n_boot=args.boot)
            if not args.json:
                print(render(rep))
        else:
            series, labels = load_returns(args.csv, pct=args.pct, prices=args.prices)
            names, matrix, _labels = align(series, labels)
            if args.rf:
                matrix = [[v - args.rf for v in row] for row in matrix]
            if len(names) == 1:
                rep = analyse_single(names[0], [row[0] for row in matrix],
                                     periods_per_year=args.periods,
                                     stated_trials=args.trials,
                                     trial_sr_sd_annual=args.trial_sd)
            else:
                rep = analyse_search(names, matrix, periods_per_year=args.periods,
                                     n_boot=args.boot, s_blocks=args.blocks,
                                     seed=args.seed, stated_trials=args.trials,
                                     label=args.csv)
            if not args.json:
                print(render(rep))
    except LuckFilterError as exc:
        sys.stderr.write("luckfilter: %s\n" % exc)
        return 1
    except (OSError, csv.Error) as exc:
        sys.stderr.write("luckfilter: could not read %s: %s\n" % (args.csv, exc))
        return 1

    if args.json:
        def clean(o):
            if isinstance(o, dict):
                return dict((k, clean(v)) for k, v in o.items() if k != "logits")
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, float) and not math.isfinite(o):
                return None
            return o
        print(json.dumps(clean(rep), indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
