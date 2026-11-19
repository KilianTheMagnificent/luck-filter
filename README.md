# Luck Filter

Tell the difference between a strategy that works and one you found.

A backtest that looks good proves almost nothing on its own. If you tried 200 parameter
combinations, the best of them looks good even when the data contains no signal at all.
`luck-filter` is a zero-dependency Python 3.9+ statistical backtest-overfitting and
multiple-testing tool — plus a self-contained offline HTML companion (`skill-or-luck.html`,
also [live in the browser](https://kilianthemagnificent.github.io/luck-filter/)) —
that prices that selection bias in.

## Quick start

Requires Python 3.9 or newer. Zero third-party dependencies.

Run directly without cloning using `uvx` or install as a CLI with `pipx`:

```sh
uvx --from git+https://github.com/KilianTheMagnificent/luck-filter luck-filter --demo
# or install permanently:
pipx install git+https://github.com/KilianTheMagnificent/luck-filter
luck-filter --demo
```

Or run with zero install step from a local clone:

```sh
git clone https://github.com/KilianTheMagnificent/luck-filter.git
cd luck-filter
python3 -B luckfilter.py --demo
python3 -B test_luckfilter.py
```

The `--demo` command mines a **geometric random walk with zero drift** — data with no signal
by construction — with a 190-configuration parameter sweep, keeps the winner, and then takes it
apart. Across seeds 1–25 the median best-of-190 Sharpe on that worthless data is **1.14**, and
no seed produced a deflated Sharpe above 0.95.

To explore the argument interactively in a browser, visit the
[live interactive explainer](https://kilianthemagnificent.github.io/luck-filter/) or open
`skill-or-luck.html` directly from the cloned folder (offline, zero network requests).

## Usage

```sh
python3 -B luckfilter.py returns.csv --trials 200   # one series; declare what you searched
python3 -B luckfilter.py grid.csv                   # many columns = many candidates
python3 -B luckfilter.py calls.csv --directional    # a ledger of up/down calls
python3 -B luckfilter.py returns.csv --json         # the raw numbers
```

## What it runs

| Test | Answers | Source |
|---|---|---|
| Probabilistic Sharpe Ratio | Is the Sharpe above zero, allowing for skew and fat tails? | Bailey & López de Prado (2012) |
| Deflated Sharpe Ratio | Is it above what the best of N worthless attempts would score? | Bailey & López de Prado (2014) |
| Minimum Track Record Length | How long a record would settle it? | Bailey & López de Prado (2012) |
| Probability of Backtest Overfitting | Does picking the in-sample winner generalise, or is it a coin flip? | Bailey, Borwein, López de Prado & Zhu (2016) |
| Reality Check | Could a search this wide produce a winner this good by chance? | White (2000) |
| Superior Predictive Ability | The same, robust to padding the search with junk. | Hansen (2005) |
| Stationary bootstrap | Resampling that preserves serial dependence. | Politis & Romano (1994) |
| Lo's η(q) | Is √252 scaling even valid on these returns? | Lo (2002) |
| Pesaran–Timmermann | Is the hit rate better than your own bias earns for free? | Pesaran & Timmermann (1992) |
| Henriksson–Merton | The same, exactly, when the sample is small. | Henriksson & Merton (1981) |
| Design effect | How many independent observations do you really have? | Kish (1965); Donner (1986) |
| Murphy decomposition | Do your stated confidences mean anything? | Murphy (1973) |

## Input shapes

**Returns** — an optional date column, then one column per candidate strategy. One column
gets the single-series battery; several columns are treated as a search and get PBO, the
Reality Check and SPA as well. `--prices` differences price levels into returns first;
`--pct` divides by 100; `--rf` subtracts a per-period risk-free rate.

**Predictions** — `predicted` and `actual` columns (`up`/`down`, `1`/`-1`, `long`/`short`,
`true`/`false` all parse). An optional date column turns on the clustering correction; an
optional `confidence` column in [0, 1] turns on calibration scoring.

## The three traps it exists to catch

1. **You searched.** The Sharpe ratio does not know how many configurations you tried, and
   neither does anyone reading your backtest. `--trials` is the number that matters most and
   it lives only in your memory. Count every parameter combination, every universe, and every
   time you looked at a result and went back to change something.

2. **Your calls share a day.** Forty predictions made on one morning are not forty
   independent facts — when the market rallies most "up" calls are right at once. The tool
   estimates the within-day correlation and reports the effective sample size that should
   replace the raw count in every standard error.

3. **Your hit rate is the base rate.** In a market that rises 58% of days, a forecaster who
   says "up" every day is right 58% of the time with no information whatsoever. A naive
   binomial test calls that p = 2e-7. Pesaran–Timmermann calls it what it is.

## Verification

```sh
python3 -B test_luckfilter.py        # 55 checks
```

These are mostly not "does it run" tests. They are the published numbers from the source
papers and the properties that must hold by construction:

- **Bailey & López de Prado (2014) p.10** — the DSR paper's own worked example reproduces to
  the printed digit: expected best-of-100 Sharpe **0.1132**, deflated Sharpe **0.9004**, and
  both published breakeven trial counts (**47** non-normal, **89** normal) exactly.
- **Bailey & López de Prado (2012) Figures 8–11** — all four published minimum track record
  lengths (2.73, 2.83, 3.24, 4.99 years).
- **Lo (2002) Table 1** — 32 published Sharpe standard errors; **Table 2** — 50 published
  η(q) cells, plus monotonicity in ρ.
- **PBO** — a hand-computed 4-block, 2-strategy case exact to 1e-12 including the
  degradation slope; identical strategies give exactly 1.0 with every logit exactly 0; pure
  noise averages 0.5 at even N; a genuinely dominant strategy gives 0.
- **Calibration by simulation** — PSR is Uniform(0,1) under the null, so its 5% tail is 5%,
  and it stays at 5% on serially correlated returns rather than drifting to 11%.
- **Guards** — passing excess kurtosis where raw kurtosis belongs fails loudly rather than
  silently returning a wrong answer, and a flat return series is refused rather than reported
  as certain skill.

## Status

`v1.0.0` — working, standalone local CLI/library and offline HTML companion with 55 automated
self-checks passing on Python 3.9+.

Research and analysis only. It runs completely locally with zero network calls and does not
handle logins, accounts, passwords, encryption, backups, or trade execution.

## How this was built

Andre Romero directs development with AI coding assistants and verifies results empirically
rather than writing Python by hand.

For `luck-filter`, the formulas were established from the primary papers across four
independent AI research passes, each adversarially audited by a second pass that recomputed
every numerical claim. The finished implementation was then attacked from several more angles,
including a review by a different model family to break correlated blind spots. Every defect
caught during those audits is pinned by an automated check you can re-run locally with
`python3 -B test_luckfilter.py`:

**Wrong numbers**

- PBO depended on the **order the columns were passed in** whenever two configurations tied
  for in-sample best — rare for Gaussian returns, routine once payoffs are discrete.
  Reordering the same data moved PBO from 0.833 to 0.667. Ties are now averaged over.
- `min_track_record_length` squares Z(α), so α = 0.3 silently returned the α = 0.7 answer.
  Now refused.
- The Brier decomposition binned by unique forecast value, which on a short ledger of
  distinct probabilities makes reliability **equal the Brier score by construction** — a
  catastrophic-miscalibration reading that is a pure artefact. It now declines to report
  calibration it cannot measure.
- `math.comb` is exact, but its product with a float **overflows** once a ledger reaches a
  few thousand rows — a year of daily multi-ticker predictions. Both exact tails now run in
  log space.
- On a grid of near-duplicate configurations the implied independent-trial count falls below
  2, where the extreme-value approximation returns a **negative** noise threshold and
  deflation *raised* the verdict. Floored at 2, with a test asserting DSR ≤ PSR.
- PBO's fast path recovers each split's Sharpe from block-level sums instead of
  re-concatenating rows. Checked against direct computation, it agreed to 1e-16 on
  continuous returns but **diverged on discrete payoffs** — the ledger case this tool is
  for. Two causes: a block set whose true mean is exactly zero came back as +2.7e-17, and
  tie detection used exact float equality, so two algebraically identical routes picked
  different winners. Both now use tolerances, and a test compares the shortcut against
  direct computation on continuous, discrete, binary and sparse payoffs.
- On a 4-call ledger the asymptotic Pesaran–Timmermann test read p = 0.01 while the exact
  Henriksson–Merton test read 0.25, and the card reported SURVIVES on the strength of the
  former. Below 50 calls or an expected cell count of 5, the exact test is now primary and
  the asymptotic one is marked advisory.
- **There was no minimum sample size at all.** Four observations produced an annualised
  Sharpe of 188 and `VERDICT: SURVIVES`. Below 30 observations both cards now refuse to
  return a verdict.
- **Two of the three documented invocations were structurally incapable of ever printing
  LUCK.** A two-column prediction ledger of pure coin flips, and ten years of single-column
  noise without `--trials`, both maxed out below the evidence threshold and read
  "Suggestive, not established". Both now reach LUCK.
- **A ledger where every call falls on one day — the maximally clustered input — received
  no clustering correction**, because the gate required at least 2 distinct dates. It is
  now the loudest finding on the card.
- The cluster bootstrap was reported at any number of clusters. Measured coverage of its
  nominal 95% interval is 0.39 at 2 clusters and 0.85 at 5, i.e. **narrower** than the naive
  interval it exists to widen. It is now withheld below 10 clusters and caveated below 25.
- Serial correlation was computed, displayed, and then ignored by the verdict, so a price
  series pasted in as returns scored a Sharpe of 129 and SURVIVES. An autocorrelation above
  0.3 is now dispositive.
- Duplicate CSV column names were silently collapsed by `csv.DictReader`, dropping a
  strategy and switching the run out of search mode — so a real search got no deflation at
  all. Now refused.
- A single flat column aborted an entire grid. Now dropped and reported.
- Confidences on a 0–100 scale silently deleted the whole calibration section. Now
  rescaled.

**Wrong sizes** — the p-values not meaning what they said

- **The Sharpe-ratio family assumed independent observations.** Measured under an AR(1),
  the nominal 5% PSR test fired **11.3% of the time at ρ₁ = 0.3 and 16.2% at ρ₁ = 0.5** —
  anti-conservative, on exactly the smoothed and overlapping-position returns this tool
  exists for. The card detected the autocorrelation, printed it, and then ignored it. The
  variance factor now carries the AR(1) long-run correction, which restores the measured
  size to **5.0% across ρ₁ from −0.2 to 0.5**, and a test asserts it.
- The Deflated Sharpe took a haircut for correlated trials *on top of* a trial variance
  that had already absorbed the correlation, which lowers the bar in the wrong direction —
  8.4% rejection against a nominal 5% at ρ = 0.9. The haircut is now reported as
  description only and no longer feeds the statistic.
- The automatic block length computed m̂ one lag short, shortening every bootstrap block,
  narrowing the null, and making the Reality Check anti-conservative. It also chose the
  block from the best-performing candidate, which can look independent while its siblings
  do not; it now takes the longest block any candidate demands.
- Residual over-rejection under strong serial correlation could not be fully removed, so
  above ρ₁ = 0.1 the card **stops marking the bootstrap p-values pass or fail** and says
  what it measured instead. That is the honest end state, not a fix.
- The extreme-value approximation is 7.8% *below* the truth at N = 2, which the
  near-duplicate guard had made the operative floor. Floored at 3.

**Things it said that were not true**

- The demo interpolated whatever `--seed` you passed into the sentence claiming that seed was
  the median draw. On `--seed 3` — the single worst draw in the sweep — it printed a
  fabricated provenance claim. A tool about selection bias does not get to do that.
- The deflated Sharpe was computed against an unstated assumption about how much the trial
  Sharpes varied, and the default was the *flattering* end. The card now prints that
  assumption on the same row.
- The directional card asserted "There is information in these calls", an affirmative claim,
  where the search card correctly says only that nothing has been ruled out. It also cannot
  see your search at all, and now says so.
- The DSR row was labelled "P(real skill)". It is not.
- "Track record needed for proof" was measured against zero while sitting directly beneath a
  much higher noise threshold, so a strategy could be told it had "enough" record to prove a
  bar the card had just said was the wrong one.
- `p_up` and `prob` were accepted as confidence columns and scored against whether the call
  was right. For a bearish call, P(market up) = 0.2 means 80% confident, not 20%; a perfectly
  calibrated forecaster was being reported at reliability 0.186 and skill −1.18. Those column
  names are no longer accepted under that meaning.

## What passing does not mean

Surviving these checks does not mean a strategy works. It means one specific way of being
wrong has been ruled out. The deflated Sharpe is only as honest as the trial count you feed
it; PBO and the Reality Check lose power when candidates are highly correlated; and the
clustered correction is a first-order approximation that scales the whole null variance by
the design effect.

Research and analysis only. Nothing here places trades, moves money, or recommends a
position.

## Files

- `luckfilter.py` — the whole tool
- `test_luckfilter.py` — the 55 self-checks above
- `skill-or-luck.html` — an interactive version of the argument, self-contained, no network ([live version](https://kilianthemagnificent.github.io/luck-filter/))
- `pyproject.toml` — minimal packaging metadata for `uvx` / `pipx` / `pip` installs

## Contributing & Security

Bug reports and reproducible test cases are welcome through issues; outside pull requests are
turned off. See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md),
and [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE) © 2026 Andre Romero
