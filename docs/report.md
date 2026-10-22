# Cheap screens, real fills: evidence for the Backtest Orchestrator

The present report gives the evidence for the method choices of the Backtest Orchestrator, which is project I5
of the Tradeq platform. All numbers come from the command make report, which runs the command bt report with
fixed seeds and writes the files tables.md and experiments.json under the directory
$TRADEQ_DATA_DIR/backtest-orchestrator/reports/. When the text cites Table E1 to E5, it refers to the section
of tables.md that carries that name, and when it cites a number that is not in tables.md, it names
experiments.json as the source.

## 1. Question

The present project asks how a researcher should combine existing backtest engines so that a strategy that
looks good in a cheap backtest is checked in a realistic one before it reaches live trading. The specification
of the platform, section I5, states that engines compete on speed while four things are missing, which are
fill models that are calibrated on the platform's own fills, a promotion path from cheap to realistic
fidelity, automatic parity checks across engines and automatic trial logging, so we can conclude that the
comparison must test these four methods against the single-engine practice that the analogs support. The first
question is whether fill models that the project learns from fills predict the true fill process better than
the naive fills of bar-level engines. The second question is whether a fidelity ladder, which is a sequence of
engines from the cheapest to the most realistic in which a promotion gate decides whether a strategy may move
up, reaches the verdicts of the most realistic engine at a lower cost than either single engine. The third
question is whether a parity check, which runs one strategy on two engines with matched settings and compares
the results, finds engine bugs that a comparison of final profit misses.

## 2. Data

The present study uses synthetic data only, because the platform has no recorded fills from the Shadow
Exchange (project I6) and no point-in-time market history from project I2 yet, so the results show how the
methods behave and are not market evidence. The module data/synthetic.py generates the data, and the file
config/settings.yaml holds every parameter. Each of the 8 seeds, which are 101 to 108, gives one market of
1440 one-minute bars, which is one day. The logarithm of the mid price follows a random walk with a volatility
of 1.2 basis points (bps, hundredths of a percent) per square-root second, and a drift of 0.3 bps per bar
switches every 240 bars between up, flat and down. The quotes update every second with a spread of 2 to 4
ticks, the book has 5 levels per side, and trades arrive at the touch as a Poisson process with 0.8 trades per
second and a random side.

Each seed also writes 6000 synthetic orders in the fills format that section 6.4 of the platform conventions
defines, and 70% of them are limit orders. A limit order that rests d bps from the touch with q units queued
ahead fills after an exponential time whose rate is 0.03 per second times exp(-0.35 d - 0.45 log(1 + q) + 0.8
(v - 1.2)), where v is the volatility of its regime. An order that has not filled after 300 seconds is
cancelled, and it then appears only in an optional orders file, which the present project adds to the
contract. A market order pays half the spread plus 25 bps times the square root of its size divided by the
average daily volume (ADV), and Gaussian noise with a standard deviation of 0.3 bps is added. These formulas
are the true fill process, against which experiment E1 scores the fill models.

## 3. Methods

The orchestrator runs every strategy through four rungs, which are the levels of the ladder. The vectorised
rung passes a target position per bar to vectorbt, which fills every order at the bar close, or, for a limit
order, at the better of the limit price and the bar open as soon as the bar range touches the limit, and this
assumption is called touch fills. The event rung replays the bars in NautilusTrader, whose matching engine
fills orders against the open, high, low and close of each bar. The tick rung replays every book update and
trade of the tape in hftbacktest, where a resting limit order fills only after the queue ahead of it is
consumed, and every order waits for an entry latency of 5 milliseconds, or for the calibrated median latency
when a fill model exists. The reactive rung asks the S3 simulator for a stressed tape and replays it on
hftbacktest, and the present study uses the in-process stand-in of the S3 simulator, which widens the spread
and thins the book but does not react to the strategy. Every rung charges a maker fee of 2 bps and a taker fee
of 5 bps, so fees cannot explain a difference between rungs, and the orchestrator logs every run as a trial to
the Research Ledger (project S6).

The present study tests the rungs with two strategies, which the package strategies defines. The SMA-cross
strategy holds one unit long when the 30-bar simple moving average (SMA) of the close is above the 120-bar SMA
and one unit short otherwise, and it trades only with market orders. The mean-reversion strategy computes the
z-score of the close, which is its distance from the 20-bar mean in standard deviations, and it posts a limit
order 1 bp away from the close when the z-score passes 1.5, while it exits with a market order when the
z-score reverts past 0.25. The Sharpe ratio of a run is the mean of its one-minute returns divided by their
standard deviation, and we annualise it with the square root of 525600 minutes per year, which makes one-day
ratios large and noisy. The confidence interval of a Sharpe ratio, and of the Sharpe drop between two rungs,
comes from 1000 resamples of the stationary bootstrap of Politis and Romano (1994), which draws blocks of
returns with a mean length of 30 bars, so that the resamples keep the short-range dependence of the returns.

A gate compares the run of a rung with the thresholds that config/settings.yaml holds. The vectorised gate
asks for a Sharpe ratio of at least 0.5, at least 10 fills and a drawdown of at most 25%, while the event gate
asks for a Sharpe ratio of at least 0.3, at least 10 fills and a drop of at most 2.0 in the Sharpe ratio
against the vectorised rung. The tick gate asks for a Sharpe ratio of at least 0, at least 5 fills and a
drawdown of at most 30%, and it applies the same limit on the drop. A strategy climbs only while it passes, so
the expensive rungs run only for strategies that survived the cheap ones.

The calibration, which the module calibration/fit.py implements, reads the fills and the orders file and
splits them by submit time, so that the last 30% of orders serve as a held-out test set. A logistic
regression, which the present study fits on the training orders, predicts whether a limit order fills within
one bar from its distance to the touch, the logarithm of one plus its queue ahead and the realised volatility
of the arrival mids. A least-squares regression predicts the slippage of a market order from the half spread,
the square root of its size divided by ADV, and the volatility, and the empirical quantiles of the
submit-to-fill time give the latency. The orchestrator stores each fitted set as a versioned JSON file,
injects it into NautilusTrader as a fill model and into hftbacktest as the order latency plus an impact charge
on taker fills, and it leaves the vectorised rung uncalibrated.

The parity check, which the module parity.py implements, runs a strategy on vectorbt and NautilusTrader with
identical fees and touch fills. It groups the fills of each engine into executions, where an execution is all
fills of one side in one bar, and it classifies every difference between the two lists. A difference of at
most one tick in price or one lot in size counts as rounding, and an order that one engine fills in several
pieces counts as a partial fill, and these two kinds are benign. A larger price difference, a size difference,
a fee-rate difference, an execution that moves to another bar and an execution that one engine lacks are
material, and the check fails when material differences exceed 5% of the executions or the PnL, which is the
profit and loss, differs by more than 0.1% of the initial cash.

## 4. Experiments and results

The first result is that the calibrated fill models recover the true fill process, while the naive fills of
bar engines miss it badly. As Table E1 shows, the calibrated fill probability differs from the true
probability by 0.021 on average over the held-out orders, with a 95% interval from 0.018 to 0.025 over the 8
seeds, while touch fills, which set the probability to 1, differ by 0.752 and a constant base rate differs by
0.179. Student's t-test for paired samples, which compares the error of the calibrated model with the error of
touch fills under the null hypothesis of equal mean errors, equals -70.12 with 7 degrees of freedom, while the
critical value at 95% confidence is 2.36, so the calibrated model is closer to the truth. The Brier score,
which is the mean squared difference between the predicted probability and the observed outcome of 0 or 1, is
0.1451 for the calibrated model and 0.1446 for the true probability, so the model loses almost nothing to the
generating process. The same table gives a slippage error of 0.013 bps against 0.269 bps for a model that
charges only the half spread and 0.685 bps for zero slippage, and the paired t-test of the first two equals
-72.68 against the same critical value of 2.36. The slippage regression returned a coefficient of 0.991 on the
half spread against a true 1, 24.63 on the square root of participation against a true 25, and a residual
standard deviation of 0.302 bps against a true 0.3, and every true value lies inside the 95% interval of its
estimate.

| Fill model (8 seeds, about 1262 held-out limit orders each) | Mean error of fill probability (95% CI) | Brier score (95% CI) |
|---|---|---|
| calibrated, orders and fills | 0.021 (0.018 to 0.025) | 0.1451 (0.1355 to 0.1547) |
| calibrated, fills only | 0.105 (0.095 to 0.115) | 0.1562 (0.1477 to 0.1646) |
| constant base rate | 0.179 (0.170 to 0.188) | 0.1887 (0.1766 to 0.2007) |
| touch fills (probability 1) | 0.752 (0.729 to 0.776) | 0.7494 (0.7233 to 0.7755) |
| true process (reference) | 0 | 0.1446 (0.1349 to 0.1543) |

The second result is that the orders file, which lists limit orders that never filled, is necessary for an
unbiased fill model. A model that sees only the fills learns from orders that filled at some point, and as
Table E1 shows, 42.7% of those orders filled within one bar, while only 25.1% of all held-out orders did. Its
error of 0.105 is five times the error of 0.021 of the model that also reads the orders file, and the paired
t-test equals -16.77 against the critical value of 2.36, so the present project asks I6 to export unfilled
orders.

The third result is that the step from vectorbt to NautilusTrader changes the Sharpe ratio by an amount that
the bootstrap resolves in every seed, while the later steps do not. As Table E2 shows, the step from vectorbt
to NautilusTrader lowered the Sharpe ratio of the SMA-cross strategy by 1.47 on average and that of the
mean-reversion strategy by 7.19, and the bootstrap interval of the drop excluded zero in all 8 seeds for both
strategies. The step from NautilusTrader to hftbacktest raised the Sharpe ratio by 0.79 and 3.61, but the
interval excluded zero in only 1 and 2 of the 8 seeds. The mean-reversion strategy lost money on every rung,
because its mean Sharpe ratio lay between -47.56 and -29.90, so its vectorised gate failed in 7 of 8 seeds.

| Strategy and rung (8 seeds) | Sharpe ratio (95% CI) | Fills | Sharpe drop against the rung below (95% CI) | Drop interval excludes 0 |
|---|---|---|---|---|
| SMA-cross, vectorbt | 7.21 (-15.53 to 29.94) | 13.6 | not applicable | not applicable |
| SMA-cross, NautilusTrader | 5.74 (-17.27 to 28.75) | 13.6 | 1.47 (1.13 to 1.80) | 8 of 8 |
| SMA-cross, hftbacktest | 6.53 (-16.09 to 29.16) | 13.6 | -0.79 (-1.42 to -0.17) | 1 of 8 |
| SMA-cross, S3 stand-in | 9.54 (-10.50 to 29.58) | 13.4 | -3.01 (-6.54 to 0.53) | 0 of 8 |
| mean reversion, vectorbt | -40.38 (-59.99 to -20.77) | 144.0 | not applicable | not applicable |
| mean reversion, NautilusTrader | -47.56 (-66.91 to -28.22) | 216.1 | 7.19 (6.65 to 7.72) | 8 of 8 |
| mean reversion, hftbacktest | -43.96 (-62.98 to -24.94) | 147.4 | -3.61 (-5.54 to -1.67) | 2 of 8 |
| mean reversion, S3 stand-in | -29.90 (-53.02 to -6.77) | 149.1 | -14.06 (-19.44 to -8.69) | 7 of 8 |

The fourth result is that the gated ladder made no false promotion and needed 17% of the engine time of a
tick-only screen, but it missed 22 of the 51 candidates that the tick rung accepts. Experiment E3 ran all 12
grid points of the SMA-cross strategy and all 18 grid points of the mean-reversion strategy on the vectorised,
event and tick rungs in each of the 8 seeds, which gives 240 candidates. The reference verdict is the gate of
the tick rung, because hftbacktest replays the same tape with queue positions, so it is the most realistic
rung that sees the same prices. As Table E3 shows, the ladder promoted 29 candidates, all of which the tick
rung accepts, and it used 24.5 seconds of engine time, while hftbacktest alone used 140.5 seconds. Vectorbt
alone used 6.5 seconds, but it promoted 6 candidates that the tick rung rejects, which is 13% of its 46
promotions. The present study computed from the rows of experiments.json where the 22 missed candidates
stopped, and 11 of them failed the vectorised gate with 5 to 9 fills, while 11 failed the event gate, in 10
cases only because the Sharpe ratio fell by more than 2.0. These misses therefore come from gate thresholds
that disagree across rungs, because the tick gate accepts 5 fills while the vectorised gate asks for 10.

| Policy (240 candidates, 51 accepted by the tick rung) | Promoted (95% CI) | False promotions (95% CI) | Missed (95% CI) | Engine time |
|---|---|---|---|---|
| gated ladder | 29 (9% to 17% of 240) | 0 (0% to 2%) | 22 (6% to 13%) | 24.5 s |
| vectorbt alone | 46 (15% to 25%) | 6 (1% to 5%) | 11 (3% to 8%) | 6.5 s |
| hftbacktest alone | 51 (17% to 27%) | 0 (reference) | 0 (reference) | 140.5 s |

The intervals in this table are Wilson score intervals, which Wilson (1927) derived for a proportion and which
stay inside 0 and 1. As Table E3 also shows, vectorbt needed 18.9 milliseconds per 1000 bars with a 95%
interval from 16.4 to 21.3 over the 240 candidate runs, while NautilusTrader needed 166.1 (150.0 to 182.3) and
hftbacktest needed 406.5 (387.0 to 426.0). The mean absolute difference between the Sharpe ratio of a
candidate on vectorbt and on hftbacktest was 3.24, while it was 2.38 between NautilusTrader and hftbacktest.

The fifth result is that the calibrated models changed the event and tick rungs, but they did not bring the
event rung closer to the tick rung. As Table E4 shows, the calibrated NautilusTrader run lowered the Sharpe
ratio of the SMA-cross strategy from 7.19 to 5.74, and Student's t-test for paired samples, under the null
hypothesis of equal mean Sharpe ratios, equals -10.30 against the critical value of 2.36. The absolute
difference from the calibrated tick rung was 0.86 with the calibrated event rung and 0.77 with the
uncalibrated one, and the paired t-test of these two errors, under the null hypothesis of equal mean errors,
equals 0.24 with a p-value of 0.82 against the critical value of 2.36, so the difference is not significant.
The mean-reversion strategy gives the same picture, because the errors are 3.61 and 2.30 and the paired t-test
equals 1.35 with a p-value of 0.22. The fees of the calibrated and the uncalibrated NautilusTrader runs differ
by less than 0.04 in every seed, as the rows of experiments.json show, so both runs filled the same orders,
and the lower fill count of the calibrated run of the mean-reversion strategy, 216.1 against 287.5, only
reflects that NautilusTrader splits fewer orders into pieces. The documentation of the FillModel class of
NautilusTrader 1.231.0 states that the parameter prob_fill_on_limit is the probability that a limit order
fills when the market rests on its price, so we can conclude that the calibrated probability of 0.650 cannot
act on limit orders that a bar crosses, and with bar data that is almost every order.

The sixth result is that the parity check found a real disagreement in fill semantics that a profit comparison
misses. As Table E5 shows, the SMA-cross strategy passed in all 8 seeds, because all its 109 executions agreed
within one tick. The mean-reversion strategy failed in all 8 seeds, because 17.1% of its executions had a
material difference, with a 95% interval from 13.8% to 20.4%, and 195 of the 1152 executions had a fill price
that differed by more than one tick. Its PnL differed by only 0.021% of the initial cash, so a check on PnL
alone, with the tolerance of 0.1%, passed it in all 8 seeds. In the run of make reproduce, whose file
reports/reproduce/parity.json lists the first 200 disagreements, all 35 listed fill-price differences gave
vectorbt the better price for the strategy, which agrees with the optimism of touch fills, because they fill a
limit order at the better of the limit price and the bar open.

The seventh result is that the execution-level classifier detected every planted bug, while the two simpler
checks detected fewer than half. Experiment E5 copied the vectorbt run of each strategy and seed, it planted
one bug in the copy, and it compared the copy with the original. The bugs were a fee rate higher by 2 bps on
every fill, every fill one bar late, a price that is 3 ticks worse on 10% of the fills, and 10% of the fills
removed. As Table E5 shows, the classifier flagged all 16 runs of every bug and named the right kind each
time, while a PnL tolerance of 0.1% of the cash flagged 8 of 16 fee bugs, 12 of 16 removed fills and none of
the delayed or mispriced fills. A comparison of fill counts flagged only the removed fills, because the other
bugs keep the number of fills.

| Planted bug (16 runs each) | Parity classifier (95% CI) | PnL tolerance (95% CI) | Fill count (95% CI) |
|---|---|---|---|
| fee rate higher by 2 bps | 16 (81% to 100%) | 8 (28% to 72%) | 0 (0% to 19%) |
| every fill one bar late | 16 (81% to 100%) | 0 (0% to 19%) | 0 (0% to 19%) |
| 3 ticks worse on 10% of fills | 16 (81% to 100%) | 0 (0% to 19%) | 0 (0% to 19%) |
| 10% of fills removed | 16 (81% to 100%) | 12 (51% to 90%) | 16 (81% to 100%) |

## 5. Comparison with analogs

The present study compared the orchestrator with three analogs that it ran, which are vectorbt, NautilusTrader
and hftbacktest used alone, and these single-engine backtests are the alternatives of Table E3. The
specification, section I5, states that NautilusTrader offers research-to-live parity but uses generic fill
models that are not calibrated to the user's own fills, so we can conclude that its default fills correspond
to the touch-fills row of Table E1, whose error of 0.752 is about 36 times the error of the calibrated model.
The specification states for vectorbt that it offers fast vectorised sweeps and walk-forward, and Table E3
supports the speed claim, because vectorbt needed 18.9 milliseconds per 1000 bars against 406.5 for
hftbacktest, while it promoted 6 candidates that the tick rung rejects. The specification limits hftbacktest
to the scope of high-frequency trading (HFT), and Table E3 shows that it was the most expensive rung by a
factor of 2.4 over NautilusTrader.

The present study did not run backtrader, zipline-reloaded, LEAN, backtesting.py, ManifoldBT, ml4t-backtest,
DQengine, Jesse, Freqtrade, Hikyuu, RQAlpha or the vn.py CTA backtester, because each of them would be one
more single-engine backtest with bar-level or generic fills, which Table E1 and Table E3 already represent.
The table below takes the documented behaviour from the specification, section I5, and it adds the evidence of
the present study.

| Analog | Documented behaviour (specification, section I5) | Evidence in the present study |
|---|---|---|
| backtrader, vn.py, Jesse, Freqtrade, RQAlpha | event-driven or bar-level fills, no calibrated fills | touch fills: probability error 0.752 (Table E1) |
| zipline-reloaded, LEAN | classic or production engines, equities or cloud focus | not run, because neither calibrates fills |
| NautilusTrader | Rust core, parity from research to live, generic fill models | run as the event rung; calibrated probability inactive on crossed bars (Table E4) |
| vectorbt | fast vectorised sweeps and walk-forward | run as the vectorised rung: 6 false promotions of 240 (Table E3) |
| backtesting.py, ManifoldBT, ml4t-backtest, DQengine | speed or fill-semantics presets, no calibration | the parity classifier finds the kind of fill-semantics gap that ManifoldBT documents (Table E5) |
| hftbacktest | queue-position and latency fills, HFT scope | run as the tick rung: 406.5 ms per 1000 bars (Table E3) |
| Hikyuu | seven swappable components, combinatorial trials | every candidate of E3 is logged as a trial, which the Research Ledger counts |

The specification, section I5, states that ManifoldBT documents a fill-semantics difference between backtrader
and vectorbt, so we can conclude that such differences are known to occur between engines, and Table E5 shows
the same kind of difference between vectorbt and NautilusTrader for limit orders.

## 6. Decision

The present project adopts the gated fidelity ladder as the default path to promotion, because Table E3 shows
no false promotion in 240 candidates at 17% of the engine time of a tick-only screen, while vectorbt alone
promoted 6 candidates that the tick rung rejects. The project keeps the calibrated fill models and the orders
file, because Table E1 shows a probability error of 0.021 against 0.752 for touch fills and 0.105 for a
fills-only model. The project adopts the execution-level parity classifier instead of a PnL tolerance, because
Table E5 shows 64 detections of 64 planted bugs against 20 for the PnL tolerance. The project does not yet
claim that the calibrated NautilusTrader rung is more realistic than the uncalibrated one, because Table E4
shows no significant reduction of its distance to the tick rung, and the cause is the injection point that
section 4 describes.

## 7. Limitations

The present study rests on synthetic data, and the fill process of E1 is independent of the tape that the
engines replay, so E1 tests whether the calibration recovers a known process, not whether the engines then
reproduce live trading. The reference of E3 is the tick rung, which is itself a model of execution, so a
candidate that the tick rung accepts may still fail in live trading. Each seed covers one day, so the
annualised Sharpe ratios have intervals that span zero for the SMA-cross strategy. The S3 stand-in does not
react to the strategy, so the reactive rung measures a stressed tape and not market impact. The ladder missed
22 of 51 candidates because its gates ask for different minimum fill counts and the event gate caps the Sharpe
drop at 2.0, and the present study did not tune the thresholds, which config/settings.yaml still holds at the
values of section 3.

## 8. References

Politis, D. N. and Romano, J. P. (1994). The stationary bootstrap. Journal of the American Statistical
Association 89(428), 1303-1313.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the
American Statistical Association 22(158), 209-212.

NautilusTrader 1.231.0, module nautilus_trader.backtest.models.fill, documentation of the class FillModel.

vectorbt 1.2.0, Portfolio.from_orders. hftbacktest 2.4.2, BacktestAsset with the power-probability queue
model.

The Tradeq platform plan, quant_platform_plan_full.md, section I5, gives the list of analogs and their
documented limits.
