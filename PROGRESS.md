# Progress log

## Done

**The metric simulator.** Three families: zero inflated revenue, low rate
Bernoulli conversion, overdispersed negative binomial sessions. A controllable
true effect, a pre period covariate whose Pearson correlation with the outcome
is a settable parameter, and a discrete stratum derived from the pre period
value.

**Calibration.** Fitted the revenue positive part to 98,815 Olist order values.
Tried a single lognormal first and rejected it: empirical skew is 9.19, the
lognormal implies about 3.8, and the coefficient of variation comes out 30
percent low, which would have understated every revenue sample size by a factor
of 1.91. Replaced it with a spliced lognormal body and generalized Pareto tail
above the 95th percentile. The fit now reproduces the empirical mean, standard
deviation, 99th and 99.9th percentiles closely.

**Correlation by quadrature, not simulation.** The first version solved for the
Gaussian copula parameter by simulating and root finding on the result. That
was wrong in a way that took a while to see: the objective was noisy, so the
solver landed on a root of the noise, and at a target correlation of 0.7 the
achieved value was 0.035 low, which is fifteen times the Monte Carlo standard
error. Replaced it with exact quadrature per family: Gauss-Legendre over the
positive region for the continuous case, a bivariate normal orthant for
Bernoulli, an upper orthant sum for the count case. The first attempt at the
continuous case used Gauss-Hermite over the whole line and oscillated by 0.9 in
the mean across node counts, because a zero inflated outcome is a step function
of the latent normal and Hermite quadrature converges badly across a step.
Integrating only over the smooth positive region fixed it, and now agrees with
a closed form to five significant figures and with a 30 million sample Monte
Carlo to within one standard error.

**Estimators.** Difference in means with a Welch test, CUPED, Lin's interacted
regression adjustment with HC2 standard errors, stratified and post stratified
estimation, and CUPED stacked with strata. Each is tested against a hand
computable case, and against statsmodels or scipy where an equivalent exists.
Lin's estimator matches statsmodels HC2 to a relative tolerance of 1e-8.

**Performance.** The first working pipeline ran at 0.758 microseconds per user,
which put the grid at over an hour on one core. Profiling found three things:
Lin's estimator was a third of the time because it built an n by 4 design and
multiplied through it repeatedly, scipy's generic quantile wrappers cost two to
three times their closed forms, and `nbinom.ppf` alone cost 3.2 microseconds
per draw, eight times everything else combined. Rewrote Lin as two within arm
regressions, which span the same column space so the estimate, residuals and
leverages are identical, wrote the quantile pieces out in closed form, and
replaced the negative binomial quantile with a cached cdf table and a binary
search that agrees with scipy exactly. Result is 0.394 microseconds per user,
and 7.76 million users per second across 7 workers. The whole test suite
passed unchanged through every one of those rewrites, which is the only reason
they were safe to make.

**The power and sample size study.** 11,219,604,000 simulated users, 1,500
replications per design point, 1,445 seconds. Required sample size computed two
independent ways, off the measured power curve and from the measured variance,
agreeing to within 3 percent at worst. CUPED saves about 25 percent of users at
a correlation of 0.5 across every scenario. Post stratification saves 10 percent
on the heavy tailed metric and 23 percent on the count metric, which is the
more interesting number of the two: quintiles discard the within stratum
variation, and on a metric whose variance lives in the tail that is most of it.

**Sample ratio mismatch.** Chi square goodness of fit against the intended
allocation, with a test asserting that a deliberate 85/15 imbalance does not
fire and a drift to 84/16 does. Detection power measured over 4,000
replications per cell and cross checked against the closed form non central chi
square. At a million users the smallest detectable drift is 0.22 percentage
points.

**Sequential procedures.** The mSPRT with a normal mixture, and Lan-DeMets
alpha spending with the O'Brien-Fleming spending function computed by the
Armitage, McPherson and Rowe recursion. Validated the recursion against the
tabulated classical O'Brien-Fleming boundaries for five looks, reproducing them
to within 0.004.

**Bandit and peeking machinery.** Both implemented and unit tested. The bandit
collapses to four counts per round because outcomes are Bernoulli and users
within a round are exchangeable, so all replications run in lockstep.

**The peeking study.** Ran at the full 20,000 replications per metric, 14 daily
looks at 10,000 users a day, 8.4 billion simulated users in 132 seconds. Under
the null, naive daily peeking rejects at 0.2165, 0.2203 and 0.2160 across the
three metrics against a nominal 0.05, so reading a dashboard every day and
stopping at the first significant result inflates the false positive rate by a
factor of about 4.3. The fixed horizon control sits on 0.0486 to 0.0510, which
is what says the simulator and the test are behaving rather than the result
being an artefact. Alpha spending lands on nominal, 0.0490 to 0.0510. The mSPRT
is conservative at 0.0041 to 0.0130, which is the price of validity at every
stopping time rather than at fourteen committed ones. O'Brien-Fleming alpha
spending needs 1.060 times the fixed horizon sample to hold 80 percent power, so
the correction costs six percent of sample against a fourfold error it prevents.

**The bandit comparison.** Ran at 2,000 replications per cell, 200 rounds of
500 users, across three exploration floors. Under the null the naive interval
covers at 0.918, 0.872 and 0.855 as the floor drops from 5 percent to 1 percent
to 0.2 percent, against a nominal 0.95. The adaptively weighted interval holds
0.946, 0.941 and 0.935 over the same floors, and the balanced fixed horizon
control sits on nominal throughout, which is what says the harness is right and
the bandit is the thing breaking coverage. Regret saved is 145.4, 151.3 and
151.2 of a possible 200, so tightening the floor below 1 percent buys no
meaningful regret and costs measurable coverage.

## Pending

- Decide whether the mSPRT's tau squared should be swept rather than fixed at
  the square of a 5 percent relative effect. The run above uses the fixed value,
  which is pre specified and defensible, but the conservatism it produces is
  partly a consequence of that choice rather than of the procedure.
- Written `docs/guide.md`, the practitioner facing summary, from the completed
  peeking, bandit, CUPED and SRM tables.
- The Criteo real data arm, still blocked on the failed download.
- Verify determinism properly by running the full grid twice and diffing the
  result tables. Name addressed seeding makes this a design property already,
  but a design property is not a verified one and I should not claim it as one.

## Open decisions

**Where the peeking study's tau squared should come from.** The mSPRT needs a
mixing variance fixed before the experiment starts, and it controls which
effect sizes the test detects soonest. I have set it to the square of a 5
percent relative effect, which is defensible and pre specified, but it is a
choice that a reader could reasonably want to see swept rather than fixed. If
the runtime budget allows I would rather sweep it.

**The exploration floor is the headline, decided on the full run rather than
the smoke test.** The full run confirms what the smoke test hinted: coverage of
the naive interval tracks the floor almost monotonically, 0.918 to 0.872 to
0.855, while regret saved is flat between the 1 percent and 0.2 percent floors,
151.3 against 151.2. So the floor is a real dial with a real cost on one side
and almost no benefit on the other past a point, which is more useful to an
analyst than the regret comparison it was originally a robustness check for.
The bandit section in the README now leads with it.

**Whether post stratification deserves a second parameterisation.** Five
quintiles is the obvious default and it is what I measured, but the gap between
it and CUPED on the heavy tailed metric is large enough that the number of
strata is clearly doing work. Ten strata would narrow it. Reporting one bin
count and calling it "post stratification" slightly undersells the method.
