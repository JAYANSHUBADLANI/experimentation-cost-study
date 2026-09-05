# What to do about it: a practitioner's guide

Everything here is a conclusion from the measurements in this repository. Where
a number appears it came out of the code, and where I do not have a number I say
so rather than filling the gap with a rule of thumb.

## The one paragraph version

Spend your effort on variance reduction and your discipline on when you look. A
pre period covariate correlated 0.5 with the outcome removes about a quarter of
the users you need, for the cost of a feature pipeline. Looking at the result
every day and stopping when it goes significant inflates your false positive
rate from 5 percent to about 22, and fixing that costs six percent more sample.
Those two levers multiply: do both and you run at roughly 0.79 of the sample you
started with, and you are allowed to watch it.

## Decide these four things before the experiment starts

**1. Is a pre period covariate worth building?**

Only if you can get the correlation above about 0.3. The saving is
`1 - rho squared` on variance, which crosses 10 percent of users at rho = 0.316,
and that is the closed form, confirmed here to within 0.5 percent of the
variance ratio across the whole measured grid.

| pre period correlation | users saved |
|---|---|
| 0.1 | 1 percent |
| 0.3 | 9 percent |
| 0.5 | 25 percent |
| 0.7 | 49 percent |
| 0.9 | 81 percent |

Below 0.3 the plumbing is probably not worth it. Above 0.7 each further 0.1 of
correlation is worth more than the last, so if you already have good pre period
signal, push for better rather than settling.

**2. Which adjustment?**

CUPED or Lin's interacted regression. They land on top of each other to three
significant figures here, which is expected under random assignment, and both
beat post stratification. Use whichever your stack makes easier.

Do not reach for post stratification by default. On the heavy tailed revenue
metric it saves about 10 percent where CUPED saves 25, because cutting a
continuous covariate into quintiles throws away the within stratum variation and
on a metric whose variance lives in the tail that is most of the signal. On the
count metric the gap is much smaller, 23 against 25. So the penalty depends on
the metric's shape, and it is largest exactly where sample sizes hurt most.

**3. How often will you look, and what will you do when you look?**

Decide this before you start, because it changes which test you need.

- *Look once, at a pre committed sample size.* Use the fixed horizon test. It
  rejects at 0.049 to 0.051 under the null here, which is what it promises.
- *Look on a fixed schedule and might stop early.* Use O'Brien-Fleming alpha
  spending. It holds nominal, 0.049 to 0.051 across the three metrics, and costs
  1.060 times the fixed horizon sample to keep 80 percent power.
- *Look whenever you like, including because someone asked.* Use the mSPRT. It
  is valid at every stopping time, at the price of being conservative: it
  rejected at 0.004 to 0.013 where 0.05 was allowed, so it will be slower to
  call a real effect.
- *Look every day and stop at the first significant result.* This is the one to
  avoid. It rejects a true null about 22 percent of the time.

**4. What is your alerting threshold for a sample ratio mismatch?**

Use the intended split, not 50/50. A deliberate 85/15 is not a mismatch. Then
pick a threshold knowing what it can detect:

| users in the experiment | smallest detectable drift |
|---|---|
| 10,000 | 2.16 percentage points |
| 100,000 | 0.68 |
| 1,000,000 | 0.22 |

At a p value threshold of 0.0005 with 80 percent power. Below about 50,000 users
an SRM check will only catch gross failures, so do not read a quiet check as
evidence of a healthy assignment at small sample sizes.

When one fires, discard the experiment rather than adjusting it. A mismatch
means users went missing non randomly and there is no correction for a selection
process you cannot see.

## If someone proposes a bandit

Ask what the exploration floor is, and then ask who is going to compute the
confidence interval.

A bandit does buy the regret it promises: about 150 of a possible 200 units in
the setup measured here, sending 88 percent of traffic to the better arm. But
the allocation depends on the outcomes already seen, so the arms are not
independent samples any more, and a naive interval computed afterwards does not
cover what it says it covers.

| exploration floor | naive interval coverage | adaptively weighted |
|---|---|---|
| 5 percent | 0.918 | 0.946 |
| 1 percent | 0.872 | 0.941 |
| 0.2 percent | 0.855 | 0.935 |

Nominal is 0.95. So at a tight floor, roughly one in seven of your 95 percent
intervals is wrong rather than one in twenty.

Two things follow. Use an adaptively weighted estimator rather than the naive
one, and do not tighten the exploration floor below about 1 percent: between a 1
percent floor and a 0.2 percent floor the regret saved goes from 151.3 to 151.2,
which is nothing, while naive coverage falls from 0.872 to 0.855.

## What I would not conclude from this

The numbers here are simulation, calibrated to the shape of a real revenue
distribution but not measured on a real experiment. The metric families are
three, the effect sizes three, and the bandit is one algorithm rather than a
survey. Nothing here covers interference between users, ratio metrics, or long
term effects, and the real data arm of this project has not been run.

What the simulation does support is the relative ordering of the techniques and
the rough size of each effect, because those are properties of variance and
stopping rules rather than of any particular business. Treat the exact figures
as accurate for the distributions stated and the ordering as the durable part.
