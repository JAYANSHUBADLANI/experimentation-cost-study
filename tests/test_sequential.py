"""Sequential procedures against published values and against their own claims."""

import numpy as np
import pytest

from expcost.sequential import (
    _crossing_probability,
    alpha_spending_boundaries,
    always_valid_p_value,
    classical_obrien_fleming_boundaries,
    inflation_factor,
    msprt_likelihood_ratio,
    obrien_fleming_spending,
)


# Tabulated in the group sequential literature for five equally spaced looks
# at a two sided alpha of 0.05.
PUBLISHED_OBF_K5 = np.array([4.5617, 3.2256, 2.6296, 2.2777, 2.0378])


def test_classical_boundaries_match_published_values():
    computed = classical_obrien_fleming_boundaries(5, 0.05)
    assert np.abs(computed - PUBLISHED_OBF_K5).max() < 0.01


def test_classical_boundaries_are_constant_on_the_score_scale():
    k = np.arange(1, 6)
    computed = classical_obrien_fleming_boundaries(5, 0.05)
    score_scale = computed * np.sqrt(k / 5)
    assert np.allclose(score_scale, score_scale[0], rtol=1e-6)


@pytest.mark.parametrize("looks", [3, 5, 10])
def test_alpha_spending_boundaries_spend_exactly_alpha(looks):
    fractions = np.arange(1, looks + 1) / looks
    boundaries = alpha_spending_boundaries(fractions, 0.05)
    assert _crossing_probability(fractions, boundaries) == pytest.approx(0.05, abs=1e-3)


def test_alpha_spending_boundaries_decrease():
    boundaries = alpha_spending_boundaries(np.arange(1, 15) / 14, 0.05)
    assert np.all(np.diff(boundaries) < 0)
    # the last look is close to, but stricter than, the fixed horizon value
    assert 1.96 < boundaries[-1] < 2.2


def test_spending_function_is_monotone_and_ends_at_alpha():
    t = np.linspace(0.01, 1.0, 50)
    spent = obrien_fleming_spending(t, 0.05)
    # non decreasing everywhere: at very small t the spend underflows to
    # exactly zero in double precision, which is the correct behaviour here,
    # O'Brien-Fleming spends essentially nothing at the first looks.
    assert np.all(np.diff(spent) >= 0)
    assert np.all(np.diff(spent[t >= 0.1]) > 0)
    assert spent[-1] == pytest.approx(0.05, abs=1e-9)
    assert spent[0] < 1e-6


def test_msprt_likelihood_ratio_by_hand():
    # with effect 0, the ratio is sqrt(V / (V + tau^2)) exactly.
    ratio = msprt_likelihood_ratio(np.array([0.0]), np.array([2.0]), 3.0)
    assert ratio[0] == pytest.approx(np.sqrt(2.0 / 5.0))


def test_msprt_p_value_is_monotone():
    effect = np.array([0.1, 0.4, 0.2, 0.9, 0.3])
    variance = np.array([1.0, 0.5, 0.3, 0.2, 0.1])
    p = always_valid_p_value(effect, variance, 0.25)
    assert np.all(np.diff(p) <= 0)
    assert np.all((p >= 0) & (p <= 1))


def test_msprt_controls_error_under_continuous_monitoring():
    """The always valid claim, checked by simulating the whole path."""
    rng = np.random.default_rng(7)
    reps, looks = 4000, 50
    per_look = 200
    tau_squared = 0.02
    crossed = np.zeros(reps, dtype=bool)
    for start in range(0, reps, 500):
        block = min(500, reps - start)
        data = rng.normal(0.0, 1.0, size=(block, looks * per_look))
        cumulative = np.cumsum(data, axis=1)[:, per_look - 1 :: per_look]
        counts = np.arange(1, looks + 1) * per_look
        effect = cumulative / counts
        variance = 1.0 / counts
        p = always_valid_p_value(effect, np.broadcast_to(variance, effect.shape), tau_squared)
        crossed[start : start + block] = (p < 0.05).any(axis=1)
    rate = crossed.mean()
    # bounded by alpha, and in practice comfortably under it
    assert rate <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / reps)


def test_inflation_factor_is_a_modest_premium():
    factor = inflation_factor(np.arange(1, 6) / 5, 0.05, 0.80)
    assert 1.0 < factor < 1.15
