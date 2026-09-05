"""Corrections across metrics, against hand computable cases and their guarantees."""

import numpy as np
import pytest
from statsmodels.stats.multitest import multipletests

from expcost.multiplicity import adjusted_p_values, benjamini_hochberg, holm_bonferroni


def test_holm_by_hand():
    # sorted p values 0.001, 0.02, 0.04 against thresholds 0.05/3 = 0.0167,
    # 0.05/2 = 0.025 and 0.05. Each clears its own threshold, so all three go.
    p = np.array([0.02, 0.001, 0.04])
    assert list(holm_bonferroni(p, 0.05)) == [True, True, True]


def test_holm_stops_at_the_first_failure():
    # sorted 0.001, 0.03, 0.04. The second one fails against 0.025, and Holm is
    # a step down procedure, so nothing after it is rejected either, even though
    # 0.04 would have cleared its own threshold of 0.05.
    p = np.array([0.03, 0.001, 0.04])
    assert list(holm_bonferroni(p, 0.05)) == [False, True, False]


def test_benjamini_hochberg_by_hand():
    # thresholds 0.0167, 0.0333, 0.05. p = 0.001, 0.02, 0.04 all pass at the
    # largest passing rank, which is rank 3, so all three are rejected.
    p = np.array([0.02, 0.001, 0.04])
    assert list(benjamini_hochberg(p, 0.05)) == [True, True, True]


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(3)
    p = rng.random(12)
    mine = holm_bonferroni(p, 0.05)
    reference = multipletests(p, alpha=0.05, method="holm")[0]
    assert list(mine) == list(reference)


def test_bh_matches_statsmodels():
    rng = np.random.default_rng(4)
    p = rng.random(12)
    mine = benjamini_hochberg(p, 0.05)
    reference = multipletests(p, alpha=0.05, method="fdr_bh")[0]
    assert list(mine) == list(reference)


@pytest.mark.parametrize("method,sm_name", [("holm", "holm"), ("bh", "fdr_bh")])
def test_adjusted_p_values_match_statsmodels(method, sm_name):
    rng = np.random.default_rng(5)
    p = rng.random(20) ** 2
    mine = adjusted_p_values(p, method)
    reference = multipletests(p, method=sm_name)[1]
    assert np.allclose(mine, reference, atol=1e-12)


def test_holm_controls_the_family_wise_error_rate():
    """All nulls true: at most 5 percent of families should have any rejection."""
    rng = np.random.default_rng(11)
    families = 4000
    p = rng.random((families, 6))
    any_rejection = np.array([holm_bonferroni(row, 0.05).any() for row in p])
    rate = any_rejection.mean()
    assert rate <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / families)


def test_bh_is_at_least_as_permissive_as_holm():
    rng = np.random.default_rng(12)
    for _ in range(200):
        p = rng.random(8) ** 2
        assert benjamini_hochberg(p, 0.05).sum() >= holm_bonferroni(p, 0.05).sum()
