import numpy as np

from expcost.seeds import ROOT_SEED, generator, seed_sequence


def test_same_address_is_reproducible():
    assert generator("power", "revenue", 3).random() == generator("power", "revenue", 3).random()


def test_different_addresses_differ():
    a = generator("power", "revenue", 3).random(5)
    assert not np.array_equal(a, generator("power", "revenue", 4).random(5))
    assert not np.array_equal(a, generator("power", "conversion", 3).random(5))


def test_streams_are_independent():
    draws = np.array([generator("independence", i).normal(size=2000) for i in range(20)])
    corr = np.corrcoef(draws)
    off_diagonal = corr[~np.eye(20, dtype=bool)]
    assert np.abs(off_diagonal).max() < 0.1


def test_root_seed_is_the_only_entropy():
    assert seed_sequence("x").entropy == ROOT_SEED
