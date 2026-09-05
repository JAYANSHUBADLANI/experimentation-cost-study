"""Deterministic random streams.

Every random number in this project descends from ROOT_SEED through
numpy.random.SeedSequence. The legacy global numpy seed is never used.

A stream is addressed by name, not by draw order, so adding a scenario or
reordering the grid does not change the numbers any other scenario sees.
The address is turned into a spawn_key by hashing the label with blake2b,
which is stable across processes and platforms (unlike hash()).
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

ROOT_SEED = 20240917
_KEY_BITS = 32


def _label_to_key(label: str) -> int:
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**_KEY_BITS)


def seed_sequence(*labels: object) -> np.random.SeedSequence:
    """Build the SeedSequence addressed by these labels.

    Integers are used as-is so that a replication index stays legible in the
    spawn key. Everything else is hashed to a 32 bit key.
    """
    key: list[int] = []
    for label in labels:
        if isinstance(label, (int, np.integer)) and not isinstance(label, bool):
            key.append(int(label) % (2**_KEY_BITS))
        else:
            key.append(_label_to_key(str(label)))
    return np.random.SeedSequence(entropy=ROOT_SEED, spawn_key=tuple(key))


def generator(*labels: object) -> np.random.Generator:
    """Return the PCG64 generator addressed by these labels."""
    return np.random.Generator(np.random.PCG64(seed_sequence(*labels)))


def generators(labels: Iterable[object], n: int) -> list[np.random.Generator]:
    """Return n independent generators under one label address."""
    return [generator(*labels, i) for i in range(n)]
