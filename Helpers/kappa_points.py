"""Heuristics for choosing an initial value for optimization."""

import numpy as np
from numpy.random import default_rng  # rng = Random Number Generator


def starting_points(target_tuner, norm, sample_size, rng=None, n_guesses=59):
    """Initial guesses for \hat{\kappa} search; norm criterion dependent."""
    rng = default_rng(rng)
    if norm == "1":
        if target_tuner == "inf":
            x0s = rng.triangular(
                left=np.power(sample_size, 1 / 4),
                mode=np.power(sample_size, 1 / 2),
                right=np.power(sample_size, 3 / 2),
                size=n_guesses,
            )
        else:
            x0s = rng.triangular(
                left=np.log(np.power(sample_size, 1 / 4)),
                mode=np.log(sample_size),
                right=np.log(np.power(sample_size, 5 / 2)),
                size=n_guesses,
            )
    else:  # Squared Euclidean norm
        x0s = rng.triangular(
            left=2 * np.power(sample_size, 1 / 2),
            mode=sample_size / np.log(sample_size),
            right=sample_size / np.log(np.power(sample_size, 1 / 2)),
            size=n_guesses,
        )

    # Space out the guesses.
    idx = np.round(np.linspace(0, len(x0s) - 1, (n_guesses + 1) // 20)).astype(int)
    return x0s[idx]


def grid_points(norm):
    # Brute force.
    if norm == "1":
        return [
            (slice(4, 6, 1),)
        ]  # List for convenience (i.e., "nesting" with the adaptive approach).
    else:
        return [(slice(4, 6, 1),)]
