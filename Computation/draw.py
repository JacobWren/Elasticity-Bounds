"""Functions to calculate estimates."""

from numpy.random import default_rng
import numpy as np

import dgp_helpers as dgp_helpers


def discrete_marginal_draws(distribution, sample_size=1, rng=None):
    """Returns a sample for a discrete random variable."""
    rng = default_rng(rng)

    support, probabilities = dgp_helpers.list_converter(
        [distribution.keys(), distribution.values()]
    )
    if (
        dgp_helpers.support_dimension(distribution) > 1
    ):  # np.random.choice() can only handle 1 dimensional support; workaround -- draw indices.
        idx = rng.choice(np.arange(len(support)), size=sample_size, p=probabilities)
        return [support[i] for i in idx]
    else:
        return rng.choice(support, size=sample_size, p=probabilities)


def logistic_draws(sample_size, loc=0, scale=1, rng=None):
    """Returns a sample from a logistic distribution."""
    rng = default_rng(rng)
    return rng.logistic(loc, scale, sample_size)


def discrete_conditional_draws(
    conditional_distribution,
    conditional_variable_sample,
    rng=None,
):
    """Returns a random sample for a discrete rv conditional on another (potentially multiple) discrete rv(s)."""
    rng = default_rng(rng)
    sample = (
        []
    )  # While the conditioning variable(s) is already sampled, the sample will be stored as a tuple.
    non_conditioning_variable_sample = []
    for (
        conditional
    ) in (
        conditional_variable_sample
    ):  # Sample conditionally (i.e. condition on each element in "sample").
        non_conditioning_variable_sampled = discrete_marginal_draws(
            conditional_distribution[conditional],  # Slice
            sample_size=1,
            rng=rng,
        )
        sample.append(  # Store as a tuple.
            (non_conditioning_variable_sampled[0], conditional)
        )
        non_conditioning_variable_sample.append(non_conditioning_variable_sampled[0])
    return sample, non_conditioning_variable_sample
