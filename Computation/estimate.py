"""Functions to make draws."""
import sys

from numpy.random import default_rng

import draw as draw

if sys.platform == "darwin":  # OS X
    sys.path.insert(1, "/Users/jakewren/PycharmProjects/Elasticity/Helpers")

import dgp_helpers as dgp_helpers


def estimate_marginal(distribution, sample):
    """Returns estimate of marginal distribution."""
    marginal = {}
    for element in distribution.keys():
        marginal[element] = sample.count(element) / len(sample)
    return marginal


def estimate_x_conditional_on_z(x_distribution, z_distribution, sample):
    """Returns estimate of conditional distribution."""
    conditional = {}
    for b in z_distribution.keys():
        sliced_sample = [sliced for sliced in sample if sliced[1] == b]
        conditioned_portion = {}
        for a in x_distribution.keys():
            if len(sliced_sample) != 0:
                conditioned_portion[a] = sliced_sample.count((a, b)) / len(
                    sliced_sample
                )
            else:
                conditioned_portion[a] = 0
        conditional[b] = conditioned_portion
    return conditional


def estimate_marginal_from_conditional(
    a_b_conditional_distribution_estimate, a_distribution, b_distribution_estimate
):
    """Returns estimate of marginal distribution from conditional and marginal via
    P[X = x] = \sum_{y} P[X = x | Y = y] P[Y = y].
    """
    marginal = {}
    for a in a_distribution.keys():
        marginal_summer = 0
        for b in b_distribution_estimate.keys():
            marginal_summer += (
                a_b_conditional_distribution_estimate[b][a] * b_distribution_estimate[b]
            )
        marginal[a] = marginal_summer
    return marginal


def bootstrapped_distribution(y_sample, x_sample, z_sample, n):
    """Compute frequencies from the "original" sample."""
    sample_of_tuples = list(zip(y_sample, x_sample, z_sample))  # 3-tuple
    empirical_distribution = {}
    for a in sample_of_tuples:
        empirical_distribution[a] = sample_of_tuples.count(a) / n
    return empirical_distribution


def estimator(
    y_sample, z_distribution, z_sample, x_distribution, x_z_support, x_z_sample, size
):
    """Estimate the population moments vector and a few other distributions."""

    z_distribution_estimate = estimate_marginal(z_distribution, list(z_sample))

    x_z_conditional_distribution_estimate = estimate_x_conditional_on_z(
        x_distribution, z_distribution, x_z_sample
    )

    X_distribution_estimate = estimate_marginal_from_conditional(
        x_z_conditional_distribution_estimate, x_distribution, z_distribution_estimate
    )

    # Estimate the component of the vector of population moments corresponding to X=x and Z=z with the binning estimator
    # (i.e., sample mean of Y among X = x and Z = z).
    vector_of_moments_estimates = {}
    for x, z in x_z_support:
        running_conditional_total = 0
        X_equals_x_and_Z_equals_z_counter = 0
        for i in range(size):
            if (x == x_z_sample[i][0]) & (z == x_z_sample[i][1]):
                running_conditional_total += y_sample[i]
                X_equals_x_and_Z_equals_z_counter += 1
        if X_equals_x_and_Z_equals_z_counter == 0:
            vector_of_moments_estimates[(x, z)] = 0.0  # Edge case
        else:
            vector_of_moments_estimates[(x, z)] = (
                running_conditional_total / X_equals_x_and_Z_equals_z_counter
            )

    return (
        vector_of_moments_estimates,
        X_distribution_estimate,
        z_distribution_estimate,
        x_z_conditional_distribution_estimate,
    )


def vector_of_moments_estimator(
    x_distribution,
    z_distribution,
    x_z_support,
    sample_size,
    b_conditional_on_x_z_distribution,
    x_conditional_on_z_distribution,
    rng=None,
):
    """Returns an estimate of the vector of moments, and three other distributions (computed via sample means)."""
    rng = default_rng(rng)

    # Draw some Z's.
    z_sample = draw.discrete_marginal_draws(
        z_distribution,
        sample_size,
        rng=rng,
    )

    # Draw X given Z draws.
    x_z_sample, x_sample = draw.discrete_conditional_draws(
        x_conditional_on_z_distribution, z_sample, rng=rng
    )
    # Draw B given X and Z.
    b_x_z_sample = draw.discrete_conditional_draws(
        b_conditional_on_x_z_distribution, x_z_sample, rng=rng
    )[0]

    # Draw u.
    u = draw.logistic_draws(sample_size, rng=rng)

    # Compute Y.
    y_sample = [
        1
        if (b_x_z_sample[i][0][0] + b_x_z_sample[i][0][1] * b_x_z_sample[i][1][0])
        >= u[i]
        else 0
        for i in range(sample_size)
    ]

    bootstrapped = bootstrapped_distribution(y_sample, x_sample, z_sample, sample_size)

    (
        vector_of_moments_estimates,
        X_distribution_estimate,
        z_distribution_estimate,
        x_z_conditional_distribution_estimate,
    ) = estimator(
        y_sample,
        z_distribution,
        z_sample,
        x_distribution,
        x_z_support,
        x_z_sample,
        sample_size,
    )

    return (
        vector_of_moments_estimates,
        X_distribution_estimate,
        z_distribution_estimate,
        x_z_conditional_distribution_estimate,
        bootstrapped,
    )


def bootstrap_vector_of_moments_estimator(
    bootstrap_distribution,
    z_distribution,
    x_distribution,
    x_z_support,
    size,
    S,
    rng=None,
):
    """Returns all necessary bootstrapped estimates."""
    rng = default_rng(rng)
    bootstraped = {}
    for s in range(S):
        bs_sample = draw.discrete_marginal_draws(bootstrap_distribution, size, rng=rng)
        y_sample, x_sample, z_sample = tuple(dgp_helpers.zip_helper(bs_sample))
        x_z_sample = list(zip(x_sample, z_sample))
        (
            vector_of_moments_estimates,
            X_distribution_estimate,
            z_distribution_estimate,
            x_z_conditional_distribution_estimate,
        ) = estimator(
            y_sample,
            z_distribution,
            z_sample,
            x_distribution,
            x_z_support,
            x_z_sample,
            size,
        )

        bootstraped[s] = (
            vector_of_moments_estimates,
            X_distribution_estimate,
            z_distribution_estimate,
            x_z_conditional_distribution_estimate,
        )
    return bootstraped
