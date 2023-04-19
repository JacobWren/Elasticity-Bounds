import itertools
import math
import sys

import numpy as np
from scipy.stats import norm

from functools import reduce
import operator

if sys.platform == "darwin":  # OS X
    sys.path.insert(1, "/Users/jakewren/PycharmProjects/UChicago/Mixed_Logit/Helpers")

import dgp_helpers as dgp_helpers


# Functions to create the DGP.
def create_support(size, element_types, scalar=None):
    """Returns a list of the support for a discrete random variable (rv).

    Notes:
    size -- cardinality of the support
    element_types -- 'integers' or 'floats'

    Optional arguments:
    scalar --  multiples all elements in the support (default None)
    """
    if element_types == "integers":
        support = list(range(size))
        return support
    elif element_types == "floats":
        support = [scalar * (i / (size - 1)) if i != 0 else 0 for i in range(size)]
        support.sort()
        return support


def create_probabilities(support, case="uniform"):
    """Returns a list of probabilities of support of a discrete rv.

    Optional argument:
    case --  mapping from support to probabilities (default is 'uniform')
    """
    if case == "uniform":  # Other cases can be easily added.
        probs = [1 / len(support)] * len(support)
    elif case == "X non-uniform":  # Specify probabilities "by-hand".
        probs = [0.10, 0.35, 0.55]
    elif case == "Z non-uniform":  # Specify probabilities "by-hand".
        probs = [0.35, 0.65]
    elif case == "B0 non-uniform":
        probs = [
            i / (((len(support) + 1) / 2) * len(support))
            for i in range(1, len(support) + 1)
        ]
    elif case == "B1 non-uniform":
        probs = [
            (len(support) + 1 - i) / (((len(support) + 1) / 2) * len(support))
            for i in range(1, len(support) + 1)
        ]

    assert math.isclose(sum(probs), 1, abs_tol=1e-16), "Probabilities do not sum to 1."
    return probs


def marginal_distribution(size, element_types, scalar=None, case="uniform"):
    """Create marginal (i.e., unconditional) distribution.

    Notes:
    The distribution consists of a dictionary with support as the keys and probability of each corresponding element as
    the values.

    Returns:
    rv's support and distribution (as described above).
    """
    support = create_support(size, element_types, scalar)
    probabilities = create_probabilities(support, case)

    return support, {support[i]: probabilities[i] for i in range(len(support))}


def two_var_joint_support(var0_distribution, var1_distribution, lag=False):
    """Returns joint support of two discrete random variables."""
    var0_distribution = dgp_helpers.dict_keys_to_list(var0_distribution)
    var1_distribution = dgp_helpers.dict_keys_to_list(var1_distribution)

    if lag:
        start = 1
    else:
        start = 0

    return list(itertools.product(var0_distribution, var1_distribution[start:]))


def joint_distribution_from_2_rvs(
    scalar_0,
    size_0,
    element_types_0,
    scalar_1,
    size_1,
    element_types_1,
    independent_flag=True,
    case=("uniform", "uniform"),
):
    """Create joint distribution from two marginals.

    Notes:
    The distribution consists of a dictionary with support as the keys and probability of each corresponding element as
    the values. This could be generalized to a joint for n variables, not just two.

    Returns:
    The joint support and distribution.
    """
    support_0 = create_support(size_0, element_types_0, scalar_0)
    support_1 = create_support(size_1, element_types_1, scalar_1)

    probabilities_0 = create_probabilities(support_0, case[0])
    probabilities_1 = create_probabilities(support_1, case[1])

    # Compute the Cartesian product; each value in the support is a 2-vector.
    support = two_var_joint_support(support_0, support_1)
    # Cartesian product again, but now multiply vector elements.
    if independent_flag:
        probabilities = [
            s_0 * s_1
            for s_0, s_1 in itertools.product(probabilities_0, probabilities_1)
        ]

    assert math.isclose(
        sum(probabilities), 1, abs_tol=1e-16
    ), "Probabilities do not sum to 1."
    return support, {support[i]: probabilities[i] for i in range(len(support))}


def dependence_helper(x_independent_z_distribution, x_support):
    r"""Model P[X \leq x | Z = z] as \Phi(g(x) + x_z_dependence_controller \cdot z) and find g.

    Description:
    -g is chosen such that we get back P[X = x] = when x_z_dependence_controller = 0.
    -g is stored in a dictionary with domain as the keys and range as the values.

    Returns:
    g as a dictionary.
    """
    g_out = norm.ppf(np.cumsum(list(x_independent_z_distribution.values())[:-1]))
    return dict(zip(x_support[:-1], g_out))


def prob_x_given_z(
    x_support,
    z,
    g,
    x_z_dependence_controller,
):
    """Find probability that X = x | Z = z for all x.

    Description:
    -Models the dependence between X and Z.

    Notes:
    -x_z_dependence_controller -- measures strength of the dependence between X and Z.
    -x_z_dependence_controller = 0 => X and Z are independent. If Z is an instrument, then this is equivalent to not
    having Z -- this fact allows the cases with and without an instrument to be nested.

    Returns:
    Probability distribution that X = x | Z = z for all x in X and a given z.

    """
    conditioned_distribution = (
        {}
    )  # For a single Z slice (i.e., not the entire conditional).
    for x_count in range(
        len(x_support)
    ):  # Loop over support of X and find probabilities given that Z = Z.
        if len(x_support) - 1 == x_count:  # Last element of support "edge case".
            conditioned_distribution[x_support[x_count]] = 1 - sum(
                list(conditioned_distribution.values())
            )
        else:
            x_conditioned_cdf_vals = []  # CDF of X @ elem, given Z = z.
            consecutive_support = x_support[
                x_count - dgp_helpers.isolate_zero(x_count): x_count + 1
            ]
            for elem in consecutive_support:
                a_cdf_at_elem = norm.cdf(
                    g[elem] + x_z_dependence_controller * z
                )  # cdf of X <= elem | Z = z.
                x_conditioned_cdf_vals.insert(0, a_cdf_at_elem)

            conditioned_distribution[x_support[x_count]] = reduce(
                operator.__sub__, x_conditioned_cdf_vals
            )  # P[X=x|Z=z]

    return conditioned_distribution


def x_conditional_on_z(
    x_independent_z_distribution, x_support, z_support, x_z_dependence_controller
):
    r"""Finds Probability that X = x | Z = z for all (x, z).

    Description:
    The distribution consists of a nested dictionary with support of the conditioning variable as the "outer"
    key and value as a dictionary -- where the keys are the variable being conditioned on and corresponding
    probabilities as the values. The nested structure maps nicely for simulation.
    The 'degree' of dependence between the two discrete rv's is controlled by A_B_dependence_controller. Setting
    x_z_dependence_controller = 0, makes Z irrelevant for finding P[X=x].

    Returns:
    Conditional distribution of X = x | Z = z for all (x, z).
    """
    x_conditional_on_z_distribution = {}
    g = dependence_helper(
        x_independent_z_distribution, x_support
    )  # g is from modeling P[X \leq x | Z = z] as
    # \Phi(g(x) + \dependence_controller \cdot z).
    for z in z_support:
        conditioned_portion = prob_x_given_z(
            x_support,
            z,
            g,
            x_z_dependence_controller,
        )

        assert math.isclose(
            sum(conditioned_portion.values()), 1, abs_tol=1e-16
        ), "Probabilities do not sum to 1."
        x_conditional_on_z_distribution[z] = conditioned_portion
    return x_conditional_on_z_distribution


def x_marginal(z_distribution, x_conditional_on_z_distribution, x_support):
    x_distribution = dict.fromkeys(x_support, 0)
    for x in x_support:
        for (
            z,
            z_prob,
        ) in z_distribution.items():  # P[X=x] = \sum_{z} P[X=x | Z=z] * P[Z=z].
            x_distribution[x] += x_conditional_on_z_distribution[z][x] * z_prob
    assert math.isclose(
        sum(x_distribution.values()), 1, abs_tol=1e-16
    ), "Probabilities do not sum to 1."
    return x_distribution


def b_conditional_on_x_and_z(b_distribution, x_support, z_support, case="independent"):
    """Create conditional distribution of B given X = x and Z = z.

    Returns:
    The conditional distribution as a nested dict with 'outer' keys (x, z) and values as the conditioned B distribution.
    """
    b_conditional_on_x_z = {}
    for z in z_support:
        for x in x_support:
            # It may be sensible for B to depend on X, but probably not Z, at least not "directly".
            if case == "independent":
                b_conditional_on_x_z[
                    (x, z)
                ] = b_distribution  # Doesn't depend on X or Z.
                assert math.isclose(
                    sum(b_distribution.values()), 1, abs_tol=1e-16
                ), "Probabilities do not sum to 1."
    return b_conditional_on_x_z


def evaluate_cdf_of_u(b, x):
    return 1 / (1 + np.exp(-b[0] - b[1] * x))


def data_matrix(b_distribution, x_z_joint_support):
    r"""Computes (|x_support| * |z_support|) x |b_support| data matrix.

    Notes:
    -\theta is consistent with the data iff \gamma*\theta = g. This function computes \gamma (the data matrix)
    -\gamma is a known matrix (i.e., the entries are non-random).

    Returns:
    The data matrix (as a dictionary).
    """
    rows = {}
    for (
        x,
        z,
    ) in (
        x_z_joint_support
    ):  # The tuple (x, z) determines the row, while b determines the column.
        rows_elements = {}
        for b in b_distribution.keys():
            rows_elements[b] = evaluate_cdf_of_u(b, x)
        rows[(x, z)] = rows_elements
    return rows


def population_moments(
    x_z_joint_support, b_support, b_conditional_on_x_and_z_distribution
):
    moments = {}
    for x, z in x_z_joint_support:
        s = 0
        for b in b_support:
            s += b_conditional_on_x_and_z_distribution[(x, z)][b] * (
                evaluate_cdf_of_u(b, x)
            )
        moments[(x, z)] = s
    return moments
