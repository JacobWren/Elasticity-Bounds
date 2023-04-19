"""
Simplifying functions.
"""


import numpy as np


def elasticity_helper(b_0, b_1, x_bar, e_bar):
    """Indicator function for cdf of elasticity across random B."""
    if b_1 * x_bar * (1 - (1 / (1 + np.exp(-b_0 - b_1 * x_bar)))) <= e_bar:
        return 1
    return 0


def constraint_nester(constraint, upper):
    # Allows Hard and Soft constraint cases to be more easily nested.
    # There doesn't appear to be a more 'natural' way to nest the constraints.
    if constraint == "Soft":
        if upper:
            return -1
        else:
            return 1
    else:
        return 0


def bound_type(upper):
    if upper:
        return "upper"
    return "lower"


def cumsum(sols, attribute):
    """Sum Scipy's optimization result object's attribute across the initial guesses."""
    return np.array([getattr(sol, attribute) for sol in sols]).sum()
