"""Compute point estimate."""
import numpy as np

import variable_n_portion_model as vnpm
import variable_n_tuner_portion_model as vntpm
import define_solve as ds
import clean_free_model as cfm
import setup_solve as setup
import error as err
import fixed_portion_model as fpm
import control_display as display


def optimal_bounds(
    data,
    dgp_instance,
    X_independent_of_B,
    norm,
    constraint,
    rank,
    upper,
    x_bar,
    e_bar,
    kappa_tilde,
    spopt_results,
    Q_star,
    loss_type,
    method,
):
    model = fpm.model_fixed(dgp_instance, norm)
    opt = setup.solver_options(norm)
    # Attempt to set up the solve.
    setup.instantiate_solve(opt, model, rank)

    (
        moments,
        X_distribution,
        Z_distribution,
        X_conditional_on_Z_distribution,
    ) = data

    minQ_rule = vnpm.model_variable(
        model,
        opt,
        dgp_instance,
        moments,
        X_conditional_on_Z_distribution,
        Z_distribution,
        X_distribution,
        X_independent_of_B,
        norm,
    )

    optimal_tuner, min_error = err.optimum(spopt_results, method)

    # For optimal solves with original sample after S bootstraps.
    obj_expression, target = vntpm.model_tuner_variable(
        model,
        opt,
        optimal_tuner,
        constraint,
        minQ_rule,
        Q_star,
        X_conditional_on_Z_distribution,
        Z_distribution,
        dgp_instance,
        upper,
        x_bar,
        e_bar,
        kappa_tilde,
    )
    # Compute bound.
    bound = ds.find_bound(
        model,
        opt,
        obj_expression,
        target,
        constraint,
        upper,
        kappa_tilde,
    )

    display.silence()  # Quiet Gurobi
    # Release token.
    cfm.free_token(opt)
    display.speak()  # Allow output

    if loss_type != "mae":
        min_error = np.power(min_error, 1 / 2)  # Root loss

    return (
        optimal_tuner,
        bound,
        min_error,
        Q_star,  # Returned for interest.
    )
