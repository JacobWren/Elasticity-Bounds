"""Bootstrap evaluations."""

import variable_n_portion_model as vnpm
import variable_n_tuner_portion_model as vntpm
import define_solve as ds
import clean_free_model as cfm
import setup_solve as setup
import error as err
import fixed_portion_model as fpm


def boot_strap_evaluations(
    tuner,
    boot_strapped_data,
    S,
    dgp_instance,
    X_independent_of_B,
    norm,
    constraint,
    rank,
    upper,
    loss_type,
    x_bar,
    e_bar,
    X_independent_of_B_flag,
    kappa_tilde,
    bound_reference,
    sample_size,
    Q_star,
):
    if (
        tuner <= 0
    ):  # This is an edge-case for "brute force". The polishing function, fmin(), doesn't take in bounds -- so
        # set kappa to zero. Then, the bounds are deterministic.
        # Technically the edge-case would only check for tuner < 0, but if tuner is zero, there is no reason to compute
        # the associated bound.
        bounds = [0]*S
        est_loss = err.loss(
            loss_type, bounds, bound_reference, S, sample_size, Q_star
        )  # Compute loss!
    else:
        model = fpm.model_fixed(
            dgp_instance, norm
        )  # "Baseline" model creation (variable supports, etc.); not data dependent.
        opt = setup.solver_options(norm)  # Set solver options (solver threads, etc.)
        # Attempt to set up the solve.
        setup.instantiate_solve(opt, model, rank)

        bounds = (
            []
        )  # Save bounds across bootstrap evaluations to estimate loss as a function of \kappa.
        for s in range(S):
            (
                moments,
                X_distribution,
                Z_distribution,
                X_conditional_on_Z_distribution,
            ) = boot_strapped_data[s]

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

            if constraint.startswith("Hard"):  # Captures both Hard_add and Hard_mult
                Q_star = ds.solve_minQ(model, opt, minQ_rule)
                # The next solve will be with a different objective, (hence the del below, but only if solve_minQ() was
                # invoked -- otherwise there is no model.obj... as in the else below), one more constraint (hence add
                # below), but keep all the prior constraints (hence no 'deling' of constraints below).
                cfm.clear_obj(model)
            else:
                Q_star = 0  # Default (any number will work).

            (
                obj_expression,
                target,
            ) = vntpm.model_tuner_variable(
                model,
                opt,
                tuner,
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

            bounds.append(bound)

            cfm.clear_obj(model)

            # Clean model between sample iterations.
            cfm.clean_across_samples(model, norm, X_independent_of_B_flag, opt)

            # Toss part of the model that is associated with a particular value of epsilon or kappa.
            # For the soft case, the constraint (w/ kappa) is embedded in the objective, which is already disposed of.
            if constraint.startswith("Hard"):
                cfm.clean_constraint(model, opt)

        # If you hit the Gurobi server token limit, then you should occasionally release a token.
        # if (s % 4 == 0) or (  # '4' and S determine frequency of token release.
        #     s == S - 1
        # ):
            # display.silence()  # Quiet Gurobi
            # # Release token.
            # cfm.free_token(opt)
            # display.speak()  # Allow output

        est_loss = err.loss(
            loss_type, bounds, bound_reference, S, sample_size, Q_star
        )  # Compute loss!

    return est_loss
