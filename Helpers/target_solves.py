"""Center loss function."""
import variable_n_portion_model as vnpm
import variable_n_tuner_portion_model as vntpm
import define_solve as ds
import clean_free_model as cfm
import setup_solve as setup
import fixed_portion_model as fpm
import control_display as display


def get_target_bounds(
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
    Qstar_flag=True,  # Set to True if constant, c, in \tilde{\kappa}, involves Qstar
    # if \tilde{\kappa} = \infty, or for plotting.
):
    # My call to model_fixed() is only necessary to do once (period). But a model is difficult to pickle -- given that
    # the model creation step is fast, the cost doesn't outweigh the benefit.
    model = fpm.model_fixed(
        dgp_instance, norm
    )  # Begin "baseline" model creation (variable supports, etc.).
    opt = setup.solver_options(norm)  # Set solver options (solver threads, etc.)
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

    # If the constraint is soft, solve minQ with the original sample (iff Qstar_flag is True) regardless of
    # \tilde{\kappa} (if \tilde{\kappa}=+\infty, minQ would be necessary to solve.); I solve minQ by default since its
    # value is of interest.
    # Captures both Hard_add and Hard_mult.
    if constraint.startswith("Hard") or Qstar_flag:
        Q_star = ds.solve_minQ(model, opt, minQ_rule)
        # The next solve will be with a different objective, (hence the del below, but only if solve_minQ() was invoked
        # -- otherwise there is no model.obj... as in the else below), one more constraint (hence add below), but keep
        # all the prior constraints (hence no 'deling' of constraints below).
        cfm.clear_obj(model)
    else:
        Q_star = None  # Placeholder for nesting constraints.

    # Case seven is identified by "kappa_tilde == 'inf'" for the soft constraint and "kappa_tilde == '0'" for the
    # hard constraint. While, seven can be nested w/ both constraints, I only nest it with the hard constraint
    # (hence the tuner=0), since infinity can be computationally unstable to work with (but yes, sometimes
    # possible).
    # Get tuner for target bound computation.
    if (kappa_tilde == "inf") or (kappa_tilde == "0"):
        tuner = 0
    else:  # Finite \tilde{\kappa} for soft constraint.
        tuner = kappa_tilde

    obj_expression, target = vntpm.model_tuner_variable(
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
    # Compute bound :)
    target_bound = ds.find_bound(
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

    return (
        target_bound,
        Q_star,
    )  # Return Qstar only for interest (as opposed to use).
