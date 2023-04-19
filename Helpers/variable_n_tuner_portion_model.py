"""
Portion of model that depends on tuner (or is sensibly related to the portion that does) and sampling.
"""

import pyomo.environ as pyo

import notify_solver as update_solver
import general_helpers as helpers


def model_tuner_variable(
    m,
    opt,
    tune,
    constraint,
    rule,
    Q_star_out,
    X_conditional_on_Z_distribution,
    Z_distribution,
    dgp_instance,
    upper,
    x_bar,
    e_bar,
    tuner_tilde,
):
    """
    Complete model and return objective expressions.
    """

    def hard_constraint_upper_bound():
        if (constraint == "Hard_add") or (tuner_tilde == "inf"):
            return Q_star_out + tune  # epsilon enters in additively
        elif constraint == "Hard_mult":
            return Q_star_out * (1 + tune)  # epsilon enters in multiplicatively

    def hard_constraint(model):
        if (tuner_tilde == "inf") or (tuner_tilde == "0"):
            return rule(model) == hard_constraint_upper_bound()
        return rule(model) <= hard_constraint_upper_bound()

    if constraint.startswith("Hard") or (tuner_tilde == "inf"):
        m.add_component(
            "any_norm_hard_constraint", pyo.Constraint(rule=hard_constraint)
        )
        update_solver.add_constraint(opt, m.any_norm_hard_constraint)

    def target(model, evaluate):
        return sum(
            helpers.elasticity_helper(b[0], b[1], x_bar, e_bar)
            * sum(
                pyo.value(model.theta[b[0], b[1], x, z])
                * X_conditional_on_Z_distribution[z][x]
                * Z_distribution[z]
                if evaluate
                else model.theta[b[0], b[1], x, z]
                * X_conditional_on_Z_distribution[z][x]
                * Z_distribution[z]
                for z in model.Z_support
                for x in model.X_support
            )
            for b in dgp_instance["B_support"]
        )

    def obj_expression(model):
        return target(model, False) + helpers.constraint_nester(
            constraint, upper
        ) * tune * (rule(model) - Q_star_out)

    return obj_expression, target
