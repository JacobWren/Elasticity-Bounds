"""
Portion of model that depends on sampling (or is sensibly related to the portion that does), but not tuning.
"""

import pyomo.environ as pyo
import notify_solver as update_solver


def model_variable(
    m,
    opt,
    dgp_instance,
    moments,
    X_conditional_on_Z_distribution,
    Z_distribution,
    X_distribution,
    X_independent_of_B,
    norm,
):
    """
    Add to model and return updated model + data consistency rule.
    """

    def Q_hat_elements(
        model, x, z
    ):  # A Q has yet to be specified (this is the blood without the body).
        return (
            sum(
                (
                    dgp_instance["known_data_matrix"][(x, z)][b]
                    * model.theta[b[0], b[1], x, z]
                )
                for b in dgp_instance["B_support"]
            )
            - moments[(x, z)]
        )

    def slack_variables_1_norm(model, x, z):
        return model.slack_positive[x, z] - model.slack_negative[
            x, z
        ] == Q_hat_elements(model, x, z)

    def slack_lower_bound_max_norm(model, x, z):
        return -model.slack <= Q_hat_elements(model, x, z)

    def slack_upper_bound_max_norm(model, x, z):
        return Q_hat_elements(model, x, z) <= model.slack

    if norm == "1":
        m.slack_variables_1_norm_constraint = pyo.Constraint(
            m.X_Z_support, rule=slack_variables_1_norm
        )
        update_solver.add_constraint(opt, m.slack_variables_1_norm_constraint)

    elif norm == "max":
        m.slack_lower_bound_max_norm_constraint = pyo.Constraint(
            m.X_Z_support, rule=slack_lower_bound_max_norm
        )
        update_solver.add_constraint(opt, m.slack_lower_bound_max_norm_constraint)

        m.slack_upper_bound_max_norm_constraint = pyo.Constraint(
            m.X_Z_support, rule=slack_upper_bound_max_norm
        )
        update_solver.add_constraint(opt, m.slack_upper_bound_max_norm_constraint)

    def B_independent_of_Z_rule(model, b_0, b_1, z):
        r"""Impose the constraint that B is independent of Z, by appropriately constraining \theta."""
        return (
            sum(
                (model.theta[b_0, b_1, x, z] * X_conditional_on_Z_distribution[z][x])
                - (
                    model.theta[b_0, b_1, x, model.Z_support[0]]
                    * X_conditional_on_Z_distribution[model.Z_support[0]][x]
                )
                for x in model.X_support
            )
            == 0
        )

    m.B_independent_of_Z_constraint = pyo.Constraint(
        m.B_Z_support, rule=B_independent_of_Z_rule
    )
    update_solver.add_constraint(opt, m.B_independent_of_Z_constraint)

    def X_independent_of_B_rule(model, b_0, b_1, x):
        """Impose independence between X and B."""
        return (
            sum(
                (
                    model.theta[b_0, b_1, x, z]
                    * X_conditional_on_Z_distribution[z][x]
                    * Z_distribution[z]
                    / X_distribution[x]
                )
                - (
                    model.theta[b_0, b_1, model.X_support[0], z]
                    * X_conditional_on_Z_distribution[z][model.X_support[0]]
                    * Z_distribution[z]
                    / X_distribution[model.X_support[0]]
                )
                for z in model.Z_support
            )
            == 0
        )

    if X_independent_of_B:
        m.X_independent_of_B_constraint = pyo.Constraint(
            m.B_X_support, rule=X_independent_of_B_rule
        )
        update_solver.add_constraint(opt, m.X_independent_of_B_constraint)

    def minQ_1_norm_obj(model):
        return sum(
            model.slack_positive[x, z] + model.slack_negative[x, z]
            for x, z in model.X_Z_support
        )

    def minQ_max_norm_obj(model):
        return model.slack

    def minQ_euclidean_norm_obj(model):
        return sum(Q_hat_elements(model, x, z) ** 2 for x, z in model.X_Z_support)

    if norm == "1":
        rule = minQ_1_norm_obj
    elif norm == "max":
        rule = minQ_max_norm_obj
    else:
        rule = minQ_euclidean_norm_obj

    return rule
