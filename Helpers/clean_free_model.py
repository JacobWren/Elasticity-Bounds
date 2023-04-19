"""
Clean model or free server license token (4,096 is the max limit for Gurobi floating licenses).
"""

import gurobipy as gp
import notify_solver as update_solver


def clear_obj(m):
    m.del_component(m.obj)
    # set_objective() will discard the existing objective (on the gurobipy Model object).


def clean_constraint(m, opt_s):
    update_solver.remove_constraint(opt_s, m.any_norm_hard_constraint)
    m.del_component(
        m.any_norm_hard_constraint
    )  # For the soft case, the constraint is embedded in the objective, which is tossed in either case.


def free_token(opt_s):
    # Clean model between sample iterations
    opt_s._solver_model.dispose()
    gp.disposeDefaultEnv()


def clean_across_samples(m, norm, X_independent_of_B_flag, opt):
    """Remove constraints that differ across bootstrap evaluations, before adding new ones."""
    if norm == "1":
        update_solver.remove_constraint(opt, m.slack_variables_1_norm_constraint)
        m.del_component(m.slack_variables_1_norm_constraint)
    elif norm == "max":
        update_solver.remove_constraint(opt, m.slack_lower_bound_max_norm_constraint)
        m.del_component(m.slack_lower_bound_max_norm_constraint)
        update_solver.remove_constraint(opt, m.slack_upper_bound_max_norm_constraint)
        m.del_component(m.slack_upper_bound_max_norm_constraint)

    update_solver.remove_constraint(opt, m.B_independent_of_Z_constraint)
    m.del_component(m.B_independent_of_Z_constraint)

    if X_independent_of_B_flag:
        update_solver.remove_constraint(opt, m.X_independent_of_B_constraint)
        m.del_component(m.X_independent_of_B_constraint)
