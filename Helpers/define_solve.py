"""
Define solves.
"""

import pyomo.environ as pyo


def solve_minQ(model, opt_s, rule):
    model.obj = pyo.Objective(rule=rule, sense=pyo.minimize)
    opt_s.set_objective(model.obj)
    opt_s.solve(model, save_results=False)
    return model.obj.expr()


def find_bound(
    model,
    opt_s,
    obj_expression,
    target,
    constraint,
    upper,
    tuner_tilde,
):
    if upper:
        model.obj = pyo.Objective(rule=obj_expression, sense=pyo.maximize)
    else:
        model.obj = pyo.Objective(rule=obj_expression, sense=pyo.minimize)
    opt_s.set_objective(model.obj)

    # Solve
    opt_s.solve(
        model, save_results=False
    )  # Pass in 'load_solutions=True'/'tee=True' for model/solve information.

    # model.solutions.store_to(results)
    # results.write()
    # model.slack.display()

    if constraint.startswith("Hard") or (tuner_tilde == "inf"):
        return pyo.value(model.obj)
    elif constraint == "Soft":
        return target(model, True)
