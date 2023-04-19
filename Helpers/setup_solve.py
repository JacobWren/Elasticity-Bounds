"""
Pass in Gurobi solving parameters and choose Gurobi Algorithm.
"""
import pyomo.environ as pyo
import time
import control_display as display


def instantiate_solve(opt, model, rank):
    while True:
        try:
            display.silence()  # Quiet Gurobi
            # Tell the solver about our model (i.e., create a gurobipy Model object); this requires an available token.
            opt.set_instance(model)
            display.speak()  # Allow output
            break
        except (
            Exception
        ):  # This is a gurobipy.GurobiError Error, but for a reason I can not understand, the
            # cluster throws the error anyway...
            # Hit Gurobi token server limit (4096).
            print("No tokens available.")
            # Vary wait times across compute nodes.
            time.sleep(min(60 + rank * 12, 500))


def solver_options(norm):
    # Persistent solver interfaces are best suited for incremental changes to a Pyomo model.
    opt = pyo.SolverFactory("gurobi_persistent")

    # if norm != "Euclidean":
    #     gurobi_algorithm = 3  # concurrent
    #     opt.options[
    #         "Threads"
    #     ] = 3  # Setting Threads > 1 may kill determinism, which is helpful for testing/making comparisons.
    # else:
    #     gurobi_algorithm = 2  # barrier
    #     opt.options[
    #         "Threads"
    #     ] = 4

    opt.options[
        "Method"
    ] = 3  # This is not strictly necessary; but certain algorithms are deterministic, which makes
    # # testing easier.

    opt.options["Threads"] = 2

    # I left in some alternative gurobi parameters that can be helpful for solving challenging programs.
    # opt.options["NonConvex"] = 2
    # opt_s.options['SolCount'] = 2
    # opt.options["NumericFocus"] = 1
    # opt_s.options['BarHomogeneous'] = 1
    # opt.options['Quad'] = 1
    # opt_s.options['MarkowitzTol'] = .5
    return opt
