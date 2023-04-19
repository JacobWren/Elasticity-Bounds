"""
Portion of model that doesn't depend on sampling.
"""

import pyomo.environ as pyo


def model_fixed(dgp_instance, norm):
    """Begin model creation; then pass it along."""
    m = pyo.ConcreteModel()  # Declare model

    # The assignments below are called 'index sets' (see documentation). But my use is non-standard. Rather than
    # creating parameters explicitly, I index the variable (theta) with the support of each potential parameter. That
    # is, my index is not a typical index (i.e. not 1, 2,... n). Setting up the computation of the bounds this way maps
    # to the notation of the problem.
    m.X_support = dgp_instance["X_support"]
    m.Z_support = dgp_instance["Z_support"]
    m.B_support = dgp_instance["B_support"]

    #  It is best to create an indexed component using a single set that is already on the model, otherwise implicit
    # sets will be created, which can cause trouble when deleting components of a model.
    m.X_Z_support = pyo.Set(initialize=dgp_instance["X_Z_support"])
    m.B_X_support = pyo.Set(initialize=dgp_instance["B_X_support"])
    m.B_Z_support = pyo.Set(initialize=dgp_instance["B_Z_support"])

    # Add \theta variable; Notice an
    # embedded constraint: all entries live in [0, 1].
    m.theta = pyo.Var(
        m.B_support,
        m.X_support,
        m.Z_support,
        domain=pyo.NonNegativeReals,
        bounds=(0, 1),
    )

    def valid_conditional_rule(model, x, z):
        r"""\theta sums to 1 across all (b_0, b_1) pairs for each x and z. The 'for each...' part is given in the
        constraint itself.

        Arguments:
        'model' is always the first argument passed in when using a function to get objective or constraint expressions.
        'x' lives in the first argument of pyo.Constraint() below and 'z' lives in the second.

        Returns:
        The constraint that \theta sums to 1 across all (b_0, b_1) pairs for a given x and z.
        """
        return (
            sum(model.theta[b[0], b[1], x, z] for b in dgp_instance["B_support"]) == 1
        )

    # Creates one constraint for each element (i.e. all combinations of the elements) of the respective supports.
    m.valid_conditional_constraint = pyo.Constraint(
        m.X_Z_support, rule=valid_conditional_rule
    )

    if norm == "1":
        # Add slack variables; non-negative with dimension |X|*|Z| x 1.
        m.slack_positive = pyo.Var(m.X_Z_support, domain=pyo.NonNegativeReals)

        m.slack_negative = pyo.Var(m.X_Z_support, domain=pyo.NonNegativeReals)
    elif norm == "max":
        m.slack = pyo.Var(domain=pyo.Reals)

    return m
