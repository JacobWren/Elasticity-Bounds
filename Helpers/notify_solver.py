"""Users are responsible for notifying persistent solver interfaces when changes to a model are made! The functions
below are necessary for indexed constraints, but can handle simpler ones as well."""


def add_constraint(opt, constraint):
    if constraint.is_indexed():
        for my_con in constraint.values():
            opt.add_constraint(my_con)
    else:
        opt.add_constraint(constraint)


def remove_constraint(opt, constraint):
    if constraint.is_indexed():
        for my_con in constraint.values():
            opt.remove_constraint(my_con)
    else:
        opt.remove_constraint(constraint)
