"""
Main script :)
"""
import re
import sys

import numpy as np
import pandas as pd
import scipy.optimize as spopt

from joblib import Parallel, delayed, parallel_backend
from mpi4py import MPI

if sys.platform == "darwin":  # OS X
    sys.path.insert(1, "/Users/jakewren/PycharmProjects/Elasticity/Helpers")

import dgp_setup as dgp_setup
import estimate as estimation
import general_helpers as helpers
import evaluate_engine as engine
import opt_solves as opt_solve
import target_solves as target_solve
import kappa_points as kappas

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()


def compute_bounds(
    S,
    norm,
    upper,
    loss_type,
    constraint,
    X_independent_of_B_flag,
    x_bar,
    e_bar,
    sample_size,
    dgp_instance,
    population_flag,
    rng,
    backend,
    method,
    n_workers,
):
    def draw_original():
        # On each Monte Carlo replication I draw w and estimate the necessary features.
        (
            w_moments,
            w_X_distribution,
            w_Z_distribution,
            w_X_conditional_on_Z_distribution,
            w_bootstrap_distribution,
        ) = estimation.vector_of_moments_estimator(
            dgp_instance["X_distribution"],
            dgp_instance["Z_distribution"],
            dgp_instance["X_Z_support"],
            sample_size,
            dgp_instance["B_conditional_on_X_and_Z_distribution"],
            dgp_instance["X_conditional_on_Z_distribution"],
            rng,
        )

        return (
            w_bootstrap_distribution,
            w_moments,
            w_X_distribution,
            w_Z_distribution,
            w_X_conditional_on_Z_distribution,
        )

    # Do we have all the data? You have to know if you're close to the truth.
    if population_flag:
        data_w = (
            dgp_instance["population_moments"],
            dgp_instance["X_distribution"],
            dgp_instance["Z_distribution"],
            dgp_instance["X_conditional_on_Z_distribution"],
        )
    else:
        distribution_w, *data_w = draw_original()
        num_bootstraps = S
        # Draw w_s for all s=1,...,S
        boot_strapped_data = estimation.bootstrap_vector_of_moments_estimator(
            distribution_w,
            dgp_instance["Z_distribution"],
            dgp_instance["X_distribution"],
            dgp_instance["X_Z_support"],
            sample_size,
            num_bootstraps,
            rng,
        )

    if norm == "1":
        # The 1-norm criterion should be converging to 0 at the sqrt(n) rate.
        # So log(n) is slower, and thus justified, although not the only possible choice.
        target_tuner = np.log(sample_size)
    else:
        target_tuner = sample_size / np.log(sample_size)

    target_bound, Q_star = target_solve.get_target_bounds(
        data_w,
        dgp_instance,
        X_independent_of_B_flag,
        norm,
        constraint,
        rank,
        upper,
        x_bar,
        e_bar,
        target_tuner,
    )

    # A single bound.
    if population_flag:
        print(
            "The population",
            helpers.bound_type(upper),
            "bound is",
            round(target_bound, 6),
        )
        exit()

    if target_tuner == "inf":
        target_tuner = None  # Only solve "seven" for target bound (this is usually a bad choice, however).

    if method == "brute force":
        approach = spopt.brute
        x0s = kappas.grid_points(
            norm
        )  # Non-adaptive search; this is slow, but serves as a comparison for the adaptive
        # approaches.
        solver_options = {
            "finish": spopt.fmin,  # "finish" adds a touch of "last-minute" adaptation; I manually ensure
            # non-negative kappas.
            "full_output": True,
            "workers": n_workers,
        }
        n_jobs = 1  # Recall, only one level of parallelization!
    else:
        approach = spopt.minimize
        x0s = kappas.starting_points(
            target_tuner, norm, sample_size, rng
        )  # Quasi random guesses -- good heuristics matter :)

        solver_options = {"method": method, "bounds": ((0, None),)}  # kappa >= 0

        if method == "trust-constr":  # GD based.
            solver_options["hess"] = spopt.BFGS()

        n_jobs = n_workers

    # Find the kappa (denoted \hat{\kappa}) that minimizes the loss function!
    sols = []
    with parallel_backend(backend):
        (
            sols.append(
                Parallel(n_jobs=n_jobs)(
                    delayed(approach)(
                        engine.boot_strap_evaluations,
                        x0,
                        args=(
                            boot_strapped_data,
                            num_bootstraps,
                            dgp_instance,
                            X_independent_of_B_flag,
                            norm,
                            constraint,
                            rank,
                            upper,
                            loss_type,
                            x_bar,
                            e_bar,
                            X_independent_of_B_flag,
                            target_tuner,
                            target_bound,
                            sample_size,
                            Q_star,
                        ),
                        **solver_options,  # Unpack kwargs.
                    )
                    for x0 in x0s
                )
            )
        )

    sols = sols[0]
    if method != "brute force":
        n_iterations = helpers.cumsum(
            sols, "nit"
        )  # Total number of iterations across all initial guesses.
        n_obj_evals = helpers.cumsum(
            sols, "nfev"
        )  # Total number of evaluations of the objective function...

        if helpers.cumsum(sols, "success") != len(x0s):
            print(
                "Your gradient is probably zero -- at least up to the solver's tolerance."
            )
            comm.Abort()

        solver_info = (n_iterations, n_obj_evals)
        print(
            "solver_info:", solver_info
        )  # This way, I can track the job on the cluster.

        best_sol = np.argmin([sol.fun for sol in sols])
        sol = sols[best_sol]  # Ding, ding, winner!
    else:
        sol = sols

    opt_res = opt_solve.optimal_bounds(
        data_w,
        dgp_instance,
        X_independent_of_B_flag,
        norm,
        constraint,
        rank,
        upper,
        x_bar,
        e_bar,
        target_tuner,
        sol,
        Q_star,
        loss_type,
        method,
    )

    if method != "brute force":
        results = opt_res + solver_info
    else:
        results = opt_res

    return results


def compute_bounds_manager(
    reps,
    S,
    n_workers,
    backend,
    norm,
    upper,
    loss_type,
    constraint,
    X_independent_of_B_flag,
    x_bar,
    e_bar,
    sample_size,
    dgp_instance,
    population_flag,
    method,
):
    if not population_flag:  # Only sample if you don't know the population.
        if rank == 0:
            if reps < nprocs:
                print("Error: the number of processes > the number of repetitions.")
                comm.Abort()

            # Create the RNG that you want to pass around.
            rng = np.random.default_rng(98765)  # Seed it for reproducibility.
            # Get the SeedSequence of the passed RNG.
            ss = rng.bit_generator._seed_seq
            # Create reps (some number of) initial independent states
            child_states = ss.spawn(reps)
            # 'Chunkify' (i.e., split into sub-tasks) the child_states vector before scattering across the processes.
            child_state_sub_tasks = np.array_split(child_states, nprocs)
        else:
            # Each of the nprocs processes must have a value for 'child_state_sub_tasks'.
            child_state_sub_tasks = None

        # Scatter the child_states vector across all processes.
        # Remember you cannot use Scatterv() for non-numeric dtypes involving Python objects. Passing around things like
        # SeedSequence with pickle should not be expensive, so lowercase scatter() is just fine.
        sub_task = comm.scatter(child_state_sub_tasks, root=0)

    else:
        sub_task = np.array([None])

    # Only one level of parallelization, since Gurobi is not thread safe. Either parallelize the repetitions or the
    # search for \hat{\kappa}.
    if reps > 4000:
        n_jobs = n_workers
    else:
        n_jobs = 1  # i.e., no parallelization.

    # If n_reps > n_processes, the delta (i.e., n_reps - n_processes) needs to be looped over, in parallel or not.
    output_per_mpi_process = []
    with parallel_backend(backend):
        (
            output_per_mpi_process.append(
                Parallel(n_jobs=n_jobs)(
                    delayed(compute_bounds)(
                        S=S,
                        norm=norm,
                        upper=upper,
                        loss_type=loss_type,
                        constraint=constraint,
                        X_independent_of_B_flag=X_independent_of_B_flag,
                        x_bar=x_bar,
                        e_bar=e_bar,
                        sample_size=sample_size,
                        dgp_instance=dgp_instance,
                        population_flag=population_flag,
                        rng=random_state,
                        backend=backend,
                        method=method,
                        n_workers=n_workers,
                    )
                    for random_state in sub_task
                )
            )
        )
    #  Collect a list of numpy arrays from each rank and get a list of list (hence the '[0]') of numpy arrays on the
    #  master process.
    all_output = comm.gather(
        output_per_mpi_process[0], root=0
    )  # Combine results into single process (-> rank zero)
    if rank == 0:
        return all_output, (reps, constraint, norm, sample_size)
    else:
        if all_output is not None:
            comm.Abort()


def generate_data(
    constraint,
    norm,
    upper,
    loss_type,
    reps,
    S,
    sample_size,
    dgp_instance,
    x_bar,
    e_bar,
    X_independent_of_B_flag,
    population_flag,
    backend,
    n_workers,
    method,
):
    output = compute_bounds_manager(
        reps=reps,
        S=S,
        n_workers=n_workers,
        backend=backend,
        norm=norm,
        upper=upper,
        loss_type=loss_type,
        constraint=constraint,
        X_independent_of_B_flag=X_independent_of_B_flag,
        x_bar=x_bar,
        e_bar=e_bar,
        sample_size=sample_size,
        dgp_instance=dgp_instance,
        population_flag=population_flag,
        method=method,
    )

    if rank == 0:
        (results, notes) = output
        opt_res = []  # Unpack and spit out results.
        for (
            gathered_reps
        ) in (
            results
        ):  # gathered_reps will have size reps/cores (although, this may not be true for all).
            for single_rep in gathered_reps:
                opt_res.append(single_rep)

        columns = [
            "kappa",
            "bound",
            "loss",
            "Qstar",
        ]

        if method != "brute force":
            columns += [
                "n_iterations",
                "n_func_evals",
            ]

        df = pd.DataFrame(
            opt_res,
            columns=columns,
        )  # List of row tuples.

        df.to_csv(  # Download (and name) results as CSV file.
            notes[1]
            + "_"
            + notes[2]
            + "_"
            + str(notes[0])
            + "_"
            + str(notes[3])
            + "_"
            + loss_type
            + "_"
            + helpers.bound_type(upper)
            + "_"
            + re.sub(r"\s|-", "_", method),
            index=False,
        )


def debugger(port_mapping):
    # Call this to debug an MPI program.
    import pydevd_pycharm

    pydevd_pycharm.settrace(
        "localhost",
        port=port_mapping[rank],
        stdoutToServer=True,
        stderrToServer=True,
    )


# debugger([64734])  # Start debugging :)

# Specified some options for convenience #
computational_methods = {0: "brute force", 1: "Nelder-Mead", 2: "trust-constr"}
norms = {0: "Euclidean", 1: "1"}
loss = {0: "rmse", 1: "rrtse", 2: "mae"}

generate_data(
    constraint="Soft",
    norm=norms[1],
    upper=False,
    loss_type=loss[0],
    reps=500,
    S=500,  # Number of draws from F.
    sample_size=8000,
    dgp_instance=dgp_setup.dgp_constructor(),  # Default arguments are passed in the "dgp_setup" module.
    x_bar=1,
    e_bar=-1,
    X_independent_of_B_flag=True,
    population_flag=False,
    backend="multiprocessing",
    n_workers=4,
    method=computational_methods[1],
)
