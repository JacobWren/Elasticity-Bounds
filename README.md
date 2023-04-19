All methods are implemented in Python 3.11, where a list of the dependencies can be found in `requirements.txt`.

From the command line, `cd` into `Computation`, and run, `mpiexec -n np python3 compute_bounds.py`, where np is the number of processes mpi will run.

A few of the relevant arguments to our method, `sim_bounds()`, include:
1. The criterion choice (**1**-norm or Euclidean norm)
2. The loss function (MSE loss or an asymmetric variant)
3. Method to optimize the tuning parameter (Nelder-Mead, gradient descent, or brute force)

You can obtain a license from [Gurobi's](https://www.gurobi.com/free-trial/) website to run the simulation.

*Please note, this project was not able to be completed.
