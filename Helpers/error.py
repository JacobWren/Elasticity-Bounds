"""
Loss/error helper functions.
"""
import numpy as np


def optimum(spopt_results, method):
    if method != "brute force":
        k_hat = spopt_results.x[0]
        min_loss = spopt_results.fun
        # print("Description of the cause of the termination:", spopt_results.message)
    else:
        spopt_results = spopt_results[0]
        k_hat = spopt_results[0][0]
        min_loss = spopt_results[1]
    return k_hat, min_loss


def loss(loss_type, bounds, bound_reference, S, sample_size, Q_star):
    preds = np.array(bounds)
    err = preds - bound_reference

    if loss_type == "rmse":  # squared error
        return np.sum(np.square(err)) / S
    elif loss_type == "rrtse":  # right-tilted squared error
        alpha = min(
            (np.log(sample_size) / np.power(sample_size, 1 / 2)) + ((Q_star + 1) / 2), 1
        )
        return (
            np.sum(
                np.where(err >= 0, alpha * np.square(err), (1 - alpha) * np.square(err))
            )
            / S
        )

    elif loss_type == "mae":  # mean absolute error
        return np.sum(np.abs(err)) / S
