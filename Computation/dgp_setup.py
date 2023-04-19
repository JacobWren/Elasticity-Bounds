"""No draws are made in this module. Rather, this module picks the dgp (i.e., the "base" for which estimation is
conducted upon)."""

import dgp as dgp
import pytest


def dgp_constructor(
    x_z_dependence_controller=0,
    b0_size=10,
    b0_element_types="floats",
    b0_scalar=0.5,
    b1_size=10,
    b1_element_types="floats",
    b1_scalar=-3,
    x_size=4,
    x_element_types="integers",
    z_size=2,
    z_element_types="integers",
):
    # (Joint) Distribution for B = (B_0, B_1).
    b_support, b_distribution = dgp.joint_distribution_from_2_rvs(
        b0_scalar,
        b0_size,
        b0_element_types,
        b1_scalar,
        b1_size,
        b1_element_types,
        # case=["B0 non-uniform", "B1 non-uniform"]
    )

    assert len(b_support) == len(b_distribution) == b0_size * b1_size

    z_support, z_distribution = dgp.marginal_distribution(
        z_size, z_element_types  # Set case == "Z non-uniform" as an alternative.
    )  # Create distribution for Z.

    assert len(z_support) == len(z_distribution) == z_size

    # Regardless of whether X is independent of Z (i.e., x_z_dependence_controller = 0 or not), I specify the
    # distribution of X as if they were independent, which is used as a sort of "base" for the "updated" X distribution,
    # if in fact, X and Z are dependent. This is done to nest the independent/dependent cases (i.e., for convenience).
    (
        x_support,
        x_independent_z_distribution,
    ) = dgp.marginal_distribution(
        x_size,
        x_element_types,
    )  # Set case == "X non-uniform" as an alternative.

    assert len(x_support) == len(x_independent_z_distribution) == x_size

    x_z_joint_support = dgp.two_var_joint_support(x_support, z_support)
    assert len(x_z_joint_support) == x_size * z_size
    b_x_joint_support = dgp.two_var_joint_support(b_distribution, x_support, lag=True)
    assert len(b_x_joint_support) == b0_size * b1_size * (x_size - 1)
    b_z_joint_support = dgp.two_var_joint_support(b_distribution, z_support, lag=True)
    assert len(b_z_joint_support) == b0_size * b1_size * (z_size - 1)

    # Conditional distribution of X given Z = z. If x_z_dependence_controller = 0, then x_conditional_on_z_distribution
    # will just be x_independent_z_distribution for each z in Z.
    x_conditional_on_z_distribution = dgp.x_conditional_on_z(
        x_independent_z_distribution, x_support, z_support, x_z_dependence_controller
    )
    assert len(x_conditional_on_z_distribution) == z_size
    assert (
        pytest.approx(x_conditional_on_z_distribution[z_support[0]], abs=1e-12)
        == x_independent_z_distribution  # This is specific to the way I "baked" the DGP.
    )
    for z in z_support:
        assert len(x_conditional_on_z_distribution[z]) == x_size

    if x_z_dependence_controller == 0:
        x_distribution = x_independent_z_distribution
    else:  # "Update"
        x_distribution = dgp.x_marginal(
            z_distribution, x_conditional_on_z_distribution, x_support
        )
    assert len(x_distribution) == x_size

    if x_z_dependence_controller == 0:  # X and Z are independent.
        for z in z_support:
            assert (
                pytest.approx(x_conditional_on_z_distribution[z], abs=1e-12)
                == x_distribution
            )

    # Conditional distribution of B given X = x and Z = z.
    b_conditional_on_x_and_z_distribution = dgp.b_conditional_on_x_and_z(
        b_distribution, x_support, z_support
    )
    assert len(b_conditional_on_x_and_z_distribution) == x_size * z_size
    for z in z_support:
        for x in x_support:
            assert (
                len(b_conditional_on_x_and_z_distribution[(x, z)]) == b0_size * b1_size
            )

    known_data_matrix = dgp.data_matrix(b_distribution, x_z_joint_support)
    assert len(known_data_matrix) == x_size * z_size

    population_moments = dgp.population_moments(
        x_z_joint_support, b_support, b_conditional_on_x_and_z_distribution
    )
    assert len(population_moments) == x_size * z_size

    return {
        "B_support": b_support,
        "B_distribution": b_distribution,
        "X_support": x_support,
        "X_distribution": x_distribution,
        "Z_support": z_support,
        "Z_distribution": z_distribution,
        "X_Z_support": x_z_joint_support,
        "B_X_support": b_x_joint_support,
        "B_Z_support": b_z_joint_support,
        "X_conditional_on_Z_distribution": x_conditional_on_z_distribution,
        "B_conditional_on_X_and_Z_distribution": b_conditional_on_x_and_z_distribution,
        "known_data_matrix": known_data_matrix,
        "population_moments": population_moments,
    }
