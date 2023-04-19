"""Helper functions for DGP creation."""


def list_converter(lists):
    """Takes in list of list-like objects and returns a nested list."""
    converted = []
    for lt in lists:
        if not isinstance(lt, list):
            lt = list(lt)
        converted.append(lt)
    return converted


def dict_keys_to_list(distribution):
    if isinstance(distribution, dict):
        return list(distribution.keys())
    return distribution  # Otherwise


def isolate_zero(val):
    """Returns 1 unless val is zero."""
    val = 1 if val else 0
    return val


def support_dimension(distribution):
    distribution_key = next(iter(distribution))  # Any key will do, not just the first.
    if isinstance(distribution_key, int):
        return 1
    return len(distribution_key)  # Will be of type tuple.


def zip_helper(sample):
    sample_dimension = len(sample[0])  # All elements will have the same dimension.
    for count in range(sample_dimension):
        yield list(list(zip(*sample))[count])  # Create a generator.
