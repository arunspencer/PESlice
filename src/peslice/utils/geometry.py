# ruff: noqa: F821, F722


import torch
import numpy as np
from jaxtyping import Float


def check_orthogonal(*args, function_dim) -> bool:
    """
    Checks whether all parsed vectors are orthogonal

    Vectors parsed to this function must be contain sub-vectors of length 3 (x, y, z)

    The respective sub-vectors between each vector are checked for orthogonality

    Parameters
    ----------
    *args
        n vectors to check

        Shape `(1, function_dim)`

    Returns
    -------
    Whether the vectors are all othogonal

    Raises
    ------
    ValueError
        If all of the parsed vectors are not the same shape
    """
    for vec in args:
        if len(vec) != function_dim:
            raise ValueError(
                "The length of the vector must match the number of dimensions in the function"
            )

    if len(args) == 1:
        return True

    A = [np.zeros((len(args), 3)) for _ in range(int(function_dim / 3))]

    for i, vec in enumerate(args):
        for j, row in enumerate(vec.reshape(-1, 3)):
            A[i][j, :] = row

    for a in A:
        dot_product_matrix = np.dot(a, a.T)

        if not np.allclose(
            dot_product_matrix, np.diag(np.diagonal(dot_product_matrix))
        ):
            return False

    return True


def gen_random_slice(
    m0: Float[np.ndarray, "function_dim "],
    slice_dim: int,
    seed: int,
) -> Float[np.ndarray, "slice_dim function_dim"]:
    """
    Generates a random slice in `slice_dim` dimensions.

    The overall dimension is the length of the `m0` vector

    Parameters
    ----------
    m0
        A point on the hyperplane

        Shape `(function_dim, )`
    slice_dim
        The dimension of the hyperplane
    seed
        The seed for the random generation of slices

    Raises
    ------
    ValueError
        If the slice dimension is greater than the function dimension or is < 1

        If slice_dim is outside of the range [1, 3]
    """

    rng = np.random.default_rng(seed)

    function_dim = len(m0)

    # checking that the slice dimension is in the range [1, function_dim]
    if slice_dim > 3 or slice_dim < 1:
        raise ValueError("Slice dimension must be in range [1, 3]")

    # generating random basis vectors, ensuring they are orthogonal
    orthogonal_basis = np.zeros((slice_dim, function_dim))

    m1 = rng.uniform(-1, 1, function_dim).reshape(-1, 3)
    m1 /= np.linalg.norm(m1, axis=1, keepdims=True)

    orthogonal_basis[0] = m1.reshape(-1)

    for i in range(1, slice_dim):
        obasis = rng.uniform(-1, 1, function_dim).reshape(-1, 3)
        obasis /= np.linalg.norm(obasis, axis=1, keepdims=True)
        for j in range(i):
            dot_products = np.sum(
                obasis * orthogonal_basis[j].reshape(-1, 3),
                axis=1,
                keepdims=True,
            )
            obasis -= dot_products * orthogonal_basis[j].reshape(-1, 3)
            obasis /= np.linalg.norm(obasis, axis=1, keepdims=True)

        orthogonal_basis[i] = obasis.reshape(-1)

    return orthogonal_basis


def normalised(x: torch.Tensor) -> torch.Tensor:
    norm = x.norm(dim=1, keepdim=True)
    return torch.where(norm > 0, x / norm, torch.zeros_like(x))