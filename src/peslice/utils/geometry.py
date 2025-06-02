# ruff: noqa: F821, F722


import torch
import numpy as np
from jaxtyping import Float


def check_orthogonal(*args, function_dim) -> bool:
    """
    Checks whether all parsed vectors are orthogonal

    Vectors parsed to this function must be contain sub-vectors of length 3 (x, y, z)

    Orthogonality of the sub-vectors are checked

    Parameters
    ----------
    *args
        n vectors to check sub-vector orthogonality

        Shape ``(function_dim, )``

    Returns
    -------
    Whether the sub-vectors are all orthogonal

    Raises
    ------
    ValueError
        If all of the parsed vectors are not equal to the function dimension

        If the function dimension cannot be split into Cartesian sub-vectors of length 3
    """
    if function_dim % 3 != 0:
        raise ValueError(
            "The function dimension must be a multiple of 3 to form Cartesian sub-vectors"
        )
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
    Generates a random slice (hyperplane) in 1-, 2-, or 3-dimensions

    The overall dimension is the length of the ``m0`` vector

    Parameters
    ----------
    m0
        A point on the hyperplane

        Shape ``(function_dim, )``
    slice_dim
        The dimension of the hyperplane
    seed
        The seed for random number generation

    Returns
    -------
    orthogonal_basis
        A random orthogonal basis for the hyperplane

        Shape ``(slice_dim, function_dim)``

    Raises
    ------
    ValueError
        If the slice dimension is greater than the function dimension or is < 1

        If slice_dim is not 1, 2, or 3
    """

    rng = np.random.default_rng(seed)

    function_dim = len(m0)

    # checking that the slice dimension is in [1, 2, 3]
    if slice_dim > 3 or slice_dim < 1:
        raise ValueError("Slice dimension must be 1, 2, or 3")

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
    """
    Normalises each row vector of the input tensor

    Parameters
    ----------
    x
        The input tensor to normalise

    Returns
    -------
    torch.Tensor
        The normalised tensor
    """
    norm = x.norm(dim=1, keepdim=True)
    return torch.where(norm > 0, x / norm, torch.zeros_like(x))