import numpy as np


def mac_complexity(
    y: np.ndarray,
    dx: float,
    normalise: bool = False
) -> float:
    """
    Calculates the mean absolute curvature (MAC) complexity of a 1D function

    The curvature is defined as the second derivative of the function

    Parameters
    ----------
    y
        1D array of function values
    dx
        Discrete spacing between points in the 1D function
    normalise
        Whether to normalise the MAC complexity by the standard deviation of the function values

    Returns
    -------
    float
        The mean absolute curvature (MAC) complexity of the 1D function
    """
    d2x = np.gradient(np.gradient(y, dx), dx)

    mac = np.abs(d2x).mean()

    return mac / (y.std() if normalise else 1)
