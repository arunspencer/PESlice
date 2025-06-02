import numpy as np


def mac_complexity(
    y: np.ndarray,
    dx: float,
    normalise: bool = False
) -> float:
    d2x = np.gradient(np.gradient(y, dx), dx)

    mac = np.abs(d2x).mean()

    return mac / (y.std() if normalise else 1)
