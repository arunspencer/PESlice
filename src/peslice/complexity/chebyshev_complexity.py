# ruff: noqa: F821, F722


from .chebyshev_model import ChebyshevModel
import numpy as np
from jaxtyping import Float

def chebyshev_complexity(X: Float[np.ndarray, "N A"],
    y: Float[np.ndarray, "N"],
    alpha: float = 0.0005,
    max_order: int = 100
) -> float:
    """
    See :class:`ChebyshevModel` for more details

    Parameters
    ----------
    alpha
        Regularisation parameter for the Lasso regression

        Default is 0.0005
    max_order : int
        Maximum order of the Chebyshev polynomial basis

        Default is 100
    
    Returns
    -------
    float
        The Chebyshev complexity of the function y = f(X)
    """

    model = ChebyshevModel(alpha=alpha, order=max_order)
    model.fit(X, y)
    
    return model.complexity