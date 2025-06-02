from .core.linear_slice import LinearSlicePES
from .core.circular_slice import CircularSlicePES
from .complexity.chebyshev_complexity import chebyshev_complexity
from .complexity.chebyshev_model import ChebyshevModel
from .complexity.mac_complexity import mac_complexity

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "LinearSlicePES",
    "CircularSlicePES",
    "ChebyshevModel",
    "mac_complexity",
    "chebyshev_complexity"
]

def info():
    print(
        "PESlice v0.1.0 — Tools for slicing high-dimensional PES and computing complexity measures.\n"
        "Includes:\n"
        "  • CircularSlicePES — define and evaluate 1D circular slices through PES using graph-pes models, and quantify function complexities.\n"
        "  • ChebyshevModel — fit and predict data with Chebyshev polynomial basis expansion, and quantify function complexities.\n"
        "  • chebyshev_complexity — quantify function complexity using Chebyshev polynomial expansion.\n"
        "  • mac_complexity — compute the mean absolute curvature (MAC) of 1D functions as a measure of complexity."
    )

