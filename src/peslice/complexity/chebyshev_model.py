# ruff: noqa: F821, F722

import functools

import numpy as np
from jaxtyping import Float
from typing import Optional
from numpy.polynomial.chebyshev import Chebyshev
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def outer_product(
    a: Float[np.ndarray, "N A"], b: Float[np.ndarray, "N B"]
) -> Float[np.ndarray, "N A*B"]:
    N, A = a.shape
    B = b.shape[-1]
    return (a[:, None, :] * b[:, :, None]).reshape(N, A * B)


def outer_sum(
    a: Float[np.ndarray, "N A"], b: Float[np.ndarray, "N B"]
) -> Float[np.ndarray, "N A*B"]:
    N, A = a.shape
    B = b.shape[-1]
    return (a[:, None, :] + b[:, :, None]).reshape(N, A * B)


def chebyshev_features_1d(
    x: Float[np.ndarray, "N"], n: int
) -> Float[np.ndarray, "N n"]:
    """
    Expand ``x`` of shape ``(N,)`` into ``n`` Chebyshev polynomials,
    giving a feature matrix of shape ``(N, n)``.
    """
    assert x.ndim == 1
    features = np.zeros((len(x), n))
    for i in range(n):
        T_i = Chebyshev.basis(i)
        features[:, i] = T_i(x.flatten())
    return features


def chebyshev_features(
    X: Float[np.ndarray, "N A"], n: int
) -> Float[np.ndarray, "N n**A"]:
    """
    Expand ``X`` of shape ``(N, A)`` into a basis of ``A``
    lots of ``(n+1)``-dimensional Chebyshev polynomials.
    Take the product of these features to get a
    ``(N, (n+1)**A)``-shaped feature matrix.
    """
    assert X.ndim == 2
    if X.shape[-1] == 1:
        return chebyshev_features_1d(X.flatten(), n)
    un_mixed_features = [chebyshev_features_1d(column, n) for column in X.T]
    return functools.reduce(outer_product, un_mixed_features)


def chebyshev_feature_importance(
    n: int,
    dims: int = 1,
) -> Float[np.ndarray, "n**dims"]:
    """
    Compute the "importance" of each feature in the Chebyshev basis.
    """
    importances = [np.arange(n) + 1 for _ in range(dims)]
    if dims == 1:
        return importances[0]

    return np.sqrt(
        functools.reduce(
            outer_sum,
            [imp.reshape(1, -1) ** 2 for imp in importances],
        )
    ).flatten()


class ChebyshevBasisExpansion(BaseEstimator, TransformerMixin):
    def __init__(self, degree=2):
        self.degree = degree

    def fit(self, X, y=None):
        # nothing to fit
        return self

    def transform(self, X):
        # Perform the transformation (custom basis expansion)
        return chebyshev_features(X, self.degree)


class ChebyshevModel(Pipeline):
    def __init__(
        self,
        *,
        alpha: float = 0.0005,
        order: int = 100,
    ):
        super().__init__(
            [
                ("minmax", MinMaxScaler(feature_range=(-1, 1))),
                ("chebyshev", ChebyshevBasisExpansion(degree=order)),
                ("linear", Lasso(alpha=alpha, fit_intercept=True)),
            ]
        )

        self.alpha = alpha
        self.order = order
        self._dimension: Optional[int] = None

    def fit(  # type: ignore
        self,
        X: Float[np.ndarray, "N A"],
        y: Float[np.ndarray, "N"],
    ):
        self.std = np.std(y)
        if X.ndim == 1:
            X = X[:, None]
        super().fit(X, y)
        self._dimension = X.shape[-1]
        return self

    def predict(  # type: ignore
        self,
        X: Float[np.ndarray, "N A"],
    ) -> Float[np.ndarray, "N"]:
        if X.ndim == 1:
            X = X[:, None]
        return super().predict(X)  # type: ignore

    def score(  # type: ignore
        self,
        X: Float[np.ndarray, "N A"],
        y: Float[np.ndarray, "N"],
        sample_weight=None,
    ):
        if X.ndim == 1:
            X = X[:, None]
        return super().score(X, y, sample_weight=sample_weight)

    @property
    def complexity(self):
        if self._dimension is None:
            raise ValueError("Model must be fitted before computing complexity")

        importances = chebyshev_feature_importance(self.order, self._dimension)

        coeff = self.named_steps["linear"].coef_

        if np.abs(coeff).sum() == 0:
            return 0.0
        weights = np.abs(coeff) / np.abs(coeff).sum()
        return float(np.sum(weights * importances) * self.std)

    @property
    def coefficients(self):
        return self.named_steps["linear"].coef_
