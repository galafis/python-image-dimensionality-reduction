"""PCA-based dimensionality reduction for images."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

logger = logging.getLogger(__name__)


class PCAReducer:
    """PCA dimensionality reducer with evaluation metrics."""

    def __init__(self, n_components: int = 2, random_state: int = 42):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit PCA and transform data."""
        logger.info(f"Fitting PCA with {self.n_components} components")
        X_reduced = self.pca.fit_transform(X)
        self._fitted = True
        variance = sum(self.pca.explained_variance_ratio_)
        logger.info(f"Explained variance: {variance:.4f}")
        return X_reduced

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """Reconstruct original data from reduced representation."""
        return self.pca.inverse_transform(X_reduced)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Calculate mean reconstruction error."""
        X_reduced = self.pca.transform(X)
        X_reconstructed = self.pca.inverse_transform(X_reduced)
        return float(np.mean((X - X_reconstructed) ** 2))

    def evaluate(
        self,
        X: np.ndarray,
        X_reduced: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict:
        """Evaluate reduction quality."""
        metrics = {
            "explained_variance": round(
                float(sum(self.pca.explained_variance_ratio_)), 4
            ),
            "reconstruction_error": round(self.reconstruction_error(X), 4),
            "trustworthiness": round(
                float(trustworthiness(X, X_reduced, n_neighbors=5)), 4
            ),
        }
        if y is not None and len(np.unique(y)) > 1:
            metrics["silhouette_score"] = round(
                float(silhouette_score(X_reduced, y, sample_size=min(1000, len(y)))),
                4,
            )
        return metrics
