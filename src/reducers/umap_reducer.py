"""UMAP dimensionality reduction for images."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

logger = logging.getLogger(__name__)


class UMAPReducer:
    """UMAP dimensionality reducer."""

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        random_state: int = 42,
    ):
        self.n_components = n_components
        try:
            import umap
            self.reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                random_state=random_state,
            )
        except ImportError:
            logger.warning("umap-learn not installed, using PCA fallback")
            from sklearn.decomposition import PCA
            self.reducer = PCA(n_components=n_components)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        logger.info(f"Fitting UMAP with {self.n_components} components")
        return self.reducer.fit_transform(X)

    def evaluate(
        self,
        X: np.ndarray,
        X_reduced: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict:
        metrics = {
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
