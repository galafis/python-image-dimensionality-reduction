"""t-SNE dimensionality reduction for images."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


class TSNEReducer:
    """t-SNE dimensionality reducer."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            n_iter=1000,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        logger.info(f"Fitting t-SNE with {self.n_components} components")
        return self.tsne.fit_transform(X)

    def evaluate(
        self,
        X: np.ndarray,
        X_reduced: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict:
        metrics = {
            "kl_divergence": round(float(self.tsne.kl_divergence_), 4),
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
