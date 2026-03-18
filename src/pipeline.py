"""Main pipeline for image dimensionality reduction comparison."""

import argparse
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from src.reducers.pca_reducer import PCAReducer
from src.reducers.tsne_reducer import TSNEReducer
from src.reducers.umap_reducer import UMAPReducer
from src.reducers.autoencoder_reducer import AutoencoderReducer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(
    name: str = "mnist",
    n_samples: Optional[int] = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and preprocess image dataset."""
    logger.info(f"Loading {name} dataset...")
    if name == "mnist":
        data = fetch_openml("mnist_784", version=1, as_frame=False)
        X, y = data.data, data.target.astype(int)
    elif name == "fashion":
        data = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
        X, y = data.data, data.target.astype(int)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    if n_samples and n_samples < len(X):
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info(f"Loaded {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
    return X_scaled, y


def run_comparison(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 2,
    methods: Optional[List[str]] = None,
) -> Dict:
    """Run dimensionality reduction comparison."""
    if methods is None:
        methods = ["pca", "tsne", "umap", "autoencoder"]

    reducers = {
        "pca": PCAReducer(n_components=n_components),
        "tsne": TSNEReducer(n_components=n_components),
        "umap": UMAPReducer(n_components=n_components),
        "autoencoder": AutoencoderReducer(
            input_dim=X.shape[1], encoding_dim=n_components
        ),
    }

    results = {}
    for name in methods:
        if name not in reducers:
            logger.warning(f"Unknown method: {name}")
            continue

        reducer = reducers[name]
        logger.info(f"Running {name}...")

        start = time.time()
        X_reduced = reducer.fit_transform(X)
        elapsed = time.time() - start

        metrics = reducer.evaluate(X, X_reduced, y)
        metrics["time_seconds"] = round(elapsed, 2)

        results[name] = {
            "embedding": X_reduced,
            "metrics": metrics,
        }
        logger.info(f"{name}: {metrics}")

    return results


def print_summary(results: Dict) -> None:
    """Print comparison summary table."""
    print("\n" + "=" * 60)
    print(f"{'Method':<15} {'Time (s)':<12} {'Silhouette':<12} {'Trustworth.':<12}")
    print("=" * 60)
    for name, data in results.items():
        m = data["metrics"]
        print(
            f"{name:<15} {m.get('time_seconds', 'N/A'):<12} "
            f"{m.get('silhouette_score', 'N/A'):<12} "
            f"{m.get('trustworthiness', 'N/A'):<12}"
        )
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Dimensionality Reduction")
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--method", type=str, default="all")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()

    methods = None if args.method == "all" else [args.method]
    X, y = load_dataset(args.dataset, args.n_samples)
    results = run_comparison(X, y, args.n_components, methods)
    print_summary(results)
