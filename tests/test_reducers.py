"""Tests for dimensionality reduction methods."""

import numpy as np
import pytest


class TestPCAReducer:
    def test_fit_transform_shape(self):
        from src.reducers.pca_reducer import PCAReducer
        X = np.random.randn(100, 50)
        reducer = PCAReducer(n_components=2)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (100, 2)

    def test_reconstruction_error(self):
        from src.reducers.pca_reducer import PCAReducer
        X = np.random.randn(100, 50)
        reducer = PCAReducer(n_components=10)
        reducer.fit_transform(X)
        error = reducer.reconstruction_error(X)
        assert error >= 0

    def test_evaluate_returns_metrics(self):
        from src.reducers.pca_reducer import PCAReducer
        X = np.random.randn(100, 50)
        y = np.random.randint(0, 5, 100)
        reducer = PCAReducer(n_components=2)
        X_reduced = reducer.fit_transform(X)
        metrics = reducer.evaluate(X, X_reduced, y)
        assert "explained_variance" in metrics
        assert "reconstruction_error" in metrics
        assert "trustworthiness" in metrics


class TestTSNEReducer:
    def test_fit_transform_shape(self):
        from src.reducers.tsne_reducer import TSNEReducer
        X = np.random.randn(50, 30)
        reducer = TSNEReducer(n_components=2)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (50, 2)


class TestUMAPReducer:
    def test_fit_transform_shape(self):
        from src.reducers.umap_reducer import UMAPReducer
        X = np.random.randn(100, 50)
        reducer = UMAPReducer(n_components=2)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (100, 2)


class TestAutoencoder:
    def test_fit_transform_shape(self):
        from src.reducers.autoencoder_reducer import AutoencoderReducer
        X = np.random.randn(100, 50).astype(np.float32)
        reducer = AutoencoderReducer(input_dim=50, encoding_dim=2, epochs=5)
        X_reduced = reducer.fit_transform(X)
        assert X_reduced.shape == (100, 2)
