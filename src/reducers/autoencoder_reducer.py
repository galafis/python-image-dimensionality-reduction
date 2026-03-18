"""Autoencoder-based dimensionality reduction for images."""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness

logger = logging.getLogger(__name__)


class AutoencoderReducer:
    """Autoencoder dimensionality reducer using TensorFlow/Keras."""

    def __init__(
        self,
        input_dim: int = 784,
        encoding_dim: int = 2,
        hidden_dims: list = None,
        epochs: int = 50,
        batch_size: int = 256,
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.epochs = epochs
        self.batch_size = batch_size
        self.autoencoder = None
        self.encoder = None
        self._build_model()

    def _build_model(self):
        """Build autoencoder architecture."""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers

            # Encoder
            inputs = keras.Input(shape=(self.input_dim,))
            x = inputs
            for dim in self.hidden_dims:
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.BatchNormalization()(x)
            encoded = layers.Dense(self.encoding_dim, activation="linear", name="encoding")(x)

            # Decoder
            x = encoded
            for dim in reversed(self.hidden_dims):
                x = layers.Dense(dim, activation="relu")(x)
                x = layers.BatchNormalization()(x)
            decoded = layers.Dense(self.input_dim, activation="linear")(x)

            self.autoencoder = keras.Model(inputs, decoded, name="autoencoder")
            self.encoder = keras.Model(inputs, encoded, name="encoder")

            self.autoencoder.compile(
                optimizer="adam", loss="mse"
            )
            logger.info(f"Built autoencoder: {self.input_dim} -> {self.encoding_dim}")
        except ImportError:
            logger.error("TensorFlow not available")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Train autoencoder and return encoded representation."""
        if self.autoencoder is None:
            raise RuntimeError("Model not built")

        logger.info(f"Training autoencoder for {self.epochs} epochs")
        self.autoencoder.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0,
        )
        return self.encoder.predict(X, verbose=0)

    def evaluate(
        self,
        X: np.ndarray,
        X_reduced: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> Dict:
        X_reconstructed = self.autoencoder.predict(X, verbose=0)
        metrics = {
            "reconstruction_error": round(float(np.mean((X - X_reconstructed) ** 2)), 4),
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
