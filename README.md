# Image Dimensionality Reduction for Neural Networks

[English](#english) | [Portugues](#portugues)

---

## English

### Overview

Production-ready image dimensionality reduction toolkit for neural networks. Compare PCA, t-SNE, UMAP, and autoencoder approaches for image data preprocessing and feature extraction. Includes visualization pipeline, reconstruction quality metrics, and performance benchmarks.

**DIO Lab Project** - Formacao Machine Learning Specialist

### Features

- **PCA (Principal Component Analysis)**: Linear dimensionality reduction with variance analysis
- **t-SNE**: Non-linear embedding for high-dimensional visualization
- **UMAP**: Fast manifold learning for large-scale datasets
- **Autoencoder**: Deep learning-based compression with reconstruction
- **Visualization Pipeline**: Side-by-side comparison of reduction methods
- **Quality Metrics**: Reconstruction error, explained variance, silhouette score
- **Benchmark Suite**: Compare methods on speed, quality, and memory usage
- **Docker Support**: Containerized execution environment
- **CI/CD Pipeline**: GitHub Actions with linting and testing

### Architecture

```
src/
|-- reducers/         # Dimensionality reduction implementations
|   |-- pca.py        # PCA reducer
|   |-- tsne.py       # t-SNE reducer
|   |-- umap_reducer.py # UMAP reducer
|   |-- autoencoder.py  # Autoencoder reducer
|-- evaluation/       # Quality metrics and benchmarks
|-- visualization/    # Plotting and comparison tools
|-- pipeline.py       # Main orchestration pipeline
tests/                # Unit tests
```

### Quick Start

```bash
git clone https://github.com/galafis/python-image-dimensionality-reduction.git
cd python-image-dimensionality-reduction
pip install -r requirements.txt
python -m src.pipeline --method all --dataset mnist
```

### Technologies

- Python 3.10+ | scikit-learn | UMAP-learn | TensorFlow/Keras
- NumPy | Pandas | Matplotlib | Seaborn
- Docker | GitHub Actions | Pytest

---

## Portugues

### Visao Geral

Kit de reducao de dimensionalidade de imagens para redes neurais, pronto para producao. Compare abordagens PCA, t-SNE, UMAP e autoencoder para pre-processamento de dados de imagem e extracao de features. Inclui pipeline de visualizacao, metricas de qualidade de reconstrucao e benchmarks de desempenho.

**Projeto de Lab DIO** - Formacao Machine Learning Specialist

### Funcionalidades

- **PCA**: Reducao de dimensionalidade linear com analise de variancia
- **t-SNE**: Embedding nao-linear para visualizacao de alta dimensao
- **UMAP**: Aprendizado de manifold rapido para datasets de grande escala
- **Autoencoder**: Compressao baseada em deep learning com reconstrucao
- **Pipeline de Visualizacao**: Comparacao lado a lado dos metodos
- **Metricas de Qualidade**: Erro de reconstrucao, variancia explicada, silhouette score
- **Suite de Benchmark**: Compare metodos em velocidade, qualidade e uso de memoria

---

## License / Licenca

MIT License - see [LICENSE](LICENSE) for details.

## Author / Autor

**Gabriel Demetrios Lafis**
- GitHub: [@galafis](https://github.com/galafis)
- LinkedIn: [Gabriel Lafis](https://linkedin.com/in/gabriel-lafis)
