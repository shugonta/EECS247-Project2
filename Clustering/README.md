# Project 2: Extending Random Projection for Clustering

## Overview
This project compares PCA, Gaussian Random Projection, Sparse Random Projection, and DCT for downstream clustering. K-Means and Gaussian Mixture Models are evaluated using clustering metrics, runtime, and distance-preservation results.

## Requirements
- Python 3.8+ recommended
- Prepared datasets in `data/`, or internet access for `download_data.py`
- Optional GPU support for CIFAR-10 CNN feature extraction

## Dependencies
Main dependencies include `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `pillow`, `umap-learn`, `scikit-image`, `torch`, and `torchvision`.

Install with pip:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pillow umap-learn scikit-image torch torchvision
```

## Downloading / Preparing Data
Run the data preparation script to download datasets and generate cached HOG/CNN features:

```bash
python3 download_data.py
```

The project uses Fashion-MNIST, Fashion-MNIST HOG features, CIFAR-10, CIFAR-10 CNN features, and UCI HAR.

## Running the Main Evaluation
Run the full clustering experiment pipeline and save CSV results to `outputs/`:

```bash
python3 main.py --datasets fashion_mnist fashion_mnist_hog cifar10 cifar10_cnn uci_har --k 50 100 200 --sample_size 10000 --output outputs/results.csv
```

Key options:

- `--datasets`: datasets to evaluate
- `--k`: reduced dimensions to test
- `--sample_size`: maximum samples per dataset
- `--output`: result CSV path

## Plotting Aggregated Metrics
Generate metric and runtime plots from a result CSV:

```bash
python3 performance.py --csv outputs/results.csv
```

This writes plots to `metric_vs_k/`, `runtime_vs_k/`, and `runtime_breakdown/` under `outputs/`.

## Generating Cluster Visualizations
Generate distance distortion plots, UMAP grids, heatmaps, and sample galleries:

```bash
python3 visualization.py --datasets fashion_mnist cifar10 uci_har --selected_k 100 --sample_size 5000
```

## Outputs
- Evaluation CSVs: `results.csv`, `results_fashion_hog.csv`, `results_cifar10_cnn.csv`
- Metric/runtime plots: `metric_vs_k/`, `runtime_vs_k/`, `runtime_breakdown/`
- Cluster visualizations: `distance_distortion/`, `umap_cluster_grid/`, `cluster_heatmaps/`, `cluster_galleries/`

## File Guide
- `main.py`: runs experiments and saves CSV results
- `download_data.py`: prepares datasets and feature caches
- `performance.py`: plots metrics and runtime
- `visualization.py`: creates clustering visualizations
- `dim_reduction.py`, `clustering.py`, `evaluation.py`, `feature_extraction.py`: helper modules
