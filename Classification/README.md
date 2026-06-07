# RP ML Evaluation Framework

This codebase evaluates dimensionality reduction methods for downstream classification tasks.

## Supported datasets

- `fashion_mnist`: dense 784-dimensional grayscale image vectors from OpenML.
- `cifar10`: dense 3,072-dimensional RGB image vectors via torchvision.
- `uci_har`: dense 561-dimensional processed sensor features from UCI HAR.
- `20newsgroups`: sparse TF-IDF text features from scikit-learn `fetch_20newsgroups` + `TfidfVectorizer`.
- `rcv1`: sparse 47,236-dimensional text features via scikit-learn. This loader converts multilabel targets into a single-label proxy task by using the first positive label.

## Supported dimensionality reduction methods

- `identity`: no dimensionality reduction baseline.
- `sparse_rp`: Sparse Random Projection.
- `pca_svd`: PCA using full SVD through scikit-learn `PCA(svd_solver="full")`.
- `dct`: DCT-based low-frequency feature selection. Uses 2D DCT for image datasets and 1D DCT otherwise.
- `randomized_svd`: randomized SVD via scikit-learn `TruncatedSVD(algorithm="randomized")`.

## Supported classifiers

- `knn`: k-Nearest Neighbors.
- `svm`: linear SVM using `LinearSVC`.
- `dnn`: PyTorch MLP classifier.

## Output

The script writes one CSV record per run / dataset / reducer / dimension / model combination. Multiple repeated runs are not averaged; each run is kept as a separate row.

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a small Fashion-MNIST experiment:

```bash
python main.py \
  --datasets fashion_mnist \
  --reducers identity sparse_rp pca_svd dct randomized_svd \
  --models knn svm \
  --dims 50 100 200 \
  --runs 3 \
  --max-train 10000 \
  --max-test 2000 \
  --output results/fashion_mnist.csv
```

Run CIFAR-10 with kNN and SVM:

```bash
python main.py \
  --datasets cifar10 \
  --reducers identity sparse_rp dct randomized_svd \
  --models knn svm \
  --dims 128 256 512 1024 \
  --runs 3 \
  --max-train 10000 \
  --max-test 2000 \
  --output results/cifar10.csv
```

Run a small DNN experiment:

```bash
python main.py \
  --datasets fashion_mnist \
  --reducers identity sparse_rp pca_svd dct randomized_svd \
  --models dnn \
  --dims 100 200 \
  --runs 2 \
  --max-train 10000 \
  --max-test 2000 \
  --dnn-epochs 5 \
  --dnn-hidden-dim 256 \
  --output results/fashion_mnist_dnn.csv
```

Run RCV1 with scalable methods only:

```bash
python main.py \
  --datasets rcv1 \
  --reducers identity sparse_rp randomized_svd \
  --models svm \
  --dims 100 300 500 \
  --runs 3 \
  --max-train 10000 \
  --max-test 2000 \
  --output results/rcv1_svm.csv
```
