from pathlib import Path
import pickle

import numpy as np


RANDOM_STATE = 42


def load_fashion_mnist_hog(data_dir, load_fashion_mnist):
    output_dir = Path(data_dir) / "openml" / "hog"
    features_path = output_dir / "fashion_mnist_hog_features.npy"
    labels_path = output_dir / "fashion_mnist_hog_labels.npy"

    if features_path.exists() and labels_path.exists():
        return np.load(features_path), np.load(labels_path)

    try:
        from skimage.feature import hog
    except ImportError as exc:
        raise ImportError("Fashion-MNIST HOG requires: pip install scikit-image") from exc

    X, y = load_fashion_mnist()
    output_dir.mkdir(parents=True, exist_ok=True)

    features = []
    for idx, image in enumerate(X):
        if idx % 5000 == 0:
            print(f"Extracting Fashion-MNIST HOG features: {idx}/{len(X)}")
        feature = hog(
            image.reshape(28, 28),
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
        features.append(feature)

    X_hog = np.asarray(features, dtype=np.float32)
    np.save(features_path, X_hog)
    np.save(labels_path, y)
    return X_hog, y


def load_cifar10_cnn(data_dir):
    output_dir = Path(data_dir) / "cifar10" / "cnn_features"
    features_path = output_dir / "resnet18_features.npy"
    labels_path = output_dir / "resnet18_labels.npy"

    if features_path.exists() and labels_path.exists():
        return np.load(features_path), np.load(labels_path)

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise ImportError("CIFAR-10 CNN features require: pip install torchvision") from exc

    cifar_dir = Path(data_dir) / "cifar10" / "cifar-10-batches-py"
    X_parts = []
    y_parts = []
    for batch_id in range(1, 6):
        with open(cifar_dir / f"data_batch_{batch_id}", "rb") as file:
            batch = pickle.load(file, encoding="latin1")
        X_parts.append(batch["data"])
        y_parts.extend(batch["labels"])

    X = np.vstack(X_parts).astype(np.float32) / 255.0
    y = np.asarray(y_parts, dtype=np.int64)

    images = X.reshape(-1, 3, 32, 32)
    images = torch.from_numpy(images)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    loader = DataLoader(TensorDataset(images), batch_size=128, shuffle=False)
    features = []

    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(loader):
            if batch_idx % 20 == 0:
                done = min(batch_idx * 128, len(images))
                print(f"Extracting CIFAR-10 CNN features: {done}/{len(images)}")
            batch = batch.to(device)
            batch = torch.nn.functional.interpolate(
                batch,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            batch = (batch - mean) / std
            features.append(model(batch).cpu().numpy())

    X_cnn = np.vstack(features).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(features_path, X_cnn)
    np.save(labels_path, y)
    return X_cnn, y
