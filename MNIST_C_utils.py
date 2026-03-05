import os
import urllib.request
import zipfile
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ==========================================
# MNIST-C Dataset Utilities
# ==========================================

# The official 15 corruptions + the clean 'identity' dataset
CORRUPTIONS = [
    'identity', 'shot_noise', 'impulse_noise', 'glass_blur', 'motion_blur',
    'shear', 'scale', 'rotate', 'brightness', 'translate', 'stripe',
    'fog', 'spatter', 'dotted_line', 'zigzag', 'canny_edges',
]


class MNISTCDataset(Dataset):
    """MNIST-C corruption benchmark dataset.

    Images are returned as float32 tensors in [0, 1] with shape [1, 28, 28],
    matching the normalisation used by ``load_mnist()`` in MNIST.py
    (i.e. ``data.float().div(255.0)`` — no extra mean/std normalisation).
    """

    def __init__(self, root='./data', corruption='identity', train=False, download=True):
        self.root = root
        self.corruption = corruption
        self.train = train

        if download:
            self._download()

        prefix = 'train' if self.train else 'test'
        img_path = os.path.join(self.root, 'mnist_c', self.corruption, f'{prefix}_images.npy')
        lbl_path = os.path.join(self.root, 'mnist_c', self.corruption, f'{prefix}_labels.npy')

        if not os.path.exists(img_path):
            raise RuntimeError(f"Dataset not found at {img_path}. Set download=True.")

        self.data = np.load(img_path)      # uint8, shape [N, 28, 28] or [N, 28, 28, 1]
        self.targets = np.load(lbl_path)   # int

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])

        # Ensure [H, W] then convert to float [1, H, W] in [0, 1]
        if img.ndim == 3:
            img = img.squeeze(-1)           # [H, W, 1] -> [H, W]
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).div(255.0)

        return img, target

    def _download(self):
        if os.path.exists(os.path.join(self.root, 'mnist_c')):
            return

        os.makedirs(self.root, exist_ok=True)
        url = "https://zenodo.org/record/3239543/files/mnist_c.zip"
        zip_path = os.path.join(self.root, "mnist_c.zip")

        print(f"Downloading MNIST-C to {zip_path}... (This is a ~250MB file, please wait)")
        urllib.request.urlretrieve(url, zip_path)

        print("Extracting MNIST-C...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.root)
        print("Download and extraction complete.\n")


def ensure_mnist_c_downloaded(root='./data'):
    """Trigger download once (idempotent)."""
    MNISTCDataset(root=root, corruption='identity', train=False, download=True)
