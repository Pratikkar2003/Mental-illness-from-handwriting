"""
dataset.py
----------
Expects a folder layout like:

    data/
      train/
        happy/    *.png / *.jpg
        sad/
        stressed/
        calm/
      val/
        happy/
        sad/
        stressed/
        calm/

Each subfolder name under train/ and val/ becomes a class label.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from feature_extraction import extract_features_from_path


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def discover_classes(split_dir: str):
    classes = sorted([
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d))
    ])
    if not classes:
        raise RuntimeError(f"No class subfolders found in {split_dir}")
    return classes


class HandwritingEmotionDataset(Dataset):
    def __init__(self, split_dir: str, classes=None, cache_features: bool = True):
        self.split_dir = split_dir
        self.classes = classes or discover_classes(split_dir)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for c in self.classes:
            folder = os.path.join(split_dir, c)
            if not os.path.isdir(folder):
                continue
            for ext in IMG_EXTS:
                for path in glob.glob(os.path.join(folder, f"*{ext}")):
                    self.samples.append((path, self.class_to_idx[c]))
        if not self.samples:
            raise RuntimeError(f"No images found under {split_dir}")
        self.cache_features = cache_features
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.cache_features and path in self._cache:
            feats = self._cache[path]
        else:
            feats = extract_features_from_path(path)
            if self.cache_features:
                self._cache[path] = feats
        return (
            torch.from_numpy(feats.point_sequence),
            torch.from_numpy(feats.seq_mask),
            torch.from_numpy(feats.global_features),
            torch.tensor(label, dtype=torch.long),
        )
