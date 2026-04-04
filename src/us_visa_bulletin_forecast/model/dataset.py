"""PyTorch Dataset and DataLoader for visa bulletin time series."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class VisaBulletinDataset(Dataset):
    """Sliding window dataset for multi-target visa bulletin prediction.

    Each sample:
        X: (lookback_window, num_features)
        y: (forecast_horizon, num_targets) - movement for all EB×country combos
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, noise_std: float = 0.01):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        if self.augment:
            x = x + torch.randn_like(x) * self.noise_std
        return x, y

    @property
    def num_features(self):
        return self.X.shape[-1]

    @property
    def num_targets(self):
        return self.y.shape[-1]

    @property
    def lookback_window(self):
        return self.X.shape[1]

    @property
    def forecast_horizon(self):
        return self.y.shape[1]


def create_dataloaders(data: dict, batch_size: int = 32) -> dict:
    """Create DataLoaders from prepared feature data."""
    train_ds = VisaBulletinDataset(data["X_train"], data["y_train"], augment=True)
    val_ds = VisaBulletinDataset(data["X_val"], data["y_val"])
    test_ds = VisaBulletinDataset(data["X_test"], data["y_test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "num_features": train_ds.num_features,
        "num_targets": train_ds.num_targets,
        "lookback_window": train_ds.lookback_window,
        "forecast_horizon": train_ds.forecast_horizon,
    }
