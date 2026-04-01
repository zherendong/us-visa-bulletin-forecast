"""Training loop for multi-target green card timeline prediction."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gc_predict.model.architecture import LSTMBaseline, QuantileLoss, TemporalFusionTransformer

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 200,
    lr: float = 5e-4,
    weight_decay: float = 0.01,
    patience: int = 30,
    max_grad_norm: float = 1.0,
    checkpoint_dir: Path | None = None,
    device: str | None = None,
) -> dict:
    """Train with early stopping and cosine annealing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training on {device}")

    model = model.to(device)
    criterion = QuantileLoss(quantiles=(0.1, 0.5, 0.9))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_losses.append(loss.item())

        avg_train = sum(train_losses) / len(train_losses)
        avg_val = sum(val_losses) / len(val_losses) if val_losses else 0
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
                f"LR: {scheduler.get_last_lr()[0]:.6f}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        name = "tft" if isinstance(model, TemporalFusionTransformer) else "lstm"
        path = checkpoint_dir / f"{name}_best.pt"
        torch.save({"model_state_dict": best_state, "history": history,
                     "best_val_loss": best_val_loss}, path)
        logger.info(f"Saved checkpoint to {path}")

    return {"model": model, "history": history, "best_val_loss": best_val_loss}


def create_and_train_lstm(data_loaders: dict, checkpoint_dir: Path | None = None, **kwargs) -> dict:
    model = LSTMBaseline(
        num_features=data_loaders["num_features"],
        num_targets=data_loaders["num_targets"],
        forecast_horizon=data_loaders["forecast_horizon"],
    )
    return train_model(model, data_loaders["train_loader"], data_loaders["val_loader"],
                       checkpoint_dir=checkpoint_dir, **kwargs)


def create_and_train_tft(data_loaders: dict, checkpoint_dir: Path | None = None, **kwargs) -> dict:
    model = TemporalFusionTransformer(
        num_features=data_loaders["num_features"],
        num_targets=data_loaders["num_targets"],
        forecast_horizon=data_loaders["forecast_horizon"],
    )
    return train_model(model, data_loaders["train_loader"], data_loaders["val_loader"],
                       checkpoint_dir=checkpoint_dir, **kwargs)
