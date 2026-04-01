"""Model architectures for green card timeline prediction.

Multi-target: predicts movement for ALL EB×country combos simultaneously.
Each target gets quantile predictions (q10, q50, q90) for confidence intervals.
"""

import torch
import torch.nn as nn


class QuantileLoss(nn.Module):
    """Pinball loss for multi-target quantile regression."""

    def __init__(self, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, horizon, num_targets, num_quantiles)
            targets: (batch, horizon, num_targets)
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = targets - predictions[..., i]
            losses.append(torch.max((q - 1) * errors, q * errors))
        return torch.stack(losses, dim=-1).mean()


class LSTMBaseline(nn.Module):
    """Multi-target LSTM with quantile outputs."""

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        forecast_horizon: int = 6,
        num_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets
        self.num_quantiles = num_quantiles

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, forecast_horizon * num_targets * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (batch, horizon, num_targets, num_quantiles)."""
        lstm_out, _ = self.lstm(x)
        last = self.dropout(lstm_out[:, -1, :])
        out = self.fc(last)
        return out.view(-1, self.forecast_horizon, self.num_targets, self.num_quantiles)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from the TFT paper."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(output_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm(output_size)
        self.skip = nn.Linear(input_size, output_size) if input_size != output_size else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x) if self.skip else x
        hidden = self.elu(self.fc1(x))
        hidden = self.dropout(self.fc2(hidden))
        gate = self.sigmoid(self.gate(hidden))
        return self.layer_norm(gate * hidden + residual)


class TemporalFusionTransformer(nn.Module):
    """Multi-target TFT for time series forecasting.

    Predicts all EB×country movements simultaneously with quantile outputs.
    """

    def __init__(
        self,
        num_features: int,
        num_targets: int,
        hidden_size: int = 64,
        num_lstm_layers: int = 2,
        num_attention_heads: int = 4,
        forecast_horizon: int = 6,
        num_quantiles: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.num_targets = num_targets
        self.num_quantiles = num_quantiles

        # Input projection (instead of per-variable VSN which is too heavy for small data)
        self.input_proj = nn.Sequential(
            nn.Linear(num_features, hidden_size),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)

        # Post-attention processing
        self.post_attn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)

        # Output: predict all targets × quantiles for each horizon step
        self.output_fc = nn.Linear(hidden_size, forecast_horizon * num_targets * num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, lookback, features)
        Returns:
            (batch, horizon, num_targets, num_quantiles)
        """
        # Project input
        projected = self.input_proj(x)

        # LSTM
        lstm_out, _ = self.lstm(projected)

        # Self-attention with residual
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.attn_norm(attn_out + lstm_out)

        # Post-attention
        processed = self.post_attn(attn_out)

        # Use last timestep
        last = processed[:, -1, :]
        out = self.output_fc(last)

        return out.view(-1, self.forecast_horizon, self.num_targets, self.num_quantiles)
