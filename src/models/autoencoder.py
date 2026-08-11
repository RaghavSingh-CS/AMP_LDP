"""Fully connected baseline Autoencoder for 30-value consumption windows."""

from __future__ import annotations

from torch import Tensor, nn


INPUT_DIM = 30
LATENT_DIM = 4
HIDDEN_DIMS = (16, 8)


class Autoencoder(nn.Module):
    """Exact 30 -> 16 -> 8 -> 4 -> 8 -> 16 -> 30 baseline."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(INPUT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, LATENT_DIM),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, INPUT_DIM),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Reconstruct a batch of standardized consumption windows."""
        if inputs.ndim != 2 or inputs.shape[1] != INPUT_DIM:
            raise ValueError(
                f"Autoencoder expects shape (batch, {INPUT_DIM}); "
                f"received {tuple(inputs.shape)}"
            )
        return self.decoder(self.encoder(inputs))


def architecture_config() -> dict[str, object]:
    """Return a JSON-serializable description of the fixed architecture."""
    return {
        "type": "fully_connected_autoencoder",
        "layer_dimensions": [30, 16, 8, 4, 8, 16, 30],
        "hidden_activation": "ReLU",
        "output_activation": "Linear",
        "input_dimension": INPUT_DIM,
        "latent_dimension": LATENT_DIM,
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Count model parameters that receive gradients."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
