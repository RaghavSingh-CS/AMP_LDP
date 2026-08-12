"""Compact 1D CNN Autoencoder for 30-position consumption windows."""

from __future__ import annotations

from torch import Tensor, nn


INPUT_CHANNELS = 1
INPUT_LENGTH = 30
ENCODED_CHANNELS = 16
ENCODED_LENGTH = 5
FLATTENED_DIM = ENCODED_CHANNELS * ENCODED_LENGTH
LATENT_DIM = 8


class CNNAutoencoder(nn.Module):
    """Encode local temporal patterns and reconstruct a `(B, 1, 30)` tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.convolutional_encoder = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.to_latent = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FLATTENED_DIM, LATENT_DIM),
            nn.ReLU(),
        )
        self.from_latent = nn.Sequential(
            nn.Linear(LATENT_DIM, FLATTENED_DIM),
            nn.ReLU(),
        )
        self.convolutional_decoder = nn.Sequential(
            nn.ConvTranspose1d(16, 8, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 1, kernel_size=2, stride=2),
        )

    def encode(self, inputs: Tensor) -> Tensor:
        """Return the eight-value latent representation."""
        self._validate_input(inputs)
        features = self.convolutional_encoder(inputs)
        if features.shape[1:] != (ENCODED_CHANNELS, ENCODED_LENGTH):
            raise RuntimeError(
                "CNN encoder shape changed; expected "
                f"(batch, {ENCODED_CHANNELS}, {ENCODED_LENGTH}), got "
                f"{tuple(features.shape)}"
            )
        return self.to_latent(features)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode a `(B, 8)` latent tensor into `(B, 1, 30)`."""
        if latent.ndim != 2 or latent.shape[1] != LATENT_DIM:
            raise ValueError(
                f"CNN decoder expects shape (batch, {LATENT_DIM}); "
                f"received {tuple(latent.shape)}"
            )
        features = self.from_latent(latent).reshape(
            latent.shape[0], ENCODED_CHANNELS, ENCODED_LENGTH
        )
        reconstruction = self.convolutional_decoder(features)
        if reconstruction.shape[1:] != (INPUT_CHANNELS, INPUT_LENGTH):
            raise RuntimeError(
                "CNN decoder shape changed; expected "
                f"(batch, {INPUT_CHANNELS}, {INPUT_LENGTH}), got "
                f"{tuple(reconstruction.shape)}"
            )
        return reconstruction

    def forward(self, inputs: Tensor) -> Tensor:
        """Reconstruct one batch with an output shape identical to the input."""
        return self.decode(self.encode(inputs))

    @staticmethod
    def _validate_input(inputs: Tensor) -> None:
        if inputs.ndim != 3 or inputs.shape[1:] != (INPUT_CHANNELS, INPUT_LENGTH):
            raise ValueError(
                "CNNAutoencoder expects shape "
                f"(batch, {INPUT_CHANNELS}, {INPUT_LENGTH}); "
                f"received {tuple(inputs.shape)}"
            )


def architecture_config() -> dict[str, object]:
    """Return the exact JSON-serializable CNN architecture specification."""
    return {
        "type": "one_dimensional_convolutional_autoencoder",
        "input_shape": [INPUT_CHANNELS, INPUT_LENGTH],
        "encoder": [
            {"layer": "Conv1d", "in_channels": 1, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1, "output_shape": [8, 30]},
            {"layer": "ReLU"},
            {"layer": "MaxPool1d", "kernel_size": 2, "stride": 2, "output_shape": [8, 15]},
            {"layer": "Conv1d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1, "output_shape": [16, 15]},
            {"layer": "ReLU"},
            {"layer": "MaxPool1d", "kernel_size": 3, "stride": 3, "output_shape": [16, 5]},
            {"layer": "Flatten", "output_features": FLATTENED_DIM},
            {"layer": "Linear", "in_features": FLATTENED_DIM, "out_features": LATENT_DIM},
            {"layer": "ReLU"},
        ],
        "latent_dimension": LATENT_DIM,
        "decoder": [
            {"layer": "Linear", "in_features": LATENT_DIM, "out_features": FLATTENED_DIM},
            {"layer": "ReLU"},
            {"layer": "Reshape", "output_shape": [16, 5]},
            {"layer": "ConvTranspose1d", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 3, "padding": 0, "output_padding": 0, "output_shape": [8, 15]},
            {"layer": "ReLU"},
            {"layer": "ConvTranspose1d", "in_channels": 8, "out_channels": 1, "kernel_size": 2, "stride": 2, "padding": 0, "output_padding": 0, "output_shape": [1, 30]},
        ],
        "output_activation": "Linear",
        "shape_adjustment": "none; transposed-convolution arithmetic yields length 30 exactly",
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that receive gradients."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
