"""Compact recurrent Autoencoder for 30-position consumption sequences."""

from __future__ import annotations

import torch
from torch import Tensor, nn


INPUT_SIZE = 1
SEQUENCE_LENGTH = 30
ENCODER_HIDDEN_SIZE = 16
DECODER_HIDDEN_SIZE = 16
NUM_LAYERS = 1
LATENT_DIM = 8


class LSTMAutoencoder(nn.Module):
    """Reconstruct a `(B, 30, 1)` sequence through an eight-value latent state.

    The encoder's final hidden state is projected to the latent representation.
    The decoder receives that latent vector at every timestep, begins with a
    learned latent-to-hidden projection, and uses a deterministic zero cell
    state. No teacher forcing is used.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=ENCODER_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.to_latent = nn.Sequential(
            nn.Linear(ENCODER_HIDDEN_SIZE, LATENT_DIM),
            nn.ReLU(),
        )
        self.decoder_hidden = nn.Sequential(
            nn.Linear(LATENT_DIM, DECODER_HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.decoder_lstm = nn.LSTM(
            input_size=LATENT_DIM,
            hidden_size=DECODER_HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
        )
        self.output_projection = nn.Linear(DECODER_HIDDEN_SIZE, INPUT_SIZE)

    def encode(self, inputs: Tensor) -> Tensor:
        """Return the `(B, 8)` latent representation."""
        self._validate_input(inputs)
        _, (hidden, _) = self.encoder_lstm(inputs)
        final_hidden = hidden[-1]
        if final_hidden.shape != (inputs.shape[0], ENCODER_HIDDEN_SIZE):
            raise RuntimeError(
                "LSTM encoder hidden shape changed; expected "
                f"({inputs.shape[0]}, {ENCODER_HIDDEN_SIZE}), got "
                f"{tuple(final_hidden.shape)}"
            )
        return self.to_latent(final_hidden)

    def decode(self, latent: Tensor) -> Tensor:
        """Decode `(B, 8)` into `(B, 30, 1)` without teacher forcing."""
        if latent.ndim != 2 or latent.shape[1] != LATENT_DIM:
            raise ValueError(
                f"LSTM decoder expects shape (batch, {LATENT_DIM}); "
                f"received {tuple(latent.shape)}"
            )
        batch_size = latent.shape[0]
        decoder_inputs = latent.unsqueeze(1).expand(
            batch_size, SEQUENCE_LENGTH, LATENT_DIM
        )
        initial_hidden = self.decoder_hidden(latent).unsqueeze(0)
        initial_cell = torch.zeros(
            NUM_LAYERS,
            batch_size,
            DECODER_HIDDEN_SIZE,
            device=latent.device,
            dtype=latent.dtype,
        )
        decoder_outputs, _ = self.decoder_lstm(
            decoder_inputs, (initial_hidden, initial_cell)
        )
        reconstruction = self.output_projection(decoder_outputs)
        expected = (batch_size, SEQUENCE_LENGTH, INPUT_SIZE)
        if reconstruction.shape != expected:
            raise RuntimeError(
                f"LSTM decoder expected output {expected}; got "
                f"{tuple(reconstruction.shape)}"
            )
        return reconstruction

    def forward(self, inputs: Tensor) -> Tensor:
        """Reconstruct one batch with an output shape identical to the input."""
        return self.decode(self.encode(inputs))

    @staticmethod
    def _validate_input(inputs: Tensor) -> None:
        expected = (SEQUENCE_LENGTH, INPUT_SIZE)
        if inputs.ndim != 3 or inputs.shape[1:] != expected:
            raise ValueError(
                f"LSTMAutoencoder expects shape (batch, {expected[0]}, "
                f"{expected[1]}); received {tuple(inputs.shape)}"
            )


def architecture_config() -> dict[str, object]:
    """Return the exact JSON-serializable recurrent architecture."""
    return {
        "type": "lstm_autoencoder",
        "input_shape": [SEQUENCE_LENGTH, INPUT_SIZE],
        "encoder": {
            "layer": "LSTM",
            "input_size": INPUT_SIZE,
            "hidden_size": ENCODER_HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "batch_first": True,
            "sequence_summary": "final encoder hidden state",
            "output_shape": [ENCODER_HIDDEN_SIZE],
        },
        "latent_projection": {
            "layers": [
                {
                    "layer": "Linear",
                    "in_features": ENCODER_HIDDEN_SIZE,
                    "out_features": LATENT_DIM,
                },
                {"layer": "ReLU"},
            ],
            "output_shape": [LATENT_DIM],
        },
        "latent_dimension": LATENT_DIM,
        "decoder": {
            "input_strategy": "repeat latent vector for all 30 timesteps",
            "teacher_forcing": False,
            "hidden_initialization": "ReLU(Linear(latent 8 -> hidden 16))",
            "cell_initialization": "deterministic zeros",
            "lstm": {
                "input_size": LATENT_DIM,
                "hidden_size": DECODER_HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "batch_first": True,
            },
            "output_projection": {
                "layer": "Linear",
                "in_features": DECODER_HIDDEN_SIZE,
                "out_features": INPUT_SIZE,
                "activation": "none (linear)",
            },
            "output_shape": [SEQUENCE_LENGTH, INPUT_SIZE],
        },
    }


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters that receive gradients."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
