"""
src/analysis/lstm_model.py
──────────────────────────
PyTorch LSTM model for traffic congestion forecasting.
Predicts future vehicle counts based on a historical sequence.
"""

import torch
import torch.nn as nn

class TrafficLSTM(nn.Module):
    """
    LSTM model for time-series forecasting of vehicle counts.

    Parameters
    ----------
    input_size : int
        Number of features per time step (e.g., 1 for vehicle count).
    hidden_size : int
        Number of hidden units in LSTM.
    num_layers : int
        Number of stacked LSTM layers.
    output_size : int
        Number of time steps to predict into the future (horizon).
    dropout : float
        Dropout probability between layers.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2
    ):
        super(TrafficLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        x shape: (batch_size, seq_length, input_size)
        """
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
