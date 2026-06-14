"""
src/analysis/lstm_dataset.py
────────────────────────────
Dataset utilities for LSTM training.
Handles CSV loading, normalization, and windowed sequence generation.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List

class TrafficDataset(Dataset):
    """
    PyTorch Dataset for traffic time-series data.
    """
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TimeSeriesGenerator:
    """
    Utility to convert CSV vehicle counts into windowed sequences.
    """
    def __init__(self, sequence_length: int = 30, prediction_horizon: int = 1):
        self.seq_len = sequence_length
        self.horizon = prediction_horizon
        self.scaler = MinMaxScaler()

    def load_and_preprocess(self, csv_path: str) -> np.ndarray:
        """Load CSV and scale vehicle counts."""
        df = pd.read_csv(csv_path)
        if "n_vehicles" not in df.columns:
            raise ValueError(f"CSV at {csv_path} must contain 'n_vehicles' column.")
        
        data = df["n_vehicles"].values.reshape(-1, 1).astype(float)
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate (X, y) windows.
        X shape: (N, seq_len, 1)
        y shape: (N, horizon)
        """
        X, y = [], []
        for i in range(len(data) - self.seq_len - self.horizon + 1):
            X.append(data[i : i + self.seq_len])
            y.append(data[i + self.seq_len : i + self.seq_len + self.horizon].flatten())
        
        return np.array(X), np.array(y)

    def get_dataloaders(
        self, 
        csv_path: str, 
        batch_size: int = 32, 
        train_split: float = 0.8
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Full pipeline to get train and validation DataLoaders.
        """
        data = self.load_and_preprocess(csv_path)
        X, y = self.create_sequences(data)

        split_idx = int(len(X) * train_split)
        
        train_dataset = TrafficDataset(X[:split_idx], y[:split_idx])
        val_dataset   = TrafficDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert scaled values back to original vehicle counts."""
        return self.scaler.inverse_transform(data)
