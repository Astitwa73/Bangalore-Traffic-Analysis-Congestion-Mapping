"""
src/analysis/predict_lstm.py
────────────────────────────
Inference pipeline for the Traffic LSTM model.
"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import List, Union

# ── Project imports ────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.analysis.lstm_model import TrafficLSTM

class TrafficPredictor:
    """
    Handles loading a trained LSTM and making predictions on new data.
    """
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # weights_only=False is required to load the MinMaxScaler object from the checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        cfg = checkpoint['config']
        
        self.seq_len = cfg['seq_len']
        self.horizon = cfg['horizon']
        self.scaler = checkpoint['scaler']
        
        self.model = TrafficLSTM(
            input_size=1,
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            output_size=self.horizon
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, sequence: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Predict future vehicle counts based on a historical sequence.
        
        Parameters
        ----------
        sequence : List or Array of shape (seq_len,) or (seq_len, 1)
            The most recent vehicle counts.
            
        Returns
        -------
        predictions : np.ndarray of shape (horizon,)
            Predicted vehicle counts.
        """
        seq = np.array(sequence).reshape(-1, 1)
        if len(seq) < self.seq_len:
            raise ValueError(f"Sequence length must be at least {self.seq_len}")
        
        # Take the most recent window
        window = seq[-self.seq_len:]
        
        # Scale
        window_scaled = self.scaler.transform(window)
        
        # Convert to tensor (Batch, Seq, Feature)
        x = torch.FloatTensor(window_scaled).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out_scaled = self.model(x).cpu().numpy()
        
        # Inverse Scale
        # scaler expects (N, Features), out_scaled is (1, horizon)
        # We need to reshape for inverse transform if horizon > 1
        preds = self.scaler.inverse_transform(out_scaled.reshape(-1, 1))
        return preds.flatten()

if __name__ == "__main__":
    # Quick test if a model exists
    import os
    if os.path.exists("lstm_congestion.pt"):
        predictor = TrafficPredictor("lstm_congestion.pt")
        # Dummy sequence of 30 frames
        dummy_seq = [10 + i % 5 for i in range(30)]
        preds = predictor.predict(dummy_seq)
        print(f"Input Sequence (last 5): {dummy_seq[-5:]}")
        print(f"Predicted next {predictor.horizon} counts: {preds}")
    else:
        print("Model file 'lstm_congestion.pt' not found. Run training first.")
