"""
src/analysis/evaluate_lstm.py
────────────────────────────
Evaluation script for the Traffic LSTM model.
Calculates MAE, RMSE, MSE and generates loss plots.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys
from pathlib import Path
import argparse

# ── Project imports ────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.analysis.lstm_model import TrafficLSTM
from src.analysis.lstm_dataset import TimeSeriesGenerator

def evaluate_lstm(
    model_path: str,
    csv_path: str,
    output_dir: str = "outputs/evaluation"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load Model and Data
    # weights_only=False is required to load the MinMaxScaler object from the checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    cfg = checkpoint['config']
    scaler = checkpoint['scaler']
    history = checkpoint.get('history', {})

    gen = TimeSeriesGenerator(
        sequence_length=cfg['seq_len'], 
        prediction_horizon=cfg['horizon']
    )
    gen.scaler = scaler # Use trained scaler
    
    _, val_loader = gen.get_dataloaders(csv_path)

    model = TrafficLSTM(
        input_size=1,
        hidden_size=cfg['hidden_size'],
        num_layers=cfg['num_layers'],
        output_size=cfg['horizon']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 2. Compute Metrics
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs = seqs.to(device)
            outputs = model(seqs)
            
            # Inverse scale
            targets_orig = scaler.inverse_transform(targets.numpy().reshape(-1, 1))
            preds_orig = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, 1))
            
            all_targets.extend(targets_orig.flatten())
            all_preds.extend(preds_orig.flatten())

    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)

    print("\n── LSTM Evaluation Metrics ────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print("───────────────────────────────────────────")

    # 3. Plot Loss History
    if history:
        plt.figure(figsize=(10, 5))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('LSTM Training & Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path / "lstm_loss_plot.png")
        print(f"  Loss plot saved to: {out_path / 'lstm_loss_plot.png'}")

    # 4. Plot Actual vs Predicted (Sample)
    plt.figure(figsize=(12, 6))
    plt.plot(all_targets[:100], label='Actual', color='blue', alpha=0.7)
    plt.plot(all_preds[:100], label='Predicted', color='red', linestyle='--')
    plt.title('Actual vs Predicted Vehicle Counts (First 100 Validation Steps)')
    plt.xlabel('Time Step')
    plt.ylabel('Vehicle Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path / "lstm_prediction_sample.png")
    print(f"  Prediction sample plot saved to: {out_path / 'lstm_prediction_sample.png'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Traffic LSTM")
    parser.add_argument("--model", default="lstm_congestion.pt")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="outputs/evaluation")
    args = parser.parse_args()

    evaluate_lstm(args.model, args.csv, args.out)
