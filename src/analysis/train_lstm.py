"""
src/analysis/train_lstm.py
──────────────────────────
Training script for the Traffic LSTM model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm

# ── Project imports ────────────────────────────────────────────────────────────
root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.analysis.lstm_model import TrafficLSTM
from src.analysis.lstm_dataset import TimeSeriesGenerator

def train_lstm(
    csv_path: str,
    config_path: str = "config.yaml",
    save_path: str = "lstm_congestion.pt",
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    seq_len: int = 30,
    horizon: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # 1. Prepare Data
    gen = TimeSeriesGenerator(sequence_length=seq_len, prediction_horizon=horizon)
    train_loader, val_loader = gen.get_dataloaders(csv_path, batch_size=batch_size)

    # 2. Initialize Model
    model = TrafficLSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=horizon
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': gen.scaler,
                'history': history,
                'config': {
                    'seq_len': seq_len,
                    'horizon': horizon,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers
                }
            }, save_path)
            # print(f"  --> Saved new best model to {save_path}")

    print(f"\n✅ Training complete. Best Val Loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Traffic LSTM")
    parser.add_argument("--csv", required=True, help="Path to vehicle counts CSV")
    parser.add_argument("--save", default="lstm_congestion.pt", help="Path to save model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    args = parser.parse_args()

    train_lstm(
        csv_path=args.csv,
        save_path=args.save,
        epochs=args.epochs,
        seq_len=args.seq_len,
        horizon=args.horizon
    )
