# 🚦 Bangalore Traffic Analysis & Congestion Mapping

An autonomous computer vision and deep learning pipeline for monitoring urban traffic density, identifying congestion hotspots, and forecasting future traffic volume. This project utilizes a fine-tuned YOLOv8 model for detection and a PyTorch LSTM for temporal forecasting.

---

## 🏗️ Project Architecture

The pipeline is organized into six modular layers for robust traffic analysis:

1.  **Detection Layer:** Uses YOLOv8 (fine-tuned on VisDrone) to detect cars, vans, trucks, and buses in aerial imagery.
2.  **Density Estimation:** Implements Kernel Density Estimation (KDE) to create smooth heatmaps of vehicle concentration.
3.  **Congestion Mapping:** A grid-based analysis system that classifies road segments into levels: *Free, Moderate, Heavy, or Severe*.
4.  **Forecasting Layer (New):** A PyTorch LSTM model that analyzes historical vehicle counts to predict traffic volume for future frames.
5.  **Visualization:** Composites detection boxes, density heatmaps, forecasting HUD, and real-time statistics into a high-quality output video.
6.  **Analytics & Logging:** Generates cumulative hotspot maps, trend reports, and per-frame CSV logs for longitudinal studies.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- PyTorch (with CUDA support recommended for training)
- Windows or Linux

### 2. Setup (Virtual Environment)
```bash
# Clone the repository
git clone https://github.com/Astitwa73/Bangalore-Traffic-Analysis-Congestion-Mapping.git
cd Bangalore-Traffic-Analysis-Congestion-Mapping

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Forecasting Pipeline (LSTM)

The forecasting module allows the system to predict future traffic trends based on recent history.

### 1. Extract Training Data
Extract vehicle counts from a video to create a time-series dataset:
```powershell
python src/analysis/extract_counts.py --video data/satellite.mp4
```

### 2. Train the LSTM
Train the model to forecast the next frame based on the last 30 frames:
```powershell
python src/analysis/train_lstm.py --csv data/counts_satellite.csv --epochs 100 --seq_len 30
```

### 3. Evaluate Results
Generate MAE/RMSE metrics and loss plots:
```powershell
python src/analysis/evaluate_lstm.py --csv data/counts_satellite.csv --model lstm_congestion.pt
```

---

## 📊 Usage (run_pipeline.py)

The `run_pipeline.py` script is the core engine. 

### Inference with Forecasting
Run detection + congestion analysis + LSTM forecasting simultaneously:
```powershell
python run_pipeline.py --mode infer --video data/satellite.mp4 --lstm lstm_congestion.pt --preview
```

### CLI Arguments Reference
| Argument | Description |
| :--- | :--- |
| `--mode` | **Required.** Choices: `preprocess`, `train`, `infer`, `all`. |
| `--video` | Path to input video (required for `infer` and `all`). |
| `--weights`| Path to YOLO weights (default: `best.pt`). |
| `--lstm`   | **(New)** Path to trained LSTM weights (e.g., `lstm_congestion.pt`). |
| `--preview`| Flag to show the live analysis window. |
| `--output` | Custom directory for results (default: `outputs/`). |

---

## 📁 Output Artifacts

After processing, check the `outputs/` directory for:
- `[video_name]_output.mp4`: Final annotated video with **Forecast HUD**.
- `[video_name]_trends.png`: Multi-panel trend analysis (Volume & Congestion).
- `[video_name]_hotspots.png`: Cumulative "Hotspot Map" of congested areas.
- `evaluation/`: Contains LSTM training history plots and prediction samples.

---

## ⚙️ Configuration
Fine-tune parameters in `config.yaml`:
- **Congestion:** Adjust `thresholds` (vehicles per cell) for level classification.
- **Density:** Change `bandwidth` and `heatmap_alpha` for visualization.
- **Inference:** Set `conf_threshold` and `iou_threshold` for the YOLO detector.
