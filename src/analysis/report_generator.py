"""
src/analysis/report_generator.py
────────────────────────────────
Generates professional summary reports and trend visualizations 
from the traffic analysis logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def generate_trend_report(csv_path: str, output_path: str) -> str:
    """
    Reads the frame-by-frame CSV and generates a multi-panel trend analysis.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return f"Error reading CSV: {e}"

    # Set style
    sns.set_theme(style="darkgrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. Vehicle Counts
    ax1.plot(df['timestamp_s'], df['n_vehicles'], color='#00c8ff', linewidth=2, label='Total Vehicles')
    # Fill under the curve
    ax1.fill_between(df['timestamp_s'], df['n_vehicles'], color='#00c8ff', alpha=0.2)
    
    # Optional: Breakdown by class if they aren't all zero
    if df['n_car'].sum() > 0:
        ax1.plot(df['timestamp_s'], df['n_car'], '--', color='#ffffff', alpha=0.5, label='Cars')
    
    ax1.set_title("Traffic Volume Over Time", fontsize=14, fontweight='bold', color='#222222')
    ax1.set_ylabel("Vehicle Count")
    ax1.legend(loc='upper right')
    
    # 2. Congestion Index
    # Map congestion levels to colors
    # 0-0.5 Free (Green), 0.5-1.5 Moderate (Yellow), 1.5-2.5 Heavy (Orange), 2.5+ Severe (Red)
    ax2.plot(df['timestamp_s'], df['congestion_idx'], color='#ff4b4b', linewidth=2)
    ax2.axhline(y=0.5, color='green', linestyle=':', alpha=0.5)
    ax2.axhline(y=1.5, color='orange', linestyle=':', alpha=0.5)
    ax2.axhline(y=2.5, color='red', linestyle=':', alpha=0.5)
    
    ax2.set_title("Congestion Severity Trend", fontsize=14, fontweight='bold', color='#222222')
    ax2.set_ylabel("Congestion Index (0-3)")
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylim(0, 3.2)
    
    # Add annotations for peak
    peak_idx = df['n_vehicles'].idxmax()
    peak_time = df.loc[peak_idx, 'timestamp_s']
    peak_val = df.loc[peak_idx, 'n_vehicles']
    ax1.annotate(f'Peak: {peak_val}', xy=(peak_time, peak_val), 
                 xytext=(peak_time + 5, peak_val + 2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return output_path
