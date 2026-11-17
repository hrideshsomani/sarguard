import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    print("[1/5] Loading overall dataset targets (train + test)...")
    t0 = time.time()
    
    # Load targets from both datasets
    train_path = "data/train_aml_transactions.csv"
    test_path = "data/test_aml_transactions.csv"
    
    df_train = pd.read_csv(train_path, usecols=['is_sar_reported'])
    df_test = pd.read_csv(test_path, usecols=['is_sar_reported'])
    
    # Combine into overall tabular target series
    overall_targets = pd.concat([df_train['is_sar_reported'], df_test['is_sar_reported']], ignore_index=True)
    total_records = len(overall_targets)
    print(f"Loaded {total_records:,} overall records in {time.time() - t0:.2f}s")
    
    # Target value distribution
    legit_total = int((overall_targets == 0).sum())
    susp_total = int((overall_targets == 1).sum())
    print(f"Total Legitimate (0): {legit_total:,} ({legit_total/total_records*100:.2f}%)")
    print(f"Total Suspicious (1): {susp_total:,} ({susp_total/total_records*100:.2f}%)")
    
    # [2/5] Shuffle the overall tabular dataset randomly (no train/test split, no model)
    print("[2/5] Shuffling overall tabular dataset with random seed 42...")
    rng = np.random.default_rng(seed=42)
    shuffled_targets = rng.permutation(overall_targets.values)
    
    # [3/5] Compute cumulative curves
    print("[3/5] Computing cumulative statistics across shuffled stream...")
    # Sample every 1,000 steps for clean plotting of 2M points
    step = 1000
    indices = np.arange(step, total_records + 1, step)
    
    # Compute cumulative sums
    cum_suspicious = np.cumsum(shuffled_targets)
    cum_legitimate = np.arange(1, total_records + 1) - cum_suspicious
    
    # Sampled points
    sample_indices = indices
    sample_total = sample_indices
    sample_legit = cum_legitimate[sample_indices - 1]
    sample_susp = cum_suspicious[sample_indices - 1]
    
    # Percentages
    pct_legit = (sample_legit / sample_total) * 100.0
    pct_susp = (sample_susp / sample_total) * 100.0
    
    # Rolling 25k rate to measure uniform dispersion
    window = 25000
    rolling_legit = pd.Series(shuffled_targets == 0).rolling(window=window, step=step).mean() * 100.0
    rolling_x = np.arange(window, total_records + 1, step)
    rolling_y = rolling_legit.dropna().values[:len(rolling_x)]
    
    # [4/5] Generate 4-Panel Visualization
    print("[4/5] Plotting publication-grade curves...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.suptitle(
        f"Overall AML Tabular Dataset (Shuffled, No Train/Split) - Total Records: {total_records:,}\n"
        f"Legitimate: {legit_total:,} (50.0%) | Suspicious: {susp_total:,} (50.0%)",
        fontsize=14, fontweight='bold', y=0.98
    )
    
    # Panel 1: Total Legitimate vs Total Processed Transactions (Cumulative Absolute Count)
    ax1 = axes[0, 0]
    total_thousands = sample_total / 1000.0
    legit_thousands = sample_legit / 1000.0
    susp_thousands = sample_susp / 1000.0
    
    ax1.plot(total_thousands, total_thousands, label="Total Records Reference (y = x)", color='#64748b', linestyle=':', linewidth=1.8)
    ax1.plot(total_thousands, legit_thousands, label="Cumulative Legitimate (Target = 0)", color='#1d4ed8', linewidth=2.5)
    ax1.plot(total_thousands, susp_thousands, label="Cumulative Suspicious (Target = 1)", color='#dc2626', linestyle='--', linewidth=2.0)
    
    # Annotate end point
    ax1.scatter([total_thousands[-1]], [legit_thousands[-1]], color='#1d4ed8', s=60, zorder=5)
    ax1.annotate(
        f"Legitimate = {legit_total:,}\n(50.0% of Total)",
        xy=(total_thousands[-1], legit_thousands[-1]),
        xytext=(total_thousands[-1] - 550, legit_thousands[-1] + 120),
        arrowprops=dict(facecolor='#1d4ed8', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold', fontsize=9, color='#1e3a8a',
        bbox=dict(boxstyle="round,pad=0.3", fc="#dbeafe", ec="#93c5fd", lw=1)
    )
    
    ax1.set_title("1. Total Legitimate vs. Total Transactions (Cumulative Counts)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Total Transactions Processed (Thousands)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Cumulative Count (Thousands)", fontsize=10, fontweight='bold')
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xlim(0, total_thousands[-1] + 20)
    ax1.set_ylim(0, total_thousands[-1] + 20)
    
    # Panel 2: Legitimate Proportion Curve (% Legitimate vs Total Transactions)
    ax2 = axes[0, 1]
    ax2.plot(total_thousands, pct_legit, color='#2563eb', linewidth=2.2, label="% Legitimate in Shuffled Table")
    ax2.axhline(50.0, color='#16a34a', linestyle='--', linewidth=2.0, label="Exact 50.0% Parity Line")
    
    # Convergence point annotation
    mid_idx = len(total_thousands) // 2
    ax2.scatter([total_thousands[-1]], [pct_legit[-1]], color='#16a34a', s=70, zorder=5)
    ax2.annotate(
        f"Final Rate: {pct_legit[-1]:.4f}%\nDelta from 50%: {abs(pct_legit[-1]-50.0):.4f}%",
        xy=(total_thousands[-1], pct_legit[-1]),
        xytext=(total_thousands[-1] - 650, pct_legit[-1] + 0.35),
        arrowprops=dict(facecolor='#16a34a', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold', fontsize=9, color='#14532d',
        bbox=dict(boxstyle="round,pad=0.3", fc="#dcfce7", ec="#86efac", lw=1)
    )
    
    ax2.set_title("2. % Legitimate vs. Total Processed (Law of Large Numbers)", fontsize=11, fontweight='bold')
    ax2.set_xlabel("Total Transactions Processed (Thousands)", fontsize=10, fontweight='bold')
    ax2.set_ylabel("% Legitimate Transactions (%)", fontsize=10, fontweight='bold')
    ax2.set_ylim(48.5, 51.5)
    ax2.legend(loc="lower left", frameon=True, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_xlim(0, total_thousands[-1] + 20)
    
    # Panel 3: Parametric Curve - Total Legitimate vs Total Suspicious (1:1 Symmetry)
    ax3 = axes[1, 0]
    ax3.plot([0, legit_thousands[-1]], [0, legit_thousands[-1]], color='#94a3b8', linestyle=':', linewidth=2, label="Ideal 1:1 Symmetry Line")
    ax3.plot(legit_thousands, susp_thousands, color='#7c3aed', linewidth=2.5, label="Observed Shuffled Trajectory")
    
    ax3.scatter([legit_thousands[-1]], [susp_thousands[-1]], color='#7c3aed', s=70, zorder=5)
    ax3.annotate(
        f"({legit_total:,} Legitimate,\n {susp_total:,} Suspicious)",
        xy=(legit_thousands[-1], susp_thousands[-1]),
        xytext=(legit_thousands[-1] - 400, susp_thousands[-1] - 150),
        arrowprops=dict(facecolor='#7c3aed', shrink=0.08, width=1.5, headwidth=6),
        fontweight='bold', fontsize=9, color='#4c1d95',
        bbox=dict(boxstyle="round,pad=0.3", fc="#f3e8ff", ec="#d8b4fe", lw=1)
    )
    
    ax3.set_title("3. Parametric Symmetry Curve (Legitimate vs. Suspicious)", fontsize=11, fontweight='bold')
    ax3.set_xlabel("Total Legitimate Transactions (Thousands)", fontsize=10, fontweight='bold')
    ax3.set_ylabel("Total Suspicious Transactions (Thousands)", fontsize=10, fontweight='bold')
    ax3.legend(loc="upper left", frameon=True, fontsize=9)
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.set_xlim(0, legit_thousands[-1] + 30)
    ax3.set_ylim(0, susp_thousands[-1] + 30)
    
    # Panel 4: Shuffled Uniformity Check (Rolling 25,000 Window Legitimate Rate)
    ax4 = axes[1, 1]
    rx_thousands = rolling_x / 1000.0
    ax4.plot(rx_thousands, rolling_y, color='#0891b2', linewidth=1.5, label="Rolling 25,000-Txn Legitimate Rate (%)")
    ax4.axhline(50.0, color='#dc2626', linestyle='--', linewidth=1.8, label="50.0% Expected Parity")
    ax4.axhspan(49.0, 51.0, color='#0891b2', alpha=0.1, label="±1.0% Normal Dispersion Band")
    
    ax4.set_title("4. Shuffled Dispersion Uniformity (Rolling 25k Window Rate)", fontsize=11, fontweight='bold')
    ax4.set_xlabel("Position in Shuffled Dataset (Thousands)", fontsize=10, fontweight='bold')
    ax4.set_ylabel("Window Legitimate Rate (%)", fontsize=10, fontweight='bold')
    ax4.set_ylim(47.5, 52.5)
    ax4.legend(loc="lower right", frameon=True, fontsize=9)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.set_xlim(0, total_thousands[-1] + 20)
    
    # [5/5] Save figures
    artifact_dir = "C:/Users/DELL/.gemini/antigravity/brain/5018f49a-2a48-4d56-afe2-576e4d944332"
    out_artifact = os.path.join(artifact_dir, "overall_shuffled_legitimate_vs_total_curves.png")
    out_local = "overall_shuffled_legitimate_vs_total_curves.png"
    
    plt.savefig(out_artifact, bbox_inches='tight')
    plt.savefig(out_local, bbox_inches='tight')
    plt.close()
    
    print(f"Saved figure to:\n- {out_artifact}\n- {out_local}")
    print("[SUCCESS] All curves generated successfully without train or split!")

if __name__ == '__main__':
    main()
