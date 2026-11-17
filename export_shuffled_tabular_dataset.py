"""
Utility: Export and Inspect Shuffled Overall AML Tabular Dataset
Usage:
    python export_shuffled_tabular_dataset.py --sample 5
    python export_shuffled_tabular_dataset.py --save data/overall_aml_shuffled.csv
"""

import os
import argparse
import pandas as pd
import numpy as np

def get_overall_shuffled_df(sample_n=None, save_path=None, random_state=42):
    print(f"Loading overall dataset (train + test)...")
    train_path = "data/train_aml_transactions.csv"
    test_path = "data/test_aml_transactions.csv"
    
    # Read both datasets
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Concatenate without split
    overall_df = pd.concat([df_train, df_test], ignore_index=True)
    print(f"Total raw records combined: {len(overall_df):,}")
    
    # Shuffle entire tabular dataset
    print(f"Shuffling tabular data uniformly across all {len(overall_df):,} rows (seed={random_state})...")
    shuffled_df = overall_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    
    legit_cnt = (shuffled_df['is_sar_reported'] == 0).sum()
    susp_cnt = (shuffled_df['is_sar_reported'] == 1).sum()
    
    print("\n--- SHUFFLED TABULAR DATASET SUMMARY ---")
    print(f"Total Records: {len(shuffled_df):,}")
    print(f"Total Legitimate (0): {legit_cnt:,} ({legit_cnt / len(shuffled_df) * 100:.2f}%)")
    print(f"Total Suspicious (1): {susp_cnt:,} ({susp_cnt / len(shuffled_df) * 100:.2f}%)")
    print(f"Ratio (Legitimate : Suspicious): {legit_cnt / susp_cnt:.4f} : 1.0000")
    print("----------------------------------------\n")
    
    if save_path:
        print(f"Saving shuffled dataset to {save_path}...")
        shuffled_df.to_csv(save_path, index=False)
        print(f"Successfully saved to {save_path}!")
        
    if sample_n:
        print(f"Displaying first {sample_n} rows of shuffled tabular data with target:")
        cols_to_show = ['transaction_id', 'amount', 'transaction_type', 'channel', 'aml_typology', 'is_sar_reported']
        cols = [c for c in cols_to_show if c in shuffled_df.columns]
        print(shuffled_df[cols].head(sample_n).to_string(index=True))
        
    return shuffled_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Shuffle overall AML tabular dataset with target.")
    parser.add_argument("--sample", type=int, default=5, help="Number of sample rows to print")
    parser.add_argument("--save", type=str, default=None, help="File path to save the full shuffled CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random shuffle seed")
    args = parser.parse_args()
    
    get_overall_shuffled_df(sample_n=args.sample, save_path=args.save, random_state=args.seed)
