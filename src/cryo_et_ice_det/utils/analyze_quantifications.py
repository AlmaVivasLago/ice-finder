import os
from argparse import ArgumentParser
from datetime import date

import numpy as np
import pandas as pd

def analyze_quantifications(csv_path):
    data = pd.read_csv(csv_path)

    global_mean = np.mean(data['perc_area'])
    global_std = np.std(data['perc_area'])
    global_min = np.min(data['perc_area'])
    global_max = np.max(data['perc_area'])
    global_conf_int_95 = 1.96 * global_std / np.sqrt(len(data['perc_area']))

    print("=" * 40)
    print("Global Quantification Statistics")
    print("-" * 40)
    print(f"Mean: {global_mean:.2f}%, 95% CI: ±{global_conf_int_95:.2f}%")
    print(f"Min: {global_min:.2f}%")
    print(f"Max: {global_max:.2f}%")
    print()

    print("=" * 40)
    print("Per Tilt Series Statistics")
    print("-" * 40)
    ts_groups = data.groupby('ts_id')
    stats = []
    for ts_id, group in ts_groups:
        ts_mean = np.mean(group['perc_area'])
        ts_std = np.std(group['perc_area'])
        ts_min = np.min(group['perc_area'])
        ts_max = np.max(group['perc_area'])
        ts_conf_int_95 = 1.96 * ts_std / np.sqrt(len(group['perc_area']))

        stats.append([ts_id, ts_mean, ts_min, ts_max, ts_conf_int_95])
        print(f"Tilt Series {ts_id}:")
        print(f"  Mean: {ts_mean:.2f}%, 95% CI: ±{ts_conf_int_95:.2f}%")
        print(f"  Min: {ts_min:.2f}%")
        print(f"  Max: {ts_max:.2f}%")
        print()

    summary_csv = os.path.join(os.path.dirname(csv_path), "quantification_summary.csv")
    summary_df = pd.DataFrame(stats, columns=['ts_id', 'mean', 'min', 'max', '95_conf_int'])
    summary_df.to_csv(summary_csv, index=False)

    print("=" * 40)
    print(f"Summary saved to: {os.path.abspath(summary_csv)}")
    print("=" * 40)


def filter_micrographs(csv_path, threshold):
    data = pd.read_csv(csv_path)
    
    data['filtered'] = data['perc_area'] >= threshold
    
    filtered_csv = os.path.join(os.path.dirname(csv_path), "filtered_micrographs.csv")
    filtered_data = data[['id', 'file_name', 'filtered']]
    filtered_data.to_csv(filtered_csv, index=False)
    
    print("=" * 40)
    print(f"Filtering Threshold: {threshold}%")
    print("=" * 40)
    print(f"Filtered Micrographs Saved To:\n{os.path.abspath(filtered_csv)}")
    print("-" * 40)
    print(f"Micrographs Passing Filter: {filtered_data['filtered'].sum()} / {len(filtered_data)}")
    print("=" * 40)

if __name__ == "__main__":
    parser = ArgumentParser(description="Analyze and filter quantification results from predictions.csv.")
    parser.add_argument('--date', type=str, default=date.today().strftime("%d-%m-%Y"), help="Date for the inference results (format: dd-mm-yyyy). Default is today's date.")
    parser.add_argument('--filter_threshold', type=float, default=None, help="Filter threshold for perc_area (e.g., 5.0).")
    args = parser.parse_args()
    
    csv_path = os.path.join('inference-results', args.date, 'quantification.csv')
    
    analyze_quantifications(csv_path)
    
    if args.filter_threshold is not None:
        filter_micrographs(csv_path, args.filter_threshold)

