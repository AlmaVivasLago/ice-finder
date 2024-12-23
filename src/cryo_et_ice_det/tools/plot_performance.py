from argparse import ArgumentParser
from datetime import date
import os

import numpy as np
import json

import matplotlib.pyplot as plt
from matplotlib import font_manager
from IPython.display import display, FileLink

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def read_json(fpath):
    with open(fpath, "r") as fid:
        data = json.load(fid)
    return data

def compute_statistics(res):
    toret = {}
    num_trials = 0
    for model_name, shots in res.items():
        toret[model_name] = {}
        for shot, values in shots.items():
            num_trials = len(values)
            mean = np.mean(values)
            std = np.std(values)
            conf_int_95 = 1.96 * std / np.sqrt(num_trials)
            toret[model_name][shot] = {"mean": mean, "95_conf_int": conf_int_95}
    return toret

def plot_performance(statistics_iou, statistics_f1, output=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), dpi=500)
    plt.rc('font', family='DejaVu Serif', size=12)
    colors = {
        'maml': '#1f77b4',
        'baseline': '#333333',
        'random': '#d62728'
    }

    def plot_data(ax, statistics, metric):
        best_performance = {'mean': -np.inf, 'model': '', 'shot': ''}
        for model_name, shots in statistics.items():
            elements = [val["mean"] for val in shots.values()]
            ax.plot(elements, label=model_name, marker="o", zorder=10, clip_on=False, alpha=.6, color=colors.get(model_name, '#333333'))

            for idx, val in enumerate(elements):
                if val > best_performance['mean']:
                    best_performance = {'mean': val, 'model': model_name, 'shot': f'K{idx}'}
        # Highlight the best performance
        best_idx = int(best_performance['shot'][1:])  # Extract the shot number from the shot key
        ax.scatter(best_idx, best_performance['mean'], color='black', marker='*', s=100, label=f"Best: {best_performance['model']} {best_performance['shot']}")

        ax.set_xlim(left=0, right=len(shots)-1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticks(range(len(shots)))
        ax.set_ylabel(metric.capitalize())
        ax.set_xlabel("Shots")
        ax.legend()
        ax.grid(which='both', linestyle='--', linewidth=0.5)

    plot_data(ax1, statistics_iou, 'iou')
    plot_data(ax2, statistics_f1, 'f1')

    plt.savefig(output)
    print(f"Plot saved to:")
    display(FileLink(output))


def main(args):
    # Determine the subfolder based on today's date or the provided date
    results_subfolder = date.today().strftime("%d-%m-%Y") if args.date is None else args.date
    results_folder = os.path.join('k-shot-results', results_subfolder)
    output_path = os.path.join(results_folder, f'k_shot_performance_plot_{results_subfolder}.png')

    # Paths to the JSON files
    iou_json_path = os.path.join(results_folder, 'test_iou.json')
    f1_json_path = os.path.join(results_folder, 'test_f1.json')

    results_iou = read_json(iou_json_path)
    results_f1 = read_json(f1_json_path)
    stats_iou = compute_statistics(results_iou)
    stats_f1 = compute_statistics(results_f1)

    plot_performance(stats_iou, stats_f1, output_path)

if __name__ == "__main__":
    parser = ArgumentParser(description="Compare model performance across different shots for both IOU and F1 metrics.")
    parser.add_argument('--date', type=str, default=None, help="Date for accessing result subfolders (format: dd-mm-yyyy). Use today's date if not provided.")
    args = parser.parse_args()
    main(args)