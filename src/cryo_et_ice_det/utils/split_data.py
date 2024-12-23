import os
import argparse
import pandas as pd

def split_data(source_csv_path, train_csv_path, test_csv_path, num_train_samples):
    data = pd.read_csv(source_csv_path,index_col=0)

    train_data = data.sample(n=num_train_samples, random_state=42)

    test_data = data.drop(train_data.index)

    train_data.to_csv(train_csv_path, index_label=None)
    test_data.to_csv(test_csv_path, index_label=None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training and test datasets.')
    parser.add_argument('--num_train_samples', type=int, default=5, help='Number of training samples to select.')
    args = parser.parse_args()

    source_csv = './data/selected/annotations/data.csv'
    train_csv = './data/selected/annotations/train.csv'
    test_csv = './data/selected/annotations/test.csv'

    split_data(source_csv, train_csv, test_csv, args.num_train_samples)
    print(f'Split files saved to:\n  Train: {os.path.abspath(train_csv)}\n  Test: {os.path.abspath(test_csv)}')

