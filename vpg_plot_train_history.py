import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm     
from functools import cmp_to_key
import traceback

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str)
    args = parser.parse_args()
    print(args)
    csv_file = os.path.join(args.log_dir, "progress.txt")
    df = pd.read_csv(csv_file, sep="\t")
    keys = ["AverageEpRet", "EpLen", "AverageVVals", "LossPi", "LossV"] # df.columns
    print(keys)
    fig, mtp_axs = plt.subplots(len(keys), figsize=(14,10), sharex=True)
    print(len(mtp_axs))
    for j,k in enumerate(keys):
        mtp_axs[j].plot(df["Epoch"], df[k])
        mtp_axs[j].set(ylabel=k)
        mtp_axs[j].grid(True)
        fig.tight_layout()
        plt.savefig(os.path.join(args.log_dir, os.path.basename(csv_file)[:-4] + ".jpg"))
