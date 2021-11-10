import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm     
from functools import cmp_to_key
import traceback

left  = 1.5  # the left side of the subplots of the figure
right = 2    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 2      # the top of the subplots of the figure
wspace = 0.5   # the amount of width reserved for blank space between subplots
hspace = 0.5   # the amount of height reserved for white space between subplots

def read_info(log_dir):
    with open(os.path.join(log_dir, "info.txt"), mode="r") as f:
        names, vals = f.readlines()
        #print(names, vals)
        vals = vals.split(",")
        #print(vals)
        delta_time, track_length, time_limit, mass, mu, capacity, max_current, voltage_0, state_of_charge_0 = (float(v) for v in vals)
        return delta_time, track_length, time_limit, mass, mu, capacity, max_current, voltage_0, state_of_charge_0
                    
def stepize_column(df, column):
    df[column].loc[0] = df[column].loc[1]
    times = []
    vals = []
    for i in range(1, len(df[column])):
        times.append(df["time"].loc[i-1])
        vals.append(float(df[column].loc[i]))
        times.append(df["time"].loc[i])
        vals.append(float(df[column].loc[i]))
    return times, vals

def plot_stepized_column(i, j, mtp_axs ,df, column):
    times, actions = stepize_column(df, column)
    mtp_axs[i][j].plot(times, actions)
    mtp_axs[i][j].set(ylabel=column)
    mtp_axs[i][j].grid(True)

def turn_csv_to_fig(log_dir, num_last_files=10, go_back_percentage=0.0, 
                    y_limits=None, time_limit=None, ):
    csv_dir = os.path.join(log_dir, "csv")
    fig_dir = os.path.join(log_dir, "fig")

    os.makedirs(fig_dir, exist_ok=True)

    csv_files = os.listdir(csv_dir)
    csv_files = sorted(csv_files, 
                    key=cmp_to_key(lambda x, y: int(x.split("_")[1].split(".")[0]) - int(y.split("_")[1].split(".")[0])),
                    reverse=True
                )
    start_index = int(len(csv_files)*go_back_percentage)
    print(start_index)
    csv_files = csv_files[(start_index):(start_index+num_last_files*2)]
    for i,fname in enumerate(csv_files):
        if fname.endswith(".csv"):
            csv_file = os.path.join(csv_dir, fname)
            info_file = os.path.join(csv_dir, os.path.basename(fname)[:-4]+"_info.txt")
                
            fig_file_path = os.path.join(fig_dir, os.path.basename(csv_file)[:-4] + ".jpg")
            if os.path.exists(fig_file_path):
                continue
            try:
                if i > num_last_files*2:
                    break
                print("{}/{} {}".format(i, len(csv_files), fname))
                
                with open(info_file, mode="r") as f:
                    lines = f.readlines()
                    print(lines, len(lines))
                    info = "    ".join(lines)
                info = info.replace("\n", " ")
                
                df = pd.read_csv (csv_file)
                # print(df)
                
                
                keys = [["x", "velocity", "E"], 
                    ["propulsion_force", "acceleration", "reward"]]
                y_limits = None
                if y_limits is not None:
                    y_limits = []
                    for lim in y_limits.split(","):
                        x = [int(x) for x in lim.split(":")]
                        y_limits.append(x)

                fig, mtp_axs = plt.subplots(nrows=3, ncols=2, figsize=(14,10), sharex=True)
                for j in range(len(keys)):
                    for i,k in enumerate(keys[j]):
                        mtp_axs[i][j].plot(df["time"], df[k])
                        mtp_axs[i][j].set(ylabel=k)
                        mtp_axs[i][j].grid(True)
                        if y_limits is not None:
                            mtp_axs[i][j].set_ylim(y_limits[j*3+i])
                        if time_limit is not None:
                            mtp_axs[i][j].set_xlim([-1, time_limit])
                # plot_stepized_column(-1, 1, mtp_axs, df, "propulsion_force")
                # plot_stepized_column(-2, 1, mtp_axs, df, "acceleration")
                
                # mtp_axs[0].set_title(info, color=('green' if "Success" in info else 'red'))        
                fig.suptitle("info:" + info, color=('green' if "Success" in info else 'red'))        

                plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                fig.tight_layout()
                
                plt.savefig(fig_file_path)
                plt.close(fig)
            except Exception as e:
                print(e)
                traceback.print_exc() 

       
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str)
    parser.add_argument('--num-last-files', type=int, default=10)
    parser.add_argument('--go-back-percentage', type=float, default=0.0)
    parser.add_argument('--track-length', type=float, default=100)
    parser.add_argument('--time-limit', type=float, default=None)
    parser.add_argument('--y-limits', type=str, default=None)
    args = parser.parse_args()
    print(args)

    turn_csv_to_fig(**args)