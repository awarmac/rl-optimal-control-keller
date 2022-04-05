import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch import preserve_format
from tqdm import tqdm     
from functools import cmp_to_key
import traceback
import math
import environment_gym

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

def run_kellers_result(track_length, time_step = 0.01):
    
    T_t1_t2_mapping = {
        100:    [10.07, 10.07, 0],
        200:    [19.25, 19.25, 0],
        400:    [43.27, 1.78, 0.86],
        1500:   [3*60+39.44, 0.88, 1.31],
        10000:   [26*60+54.10, 0.75, 2.12]
    }
    if track_length not in T_t1_t2_mapping.keys():
        return None
    T, t1, t2 = T_t1_t2_mapping[track_length]
    env = environment_gym.Env(log_dir="log-test-turn_csv_to_fig", delta_time=time_step,
                            track_length=track_length)
    
    v_constant = lambda t1: env.max_force*env.object.tau*(1-math.exp(-t1/env.object.tau))

    env.enable_log_raw_csv(every=1)
    data = []
    v_constant_this = v_constant(t1)
    prev_v = 0
    while env.object.velocity < v_constant_this:
        f = env.max_force
        state, reward, done, succ = env.step([f])
        x, v, e = state
        data.append([env.time, x,v,e,f,env.object.acceleration,reward])

        if v + (v - prev_v) >= v_constant_this:
            break

        prev_v = v
    print(f"1)log {track_length}: ", f"t1: {env.time}?", env.object.x, env.object.velocity, env.battery.E, done)
    print(f"1)v_constant: {v_constant_this} == {env.object.velocity} ?")
    
    # target_velocity = v_constant_this
    # f = (target_velocity - env.object.velocity)/time_step + env.object.velocity/env.object.tau
    # state, reward, done, succ = env.step([f])
    # x, v, e = state
    # # env.object.velocity = target_velocity
    # data.append([env.time, x,v,e,f,env.object.acceleration,reward])

    # print(f"2)log {track_length}: ", f"t1: {env.time}?", env.object.x, env.object.velocity, env.battery.E, done)
    # print(f"2)v_constant: {v_constant_this} == {env.object.velocity} ?")

    while env.battery.E > 0: #  or env.time < T - t2:
        f = env.object.velocity/env.object.tau
        if env.check_step_possible([f]):
            state, reward, done, succ = env.step([f])
            x, v, e = state
            data.append([env.time, x,v,e,f,env.object.acceleration,reward])
        else:
            print("Breaking because next step is not possible at t: {}".format(env.time))
            break
    print(f"3)log {track_length}: ", env.time, env.object.x, env.object.velocity, env.battery.E, done)
    print(f"3)t2: {T-env.time}?")
    while not done:   #env.time <= T:
        f = env.battery.sigma/env.object.velocity
        state, reward, done, succ = env.step([f])
        x, v, e = state
        data.append([env.time, x,v,e,f,env.object.acceleration,reward])
    print(f"4)log {track_length}: ", env.time, env.object.x, env.object.velocity, env.battery.E, done)
    data = pd.DataFrame(data, columns=["time", "x", "velocity", "E", "propulsion_force", "acceleration", "reward"])
    return data

def run_kellers_result_new(track_length, time_step = 0.01):
    T_t1_t2_mapping = {
        100:    [10.07, 10.07, 0],
        200:    [19.25, 19.25, 0],
        400:    [43.27, 1.78, 0.86],
        1500:   [3*60+39.44, 0.88, 1.31],
        10000:   [26*60+54.10, 0.75, 2.12]
    }
    if track_length not in T_t1_t2_mapping.keys():
        return None
    T, t1, t2 = T_t1_t2_mapping[track_length]
    env = environment_gym.Env(log_dir="log-test-turn_csv_to_fig", delta_time=time_step,
                            track_length=track_length)
    
    v_t1 = lambda tx: env.max_force*env.object.tau*(1-math.exp(-tx/env.object.tau))
    x_t1 = lambda tx: env.max_force*env.object.tau*env.object.tau*(tx/env.object.tau + math.exp(-tx/env.object.tau) - 1)
    e_t1 = lambda tx: env.battery.E_0 + \
                        env.battery.sigma*tx - \
                        env.max_force*env.max_force*env.object.tau*env.object.tau*\
                            (tx/env.object.tau + math.exp(-tx/env.object.tau) - 1)
    data = []
    lmd = math.sqrt(env.object.tau/env.battery.sigma) * \
            math.sqrt(1-4*math.exp(-2*(t2)/env.object.tau))
    for t in np.arange(time_step, t1+time_step, time_step):
        f = env.max_force
        v = v_t1(t)
        x = x_t1(t)
        a = f - v/env.object.tau
        e = e_t1(t)

        env.time = t 
        env.object.x = x
        data.append([t, x,v,e,f,a,env.rwd_fn_log_barrier_derivative()])

    print(f"v(t1): {v} == {env.object.tau/lmd}?")
    v = env.object.tau/lmd
    
    e0 = e
    e_t12 = lambda tx,fx,vx: (env.battery.sigma - fx*vx)*(tx-t1) + e0
    # e_t12 = lambda t,fx,vx: (env.battery.sigma - fx*vx)*(t-T+t2)
    for t in np.arange(t1+time_step, T-t2, time_step):
        f = v/env.object.tau
        x = x + v * time_step
        a = 0
        e = e_t12(t,f,v)

        env.time = t 
        env.object.x = x
        data.append([t, x,v,e,f,a,env.rwd_fn_log_barrier_derivative()])
    
    print(f"e(t2): {e}")
    v0 = v
    v_t2 = lambda tx: math.sqrt( \
                        env.battery.sigma*env.object.tau + \
                        (v0*v0-env.battery.sigma*env.object.tau)*math.exp(2*(T-t2-tx)/env.object.tau) \
                    )
    print(f"v0: {v0}")
    print(f"vf: {v_t2(T-t2)}")
    for t in np.arange(T-t2, T, time_step):
        v = v_t2(t)
        f = env.battery.sigma / v
        x = x + v * time_step
        a = f - v/env.object.tau
        e = 0

        env.time = t 
        env.object.x = x
        data.append([t, x,v,e,f,a,env.rwd_fn_log_barrier_derivative()])

    data = pd.DataFrame(data, columns=["time", "x", "velocity", "E", "propulsion_force", "acceleration", "reward"])
    return data

def turn_csv_to_fig(log_dir, num_last_files=10, go_back_percentage=0.0, 
                    y_limits=None, time_limit=None, 
                    plot_kellers=False, plot_kellers_track_length=None):
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

    if plot_kellers:
        kellers_values_df = run_kellers_result(plot_kellers_track_length)    

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
                    # print(lines, len(lines))
                    info = "    ".join(lines)
                info = info.replace("\n", " ")
                
                df = pd.read_csv (csv_file)
                # print(df)
                
                
                keys = [["x", "velocity", "E"], 
                    ["propulsion_force", "acceleration", "reward"]]
                y_limits_array = None
                if y_limits is not None:
                    y_limits_array = []
                    for lim in y_limits.split(","):
                        x = [int(x) for x in lim.split(":")]
                        y_limits_array.append(x)

                fig, mtp_axs = plt.subplots(nrows=3, ncols=2, figsize=(14,10), sharex=True)
                for j in range(len(keys)):
                    for i,k in enumerate(keys[j]):
                        mtp_axs[i][j].plot(df["time"], df[k])    
                        mtp_axs[i][j].set(ylabel=k)
                        if plot_kellers and kellers_values_df is not None:
                            mtp_axs[i][j].plot(kellers_values_df["time"], kellers_values_df[k])

                        mtp_axs[i][j].grid(True)
                        if y_limits_array is not None:
                            mtp_axs[i][j].set_ylim(y_limits_array[j*3+i])
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
    parser.add_argument('--time-limit', type=float, default=None)
    parser.add_argument('--y-limits', type=str, default=None)
    parser.add_argument('--plot-kellers', action="store_true")
    parser.add_argument('--plot-kellers-track-length', type=int, default=None)
    args = parser.parse_args()
    print(args)

    turn_csv_to_fig(log_dir=args.log_dir, num_last_files=args.num_last_files, 
                    go_back_percentage=args.go_back_percentage, 
                    time_limit=args.time_limit, y_limits=args.y_limits,
                    plot_kellers=args.plot_kellers, plot_kellers_track_length=args.plot_kellers_track_length)