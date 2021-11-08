
from numpy.lib.function_base import bartlett
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
import pandas as pd
import environment_gym
import os
import sys
import pickle
import argparse

from spinup.utils.mpi_tools import mpi_fork
import spinup.algos.pytorch.ddpg.core as core
from spinup.algos.pytorch.ddpg.ddpg import ddpg
  
def load_env(log_dir):
    file_path = os.path.join(log_dir, "last_env.pckl.obj")
    with open(file_path, mode="rb") as f:
        env = pickle.load(f)
        return env

def get_env(args):
    if os.path.exists(args.log_dir) and not args.restore_log_dir:
        print("log directory exists but restore-log-dir is not enabled!".format())
        # restore_log_dir = True
    exp_dir = os.path.join(args.log_dir, args.exp_name)
    # make environment, check spaces, get obs / act dims
    if args.restore_log_dir:
        env = load_env(exp_dir)
    else:
        env = environment_gym.Env(log_dir=exp_dir, delta_time=0.1,
                            track_length=args.track_length, 
                            time_limit=args.time_limit)
        if args.log_raw_csv:
            env.enable_log_raw_csv(args.log_raw_csv_every)
    
    return env

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--render-matplot', action='store_true')
    parser.add_argument('--log-raw-csv', action='store_true')
    parser.add_argument('--log-raw-csv-every', type=int, default=500, help="log once every n episodes")
    parser.add_argument('--log-dir', type=str, default="log-{}".format(dt.now().strftime("%Y-%m-%d_%H-%M-%S")))
    parser.add_argument('--exp-name', type=str, default="exp")
    parser.add_argument('--hidden-sizes', type=str, default="32")
    parser.add_argument('--restore-log-dir', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--track-length', type=float, default=100)
    parser.add_argument('--time-limit', type=float, default=20)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--cpu', type=int, default=1)
    args = parser.parse_args()
    print(" ".join(sys.argv))
    print(args)
    print('\nUsing reward-to-go formulation of policy gradient.\n')


    # mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=args.log_dir)

    ddpg(lambda : get_env(args), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[int(x) for x in args.hidden_sizes.split(",")], activation=nn.Tanh), 
        gamma=args.gamma, seed=args.seed, steps_per_epoch=args.batch_size, epochs=args.epochs,
        pi_lr=args.lr, logger_kwargs=logger_kwargs)
