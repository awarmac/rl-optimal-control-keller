from spinup.algos.pytorch.vpg.core import MLPActorCritic, MLPCategoricalActor
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
import os
import sys

import environment_gym

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)    
    parser.add_argument('--num-episodes', type=int, default=5)
    parser.add_argument('--log-raw-csv', action='store_true')
    parser.add_argument('--track-length', type=float, default=100)
    parser.add_argument('--time-limit', type=float, default=20)
    args = parser.parse_args()
    
    print(" ".join(sys.argv))
    print(args)

    model_dir = os.path.dirname(os.path.abspath(args.model_path))
    log_dir = os.path.join(model_dir, "log-{}".format(dt.now().strftime("%Y-%m-%d_%H-%M-%S")))

    env = environment_gym.Env(log_dir=log_dir, 
                        track_length=args.track_length,
                        time_limit=args.time_limit, delta_time=0.1)
    env.enable_log_raw_csv(1)

    # ac = MLPActorCritic(env.observation_space, env.action_space, [32])
    # ac = ac.load_state_dict(torch.load(args.model_path))
    
    ac = torch.load(args.model_path)
        
    for i in range(args.num_episodes):  
        done = False
        state = env.reset()
        while True:
            # action, _, _ = ac.step(torch.as_tensor(state, dtype=torch.float32)) # Categorical(logits=logits).sample().item()
            obs = torch.as_tensor(state, dtype=torch.float32)
            action = ac.pi._distribution(obs).mean
            # print(action.detach().numpy())
            # ac.step() # Categorical(logits=logits).sample().item()
            state, reward, done, succ = env.step(action.detach().numpy())
            if done:
                state = env.reset()
                break
        
        print("Episode Ended")
    from turn_csv_to_fig import turn_csv_to_fig
    turn_csv_to_fig(log_dir, num_last_files=args.num_episodes)