import random
import matplotlib.pyplot as plt
import argparse
from environment_gym import Env
from datetime import datetime as dt
import os
import pickle

def load_env(log_dir):
    file_path = os.path.join(log_dir, "last_env.pckl.obj")
    with open(file_path, mode="rb") as f:
        env = pickle.load(f)
        return env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-env', type=str, default=None)
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--log-raw-csv', action='store_true')
    parser.add_argument('--log-dir', type=str, default="log-{}".format(dt.now().strftime("%Y-%m-%d_%H-%M-%S")))
    parser.add_argument('--track-length', type=float, default=800)
    parser.add_argument('--time-limit', type=float, default=20)
    parser.add_argument('--random-actions', type=str, default="random-actions.txt")
    args = parser.parse_args()
    print(args)

    mass = 1
    if args.load_env is not None:
        env = load_env(args.log_dir)
    else:
        env = Env(mass=mass, log_dir=args.log_dir, track_length=args.track_length, 
                    time_limit=args.time_limit, delta_time=0.01)
    env.enable_log_raw_csv(1)
    random_actions = []
    with open(args.random_actions, mode='r', encoding='utf-8') as f:
        for l in f:
            random_actions.append(float(l))

    for i in range(args.num_episodes):
        done = False
        # print(env.get_state())
        t = 0
        energy = 500
        while True:
            # import pdb; pdb.set_trace()
            action = 12.2 * mass # if energy > 10 else 0 # random_actions[t]# 10 # [random.randint(0,13)] # i # 17 if t < 10 else 9 if t < 30-i else 5 # 
            #print(action)
            state, reward, done, succ = env.step([action])
            energy = state[2]
            #print(state, reward, done, succ)
            t += 1
            if done:
                env.reset()
                break
        print("Episode Ended")
    print("Logs in: {}".format(args.log_dir))
