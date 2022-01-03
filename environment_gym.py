import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
import os
import math

def runge_kutta(y_current, t, delta_time, func):
    k1 = func(y_current, t)
    k2 = func(y_current + delta_time * k1 / 2, t + delta_time / 2)
    k3 = func(y_current + delta_time * k2 / 2, t + delta_time / 2)
    k4 = func(y_current + delta_time * k3, t+delta_time)
    y_next = y_current + (delta_time / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return y_next

class Battery():
    def __init__(self, 
                E_0, sigma, mass):
        self.E_0 = E_0 * mass 
        self.E = self.E_0 
        self.sigma = sigma * mass
    def get_state(self):
        return [self.E]

    def consume_rk4(self, force, velocity, delta_time):
        dE_dt_func = lambda e,t:(self.sigma - force*velocity)
        old_E = self.E
        self.E = runge_kutta(self.E, 0, delta_time, dE_dt_func)
        return self.E - old_E 

    def consume(self, force, velocity, delta_time):
        delta_E = (self.sigma - force * velocity) * delta_time
        self.E += delta_E
        return delta_E

    def reset(self):
        self.E = self.E_0

class MovingObject():
    def __init__(self, mass, tau):
        self.mass = mass
        self.tau = tau

        self.x = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0
        
    def get_state(self):
        return [self.x, self.velocity]

    def move_rk4(self, propulsion_force, delta_time):
        self.acceleration = (propulsion_force / self.mass - self.velocity/self.tau)

        dv_dt_func = lambda v,t:(propulsion_force/self.mass-v/self.tau)
        self.velocity = runge_kutta(self.velocity, 0, delta_time, dv_dt_func)
        dx = self.velocity * delta_time
        self.x += dx
        return dx

    def move(self, propulsion_force, delta_time):
        acceleration = (propulsion_force  / self.mass - self.velocity/self.tau)
        self.velocity += acceleration * delta_time 
        delta_x = self.velocity * delta_time
        self.x += delta_x
        self.acceleration = acceleration
        return delta_x
    
    def reset(self):
        self.x = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

class Env(gym.Env):
    def __init__(self, log_dir, delta_time=0.1, 
                        track_length=10, time_limit = 20,
                        mass=1, tau=0.892, 
                        max_force=12.2, sigma=9.93 * 4.184, E_0=575 * 4.184):

        
        self.time = 0.0
        self.delta_time = delta_time
        self.time_max = 2000

        self.battery = Battery(E_0, sigma, mass)
        self.object = MovingObject(mass, tau)
        self.max_force = max_force * mass
        self.track_length = track_length

        self.time_limit = time_limit
        self.log_dir = log_dir
        self.viewer = None
        self.mtp_axs = None
        self.observation_space = [len(self.get_state())]
        self.episode_count = 1
        self.reset()

        high = np.array(
            [
                self.track_length * 2,
                np.Inf,
                np.Inf
            ],
            dtype=np.float32,
        )

        low = np.array(
            [
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        
        # self.action_space = spaces.Box(np.array([-np.Inf]), np.array([np.Inf]), dtype=np.float32)
        self.action_space = spaces.Box(-np.Inf, np.Inf, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        

        self.info_for_log_names = [
            "propulsion_force", "acceleration", "reward", "message"
        ]
        
        self.log_raw_csv = False
        self.log_raw_csv_every = 1

        os.makedirs(log_dir, exist_ok=True)

        # with open(os.path.join(log_dir, "info.txt"), mode="w") as f:
        #     print(",".join(["delta_time", "track_length", "time_limit", "mass", "mu", 
        #     "capacity","max_current","voltage_0","state_of_charge_0"]), file=f)
        #     print(",".join([                    
        #             str(x) for x in 
        #                 [delta_time, track_length, time_limit, mass, mu, capacity, max_current, voltage_0, state_of_charge_0]
        #         ]), file=f)
        self.log_file = None
    
    def enable_log_raw_csv(self, every):
        self.log_raw_csv = True
        self.log_raw_csv_every = every
    
    def reset(self): 
        if self.time > 0:
            if self.mtp_axs is not None:
                os.makedirs(os.path.join(self.log_dir, "fig"), exist_ok=True)
                plt.savefig(os.path.join(self.log_dir, "fig", "episode_{}.jpg".format(self.episode_count)))
                for ax in self.mtp_axs:
                    ax.clear()
            if self.log_file is not None:
                path_info = os.path.join(self.log_dir, "csv", "log_{}_info.txt".format(self.episode_count))
                with open(path_info, mode="w") as f:
                    print(self.done_reason, file=f)
                self.log_file.flush()
                self.log_file.close()
                self.log_file = None
    
            self.episode_count += 1
        
        self.battery.reset()
        self.object.reset()
        self.time = 0
        self.steps_beyond_done = None
        self.episode_reward = 0.0
        self.info_for_log = [
            0, 0, 0, ""
        ]
        self.done_reason = ""

        return np.array(self.get_state(), dtype=np.float32)
    

    def get_state(self):
        return (*self.object.get_state(), *self.battery.get_state())
    
    def rwd_fn_log_barrier_derivative(self):
        return (1/self.time_max) / (1 - self.time / self.time_max)

    def rwd_fn_log_barrier(self):
        return -math.log(1 - self.time / self.time_max)


    def check_failure_status(self, force):
        conds = [
            [force > self.max_force, "force > self.max_force ({:.3f},{:.3f})".format(force, self.max_force)],
            [self.battery.E <= 0, "battery.E <= 0 ({:.3f})".format(self.battery.E)],
            [self.object.x < -self.track_length / 2, "object.x < -track_length / 2 ({:.3f},{})".format(self.object.x, self.track_length)],
            [self.object.velocity < 0, "object.velocity < 0 ({:.3f})".format(self.object.velocity)],
        ]
        failed = False
        message = ""
        for c,m in conds:
            if c:
                failed = True
                message += "\n" + m
        return failed, message

    def step(self, action):        
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg
        
        propulsion_force = action[0]

        if self.time == 0 and self.log_raw_csv:
            self.log_csv_current_state()

        delta_x = self.object.move(propulsion_force, self.delta_time)
        delta_e = self.battery.consume(propulsion_force, self.object.velocity, self.delta_time)
        self.time += self.delta_time
        
        failed, fail_message = self.check_failure_status(force=propulsion_force)
        succ = (self.object.x >= self.track_length)
        
        done = succ or failed
        if failed:
            reward = 0
            self.done_reason = fail_message
        elif succ:
            if self.time > self.time_max:
                reward = 0
            else:
                reward = self.rwdfn_log_barrier_derivative()
            self.episode_reward += reward
            self.done_reason = "Success (t:{:.3f}, r:{:.3f})".format(self.time, self.episode_reward)
        else:
            reward = 0
            self.episode_reward += reward
        
        if self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        if done and (self.steps_beyond_done is None):
            self.steps_beyond_done = 0
        
        self.info_for_log = [
            propulsion_force, self.object.acceleration, reward, self.done_reason
        ]

        if self.log_raw_csv:
            self.log_csv_current_state()
        
        return np.array(self.get_state(), dtype=np.float32), reward, done, {"succ": succ}
    
    def log_csv_current_state(self):
        if self.episode_count % self.log_raw_csv_every != 0:
            return

        if self.log_file is None:
            os.makedirs(os.path.join(self.log_dir, "csv"), exist_ok=True)
            path = os.path.join(self.log_dir, "csv", "log_{}.csv".format(self.episode_count))
            self.log_file = open(path, mode="w")
            print(",".join(["time","x", "velocity", "E"]+self.info_for_log_names), file=self.log_file)
        log_list = [self.time] + list(self.get_state()) + self.info_for_log
        log_list = ["\""+str(x)+"\"" for x in log_list]
        print(",".join(log_list), file=self.log_file)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

