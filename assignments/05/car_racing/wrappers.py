#!/usr/bin/env python3
import sys
import os

from collections import deque

from gym.spaces.box import Box
import gym
import numpy as np
from PIL import Image
import tensorflow as tf
import car_racing_environment


last_rewards = deque(maxlen=30)


# class SkipFirstFrames(gym.Wrapper):
    # def __init__(self, env):


class TerminateEarlyWrapper(gym.Wrapper, ):
    def __init__(self, env, cumulative_reward_thres=-600):
        super().__init__(env)

        self.cumulative_reward_thres = cumulative_reward_thres
        
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if(sum(globals()["last_rewards"]) < self.cumulative_reward_thres):
            globals()["last_rewards"].clear()
            done = True

        return next_state, reward, done, info


class RewardWrapper(gym.RewardWrapper):
    def __init__(self, env, green_penalty= 0.05, speed_limit=0.7, speed_limit_end=500000, silent=False):
        super().__init__(env)
        
        self.speed_limit = speed_limit
        self.speed_limit_end = speed_limit_end
        self.green_penalty = green_penalty

        self.silent = silent

        self.i = 0

    
    def reward(self, rew):
        self.i += 1

        gp = self.green * self.green_penalty
        rew -= gp

        # sspeeding = abs(self.speed - 0.8)**2 * self.speed_limit * max(0, 1 - self.i / self.speed_limit_end)
        speeding = abs(self.speed - 0.6)**2 * self.speed_limit 
        rew -= speeding

        globals()["last_rewards"].append(rew)

        if not self.silent:
            print("speeding:\t", speeding, "green penalty:\t", gp)
            print("reward: ", rew)

        return rew

class CarDiscretizatinoWrapper(gym.ActionWrapper):

    def __init__(self, env):
        super(CarDiscretizatinoWrapper, self).__init__(env)


        self.action_map = [

            [-1.0, 0.0, 0.0],  # Turn hard left
            [+1.0, 0.0, 0.0],  # Turn hard right
            [0.0, 0.0, 0.8],  # Brake
            [0.0, 1.0, 0],  # Accelerate
            [0.0, 0.0, 0.0],  # Do-(almost)-nothing
        ]

        self.action_space = gym.spaces.Discrete(len(self.action_map))

    def action(self, action):
        # print(self.action_map[action])
        return self.action_map[action]


class VaeCarWrapper(gym.ObservationWrapper):
    def __init__(self, env, silent=False):
        super().__init__(env)


        from vae.vae import CVAE
        from utils import PARSER
        args = PARSER.parse_args(['--config_path', 'configs/carracing.config'])
        model_path_name = "models/tf_vae"

        self.vae = CVAE(args)



        # self.vae.set_weights(tf.keras.models.load_model(
        #     model_path_name, compile=False).get_weights())

        self.vae.set_weights(np.load("vae_weights.npy", allow_pickle=True))


        self.observation_space = Box(low=float("-inf"), high=float("inf"), shape=(40,))
        self.silent = silent

    def _process_frame(self, frame):
        obs = (frame[0:84, :, :] * 255).astype(np.uint8)
        obs = Image.fromarray(obs, mode="RGB").resize((64, 64))
        obs = np.array(obs)


        return np.array(self.vae.encode(obs.reshape(1, 64, 64, 3)/255)[0])

    def observation(self, frame):
        # far-front spike
        car_body = np.sum((frame[56:59, 47, 1] > 0.5).flatten())

        # main headlights
        car_body = np.sum((frame[59:74, 46:49, 1] > 0.5).flatten())

        # rear wheels
        car_body += np.sum((frame[72:76, 44, 1] > 0.5).flatten())   
        car_body += np.sum((frame[72:76, 50, 1] > 0.5).flatten())


        #sides
        car_body += np.sum((frame[67:77, 45, 1] > 0.5).flatten())   
        car_body += np.sum((frame[67:77, 49, 1] > 0.5).flatten())

        self.green = car_body / 55.0

        self.speed = sum(frame[85:, 2, 0]) / 5


        self.abs1 = sum(frame[85:, 9, 2])
        self.abs2 = sum(frame[85:, 14, 2])
        self.abs3 = sum(frame[85:, 19, 2])
        self.abs4 = sum(frame[85:, 24, 2])

        steering_input_left = sum(frame[90, 37:48, 1])
        steering_input_right = sum(frame[90, 47:58, 1])
        self.steering = steering_input_right - steering_input_left

        rotation_left = sum(frame[90, 59:72, 0])
        rotation_right = sum(frame[90, 72:85, 0])
        self.rotation = rotation_right - rotation_left

        if not self.silent:
            print(f"green:{self.green}\tspeed:{self.speed}\tabs:\t{self.abs1}\t{self.abs2}\t{self.abs3}\t{self.abs4}\tsteering:{self.steering}\trotation:{self.rotation}") 

        features = self._process_frame(frame)

        return np.concatenate([features, [self.speed, self.green, self.abs1, self.abs2, self.abs3, self.abs4, self.steering, self.rotation]])


"""
class VaeCarWrapperLord(gym.ObservationWrapper):
    def __init__(self, env, silent=False):
        super().__init__(env)

        from lord_vae import load_vae

        self.vae = load_vae()

        self.observation_space = Box(low=float("-inf"), high=float("inf"), shape=(40,))
        self.silent = silent

    def _process_frame(self, frame):
		self.image = tf.compat.v1.placeholder(tf.float32, [None, 96, 96, 3], name='image')
		self.resized_image = tf.image.resize(self.image, [64, 64])
		tf.compat.v1.summary.image('resized_image', self.resized_image, 20)

        obs = (frame[0:84, :, :] * 255).astype(np.uint8)
        obs = Image.fromarray(obs, mode="RGB").resize((64, 64))
        obs = np.array(obs)


        return np.array(self.vae.encode(obs.reshape(1, 64, 64, 3)/255)[0])

    def observation(self, frame):
        # far-front spike
        car_body = np.sum((frame[56:59, 47, 1] > 0.5).flatten())

        # main headlights
        car_body = np.sum((frame[59:74, 46:49, 1] > 0.5).flatten())

        # rear wheels
        car_body += np.sum((frame[72:76, 44, 1] > 0.5).flatten())   
        car_body += np.sum((frame[72:76, 50, 1] > 0.5).flatten())


        #sides
        car_body += np.sum((frame[67:77, 45, 1] > 0.5).flatten())   
        car_body += np.sum((frame[67:77, 49, 1] > 0.5).flatten())

        self.green = car_body / 55.0

        self.speed = sum(frame[85:, 2, 0]) / 5


        self.abs1 = sum(frame[85:, 9, 2])
        self.abs2 = sum(frame[85:, 14, 2])
        self.abs3 = sum(frame[85:, 19, 2])
        self.abs4 = sum(frame[85:, 24, 2])

        steering_input_left = sum(frame[90, 37:48, 1])
        steering_input_right = sum(frame[90, 47:58, 1])
        self.steering = steering_input_right - steering_input_left

        rotation_left = sum(frame[90, 59:72, 0])
        rotation_right = sum(frame[90, 72:85, 0])
        self.rotation = rotation_right - rotation_left

        if not self.silent:
            print(f"green:{self.green}\tspeed:{self.speed}\tabs:\t{self.abs1}\t{self.abs2}\t{self.abs3}\t{self.abs4}\tsteering:{self.steering}\trotation:{self.rotation}") 

        features = self._process_frame(frame)

        return np.concatenate([features, [self.speed, self.green, self.abs1, self.abs2, self.abs3, self.abs4, self.steering, self.rotation]])

"""

class EvaluationWrapper(gym.Wrapper):
    def __init__(self, env, seed=None, evaluate_for=100, report_each=10, logname="logs/default", render_every=0):
        super().__init__(env)
        self._evaluate_for = evaluate_for
        self._report_each = report_each

        self.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

        self._episode_running = False
        self._episode_returns = []
        self._evaluating_from = None
        self._render_every = render_every

        _ = os.popen("mkdir -p logs").read()
        self.logfile = open(logname, "w")

    @property
    def episode(self):
        return len(self._episode_returns)

    def reset(self, start_evaluation=False):
        if self._evaluating_from is not None and self._episode_running:
            raise RuntimeError(
                "Cannot reset a running episode after `start_evaluation=True`")

        if start_evaluation and self._evaluating_from is None:
            self._evaluating_from = self.episode

        self._episode_running = True
        self._episode_return = 0
        return super().reset()

    def step(self, action):
        if not self._episode_running:
            raise RuntimeError(
                "Cannot run `step` on environments without an active episode, run `reset` first")

        observation, reward, done, info = super().step(action)

        if (self._render_every != 0 and self.episode % self._render_every == 0):
            self.render()

        self._episode_return += reward
        if done:
            self._episode_running = False
            self._episode_returns.append(self._episode_return)

            if self.episode % self._report_each == 0:
                print("Episode {}, mean {}-episode return {:.2f} +-{:.2f}".format(
                    self.episode, self._evaluate_for, np.mean(
                        self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:])), file=sys.stderr)
                self.logfile.write(f"{self.episode} {np.mean(self._episode_returns[-self._evaluate_for:])}\n")
                self.logfile.flush()
            if self._evaluating_from is not None and self.episode >= self._evaluating_from + self._evaluate_for:
                print("The mean {}-episode return after evaluation {:.2f} +-{:.2f}".format(
                    self._evaluate_for, np.mean(
                        self._episode_returns[-self._evaluate_for:]),
                    np.std(self._episode_returns[-self._evaluate_for:]), file=sys.stderr))
                self.logfile.write(f"{self.episode} {np.mean(self._episode_returns[-self._evaluate_for:])}\n")
                self.logfile.flush()
                self.close()
                sys.exit(0)

        return observation, reward, done, info


class DiscretizationWrapper(gym.ObservationWrapper):
    def __init__(self, env, separators, tiles=None):
        super().__init__(env)
        self._separators = separators
        self._tiles = tiles

        if tiles is None:
            states = 1
            for separator in separators:
                states *= 1 + len(separator)
            self.observation_space = gym.spaces.Discrete(states)
        else:
            self._first_tile_states, self._rest_tiles_states = 1, 1
            for separator in separators:
                self._first_tile_states *= 1 + len(separator)
                self._rest_tiles_states *= 2 + len(separator)
            self.observation_space = gym.spaces.MultiDiscrete([
                self._first_tile_states + i * self._rest_tiles_states for i in range(tiles)])

            self._separator_offsets, self._separator_tops = [], []
            for separator in separators:
                self._separator_offsets.append(
                    0 if len(separator) <= 1 else (separator[1] - separator[0]) / tiles)
                self._separator_tops.append(math.inf if len(
                    separator) <= 1 else separator[-1] + (separator[1] - separator[0]))

    def observation(self, observations):
        state = 0
        for observation, separator in zip(observations, self._separators):
            state *= 1 + len(separator)
            state += np.digitize(observation, separator)
        if self._tiles is None:
            return state
        else:
            states = [state]
            for t in range(1, self._tiles):
                state = 0
                for i in range(len(self._separators)):
                    state *= 2 + len(self._separators[i])
                    value = observations[i] + ((t * (2 * i + 1)) %
                                               self._tiles) * self._separator_offsets[i]
                    if value > self._separator_tops[i]:
                        state += 1 + len(self._separators[i])
                    else:
                        state += np.digitize(value, self._separators[i])
                states.append(self._first_tile_states + (t - 1)
                              * self._rest_tiles_states + state)
            return states


class DiscreteCartPoleWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=8):
        super().__init__(env, [
            np.linspace(-2.4, 2.4, num=bins + 1)[1:-1],  # cart position
            np.linspace(-3, 3, num=bins + 1)[1:-1],     # pole angle
            np.linspace(-0.5, 0.5, num=bins + 1)[1:-1],  # cart velocity
            np.linspace(-2, 2, num=bins + 1)[1:-1],     # pole angle velocity
        ])


class DiscreteMountainCarWrapper(DiscretizationWrapper):
    def __init__(self, env, bins=None, tiles=None):
        if bins is None:
            bins = 24 if tiles is None or tiles <= 1 else 12 if tiles <= 3 else 8
        super().__init__(env, [
            np.linspace(-1.2, 0.6, num=bins + 1)[1:-1],   # car position
            np.linspace(-0.07, 0.07, num=bins + 1)[1:-1],  # car velocity
        ], tiles)


class DiscreteLunarLanderWrapper(DiscretizationWrapper):
    def __init__(self, env):
        super().__init__(env, [
            np.linspace(-.4,   .4, num=5 + 1)[1:-1],   # x
            np.linspace(-.075, 1.35, num=6 + 1)[1:-1],   # y
            np.linspace(-.5,   .5, num=5 + 1)[1:-1],   # vel x
            np.linspace(-.8,   .8, num=7 + 1)[1:-1],   # vel y
            np.linspace(-.2,   .2, num=3 + 1)[1:-1],   # rot
            np.linspace(-.2,   .2, num=5 + 1)[1:-1],   # ang vel
            [.5],  # lc
            [.5],  # rc
        ])

        self._expert = gym.make("LunarLander-v2")
        self._expert.seed(42)

    def expert_trajectory(self):
        state, trajectory, done = self._expert.reset(), [], False
        initial_state = self.observation(state)
        while not done:
            action = gym.envs.box2d.lunar_lander.heuristic(self._expert, state)
            state, reward, done, _ = self._expert.step(action)
            trajectory.append((action, reward, self.observation(state)))
        return initial_state, trajectory


gym.envs.register(
    id="MountainCar1000-v0",
    entry_point="gym.envs.classic_control:MountainCarEnv",
    max_episode_steps=1000,
    reward_threshold=-110.0,
)
