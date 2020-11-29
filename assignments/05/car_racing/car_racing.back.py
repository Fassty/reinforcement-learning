# from stable_baselines3 import DDPG
# from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DQN



import argparse
import wrappers
import tensorflow as tf
import numpy as np
import gym
# from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os
import functools


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int,
                    help="Render some episodes.")
parser.add_argument("--seed", default=13, type=int, help="Random seed.")

parser.add_argument("--frame_skip", default=2, type=int, help="Frame skip.")

parser.add_argument("--batch_size", default=2048, type=int)
parser.add_argument("--gamma", default=0.99, type=float)
parser.add_argument("--controller_size", default=64, type=int)
parser.add_argument("--controller_depth", default=2, type=int)

parser.add_argument("--epsilon", default=0.3, type=float)
parser.add_argument("--epsilon_final", default=0.075, type=float)
parser.add_argument("--epsilon_final_at", default=0.5, type=float)

parser.add_argument("--render_every", default=10, type=int)
# parser.add_argument("--save_to", default="dqn_racecar", type=str)
parser.add_argument("--load_from", default="890_best_model.zip", type=str)
parser.add_argument("--logdir", default="logs", type=str)


parser.add_argument("--discrete_actions", default=True, action="store_false")
parser.add_argument("--total_timesteps", default=2000000, type=int)

parser.add_argument("--buffer_size", default=1000000, type=int)
parser.add_argument("--train_freq", default=8, type=int)

# Reward shaping
parser.add_argument("--green_penalty", default=1, type=float)
parser.add_argument("--speed_limit", default=15, type=float)
parser.add_argument("--speed_limit_end", default=0.65, type=float)


parser.add_argument("--original_gym", default=False, action="store_true")

# Only for DQN
parser.add_argument("--target_update_interval", default=5000, type=int)
parser.add_argument("--learning_starts", default=3000, type=int)
parser.add_argument("--tau", default=1, type=float)

parser.add_argument("--lr", default=0.0001, type=float)

# Only for DDPG
parser.add_argument("--action_noise", default=0.1, type=float, help="Action noise std.")


# TODO
# - green penalty
# - concatenate multiple frames for context
# - vectorized env?
# - discretize the env
# - expert trajectories
# - punish inactivity
# - skip first 50-or-so frames



def make_env(seed, silent=False, evaluate_for=15):

    def _init():
        if args.original_gym:
            env = gym.make("CarRacing-v0") 
        else:
            env = gym.make("CarRacingSoftFS{}-v0".format(args.frame_skip))

        env = wrappers.VaeCarWrapper(env, silent=silent)
        if not args.recodex:
            env = wrappers.TerminateEarlyWrapper(env)

        if args.discrete_actions:
            env = wrappers.CarDiscretizatinoWrapper(env)

        env = wrappers.EvaluationWrapper(env, args.seed, evaluate_for=evaluate_for, render_every=args.render_every,
                                         report_each=1, logname=args.logdir + "/" + get_params_str(seed))

        env = wrappers.RewardWrapper(env, green_penalty=args.green_penalty, speed_limit=args.speed_limit, speed_limit_end=args.speed_limit_end * args.total_timesteps, silent=silent)
        return env

    return _init

def get_params_str(seed):
    if args.discrete_actions:
        return f"DQN,fs={args.frame_skip},sp_l={args.speed_limit},sp_l_end={args.speed_limit_end},gp={args.green_penalty},tau={args.tau},eps={args.epsilon},eps_fin={args.epsilon_final},eps_fin_at={args.epsilon_final_at},learning_starts={args.learning_starts},train_freq={args.train_freq},target_up_freq={args.target_update_interval},bs={args.batch_size},g={args.gamma},c_size={args.controller_size},c_depth={args.controller_depth},timesteps={args.total_timesteps},buf_size={args.buffer_size},seed={seed}"
    else:

        # TODO
        return f"DDPG,fs={args.frame_skip},noise={args.action_noise},gp={args.green_penalty},cx_len={args.context_length},bs={args.batch_size},g={args.gamma},c_size={args.controller_size},timesteps={args.total_timesteps},buf_s={args.buffer_size},seed={seed}"


def evaluate(model, seed, evaluate_for=15):
    print("Evaluating!")
    env = make_env(seed=seed, silent=True)()

    while True:
        state, done = env.reset(start_evaluation=True), False
        while not done:
            env.render()
            action, _states = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    print(get_params_str(args.seed))

    env = make_env(seed=args.seed)()

    if args.load_from is not None:
        print("loading model", args.load_from)
        model = DQN.load(args.load_from)
        # model.set_env(env)

    if not args.recodex and args.total_timesteps > 0:

        policy_kwargs = dict(net_arch=[args.controller_size] * args.controller_depth)


        if args.load_from is None:
            if args.discrete_actions:
                model = DQN('MlpPolicy', env, tau=args.tau, exploration_initial_eps=args.epsilon, exploration_final_eps=args.epsilon_final, exploration_fraction=args.epsilon_final_at, train_freq=args.train_freq,
                            batch_size=args.batch_size, buffer_size=args.buffer_size, gamma=args.gamma, target_update_interval=args.target_update_interval, learning_starts=args.learning_starts, policy_kwargs=policy_kwargs, verbose=1)

            else:
                # The noise objects for DDPG
                n_actions = env.action_space.shape[-1]
                # action_noise = NormalActionNoise(mean=np.zeros(
                    # n_actions), sigma=args.action_noise * np.ones(n_actions))

                # model = DDPG('MlpPolicy', env, action_noise=action_noise, batch_size=args.batch_size,
                            #  buffer_size=args.buffer_size, gamma=args.gamma, policy_kwargs=policy_kwargs, verbose=1)

        model.learn(total_timesteps=args.total_timesteps, log_interval=1)
        model.save("saved_models/" + get_params_str(f"envSeed-{args.seed}"))

    evaluate(model, args.seed)