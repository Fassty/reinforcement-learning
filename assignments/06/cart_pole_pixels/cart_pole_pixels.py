#!/usr/bin/env python3
import argparse
import collections
import os
import random

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import cart_pole_pixels_environment
import pickle
import wrappers
#import cma
import multiprocessing as mp
#from tqdm import tqdm
from skimage.color import rgb2gray
from skimage.transform import resize
from train_VAE import load_vae
from DQN import Network

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=True, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=3, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
parser.add_argument("--DQN", default=False, type=bool, help="Train using DQN")
parser.add_argument("--evolution", default=True, type=bool, help="Train using evolution strategy")
# For these and any other arguments you add, ReCodEx will keep your default value.

_EMBEDDING_SIZE = 4
_NUM_PREDICTIONS = 1
_NUM_ACTIONS = 1
_NUM_PARAMS = _NUM_PREDICTIONS * _EMBEDDING_SIZE + _NUM_PREDICTIONS


def get_weights_bias(params):
    weights = params[: _NUM_PARAMS - _NUM_PREDICTIONS]
    bias = params[-_NUM_PREDICTIONS:]
    weights = np.reshape(weights, [_EMBEDDING_SIZE, _NUM_PREDICTIONS])
    return weights, bias


def decide_action(model, observation, params):
    weights, bias = get_weights_bias(params)

    embedding = model.get_latent_representation(np.array([observation]))

    prediction = np.matmul(np.squeeze(embedding), weights) + bias
    prediction = np.tanh(prediction)

    if prediction[0] < 0:
        action = 0
    else:
        action = 1

    return action


def play(params, render=False):
    _NUM_TRIALS = 30
    agent_reward = 0
    model = load_vae()
    for trial in range(_NUM_TRIALS):
        observation, done = env.reset(), False

        total_reward = 0.0
        steps = 0
        while not done:
            if render:
                env.render()
            action = decide_action(model, observation, params)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1

        agent_reward += total_reward

    return - (agent_reward / _NUM_TRIALS)


# def train(load_from=None):
#     if os.path.exists('best_params.npy'):
#         best_params = np.load('best_params.npy')
#         es = cma.CMAEvolutionStrategy(best_params, 0.1, {'popsize': 50})
#     elif load_from is not None and os.path.exists(load_from):
#         es = pickle.load(open(load_from, 'rb'))
#     else:
#         es = cma.CMAEvolutionStrategy(_NUM_PARAMS * [0], 0.1, {'popsize': 40})
#
#     rewards_through_gens = []
#     generation = 1
#     try:
#         while not es.stop():
#             solutions = es.ask()
#             with mp.Pool(mp.cpu_count()) as p:
#                 rewards = list(tqdm(p.imap(play, list(solutions)), total=len(solutions)))
#
#             es.tell(solutions, rewards)
#
#             rewards = np.array(rewards) * (-1.)
#             print("\n**************")
#             print("Generation: {}".format(generation))
#             print("Min reward: {:.3f}\nMax reward: {:.3f}".format(np.min(rewards), np.max(rewards)))
#             print("Avg reward: {:.3f}".format(np.mean(rewards)))
#             print("**************\n")
#
#             # if generation % 50 == 0:
#             #     pickle.dump(es, open(f'saved_cma_gen_{generation}.pkl', 'wb'))
#             generation += 1
#             rewards_through_gens.append(rewards)
#             np.save('rewards', rewards_through_gens)
#
#         pickle.dump(es, open(f'saved_model.pkl', 'wb'))
#     except (KeyboardInterrupt, SystemExit):
#         print('Manual Interrupt')
#     except Exception as e:
#         print(f'Exception {e}')
#     return es


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    if args.recodex:
        model = load_vae()
        best_params = np.load('best_params.npy', allow_pickle=True)
        # TODO: Perform evaluation of a trained model.
        while True:
            state, done = env.reset(start_evaluation=True), False
            while not done:
                # env.render()
                # TODO: Choose an action
                action = decide_action(model, state, best_params)
                state, reward, done, _ = env.step(action)

    elif args.DQN:
        network = Network(env, args)
        if os.path.exists('dqn.model'):
            network.model = tf.keras.models.load_model('dqn.model')
        vae = load_vae()
        replay_buffer = collections.deque(maxlen=100000)
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

        epsilon = 0.25
        gamma = 1

        for i in tqdm(range(10000)):
            state, done = env.reset(), False
            while not done:
                embedding = vae.get_latent_representation(np.array([state]))

                q_values = network.predict(embedding)[0]

                if np.random.uniform() >= epsilon:
                    action = np.argmax(q_values)
                else:
                    action = np.random.randint(0, env.action_space.n)

                next_state, reward, done, _ = env.step(action)

                replay_buffer.append(Transition(embedding, action, reward, done, vae.get_latent_representation(np.array([next_state]))))

                if len(replay_buffer) > 32:
                    minibatch = random.sample(replay_buffer, 32)

                    states = np.array([t.state[0] for t in minibatch])
                    actions = np.array([t.action for t in minibatch])
                    rewards = np.array([t.reward for t in minibatch])
                    dones = np.array([t.done for t in minibatch])
                    next_states = np.array([t.next_state[0] for t in minibatch])

                    q_values = np.array(network.predict(states))
                    q_values_next = network.predict(next_states)

                    for Q, action, reward, next_Q, is_done in zip(q_values, actions, rewards, q_values_next, dones):
                        Q[action] = reward + (0 if is_done else gamma * np.max(next_Q))

                    network.train(states, q_values)

                    if i % 100 == 0:
                        network.update_target_weights()

                    if i % 100 == 0:
                        network.save()

                state = next_state

            epsilon = np.exp(np.interp(env.episode + 1, [0, 5000], [np.log(0.25), np.log(0.01)]))

    elif args.evolution:
        es = train(load_from='saved_model.pkl')
        np.save('best_params', es.best.get()[0])
        best_params = es.best.get()[0]
        play(best_params, render=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPolePixels-v0"), args.seed, report_each=10, evaluate_for=15)

    main(env, args)
