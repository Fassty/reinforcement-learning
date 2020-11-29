#!/usr/bin/env python3
import argparse
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=10, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--entropy_regularization", default=0.05, type=float, help="Entropy regularization weight.")
parser.add_argument("--evaluate_each", default=1000, type=int, help="Evaluate each number of batches.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=42, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--tiles", default=16, type=int, help="Tiles to use.")
parser.add_argument("--workers", default=32, type=int, help="Number of parallel workers.")

class Network:
    def __init__(self, env, args):
        # TODO: Analogously to paac, your model should contain two components:
        # - actor, which predicts distribution over the actions
        # - critic, which predicts the value function
        #
        # The given states are tile encoded, so they are integral indices of
        # tiles intersecting the state. Therefore, you should convert them
        # to dense encoding (one-hot-like, with with `args.tiles` ones).
        # (Or you can even use embeddings for better efficiency.)
        #
        # The actor computes `mus` and `sds`, each of shape [batch_size, actions].
        # Compute each independently using states as input, adding a fully connected
        # layer with `args.hidden_layer_size` units and ReLU activation. Then:
        # - For `mus`, add a fully connected layer with `actions` outputs.
        #   To avoid `mus` moving from the required range, you should apply
        #   properly scaled `tf.tanh` activation.
        # - For `sds`, add a fully connected layer with `actions` outputs
        #   and `tf.nn.softplus` activation.
        #
        # The critic should be a usual one, passing states through one hidden
        # layer with `args.hidden_layer_size` ReLU units and then predicting
        # the value function.

        self.entropy_regularization = args.entropy_regularization
        weights = env.observation_space.nvec[-1]
        n_actions = env.action_space.shape[0]
        inputs = tf.keras.Input(shape=(weights,))
        # TODO: try embedding layer
        embedding = tf.keras.layers.Embedding()()
        # TODO: can I share this layer? -> try to make it 3x, 9x, 27x larger
        mus_hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        sds_hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)
        values_hidden = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(inputs)

        mus = tf.keras.layers.Dense(n_actions, activation='tanh')(mus_hidden)
        sds = tf.keras.layers.Dense(n_actions, activation='softplus')(sds_hidden)
        values = tf.keras.layers.Dense(1)(values_hidden)

        self.model = tf.keras.Model(inputs=inputs, outputs=[mus, sds, values])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=1)
        )

        self.mse_loss = tf.keras.losses.MeanSquaredError()

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        # TODO: Run the model on given `states` and compute
        # sds, mus and predicted values. Then create `action_distribution` using
        # `tfp.distributions.Normal` class and computed mus and sds.
        # In PyTorch, the corresponding class is `torch.distributions.normal.Normal`.
        #
        # TODO: Compute total loss as a sum of three losses:
        # - negative log likelihood of the `actions` in the `action_distribution`
        #   (using the `log_prob` method). You then need to sum the log probabilities
        #   of actions in a single batch example (using `tf.math.reduce_sum` with `axis=1`).
        #   Finally multiply the resulting vector by (returns - predicted values)
        #   and compute its mean.
        # - negative value of the distribution entropy (use `entropy` method of
        #   the `action_distribution`) weighted by `args.entropy_regularization`.
        # - mean square error of the `returns` and predicted values.
        with tf.GradientTape() as tape:
            mus, sds, values = self.model(states, training=True)
            action_distribution = tfp.distributions.Normal(mus, sds)
            log_prob = - action_distribution.log_prob(actions)

            advantage = returns - tf.stop_gradient(tf.reshape(values, [-1]))
            loss = tf.reduce_mean(tf.reduce_sum(log_prob, axis=1) * advantage)

            loss += - action_distribution.entropy() * self.entropy_regularization

            loss += self.mse_loss(returns, values)

        gradient = tape.gradient(loss, self.model.variables)
        self.model.optimizer.apply_gradients(zip(gradient, self.model.variables))

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        mus, sds, _ = self.model(states)
        return mus, sds

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        _, _, values = self.model(states)
        return values[:, 0]


def multi_hot_states(states, weights):
    return tf.reduce_sum(tf.one_hot(states, weights, axis=2), axis=1)


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)
    weights = env.observation_space.nvec[-1]

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            state = multi_hot_states([state], weights)
            action = network.predict_actions(state)[0][0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    # Create the vectorized environment
    vector_env = gym.vector.AsyncVectorEnv(
        [lambda: wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles)] * args.workers)
    vector_env.seed(args.seed)
    states = vector_env.reset()
    states = multi_hot_states(states, weights)

    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            # TODO: Predict action distribution using `network.predict_actions`
            # and then sample it using for example `np.random.normal`. Do not
            # forget to clip the actions to the `env.action_space.{low,high}`
            # range, for example using `np.clip`.

            mus, sds = network.predict_actions(states)

            actions = np.random.normal(mus, sds)
            actions = np.clip(actions, env.action_space.low, env.action_space.high)

            # TODO(paac): Perform steps in the vectorized environment
            next_states, rewards, dones, _ = vector_env.step(actions)
            next_states = multi_hot_states(next_states, weights)

            # TODO(paac): Compute estimates of returns by one-step bootstrapping
            predicted_values = network.predict_values(next_states)
            returns = rewards + (args.gamma * np.array([0 if done else pred for done, pred in zip(dones, predicted_values)]))

            # TODO(paac): Train network using current states, chosen actions and estimated returns
            network.train(states, actions, returns)

            states = next_states

        # Periodic evaluation
        total_reward = []
        for _ in range(args.evaluate_for):
            total_reward.append(evaluate_episode())
        print(f'Mean {args.evaluate_for} episodes return {np.mean(total_reward)}')
        if np.mean(total_reward) > 90:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(wrappers.DiscreteMountainCarWrapper(gym.make("MountainCarContinuous-v0"), tiles=args.tiles), args.seed)

    main(env, args)
