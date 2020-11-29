#!/usr/bin/env python3
import argparse
import collections
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3") # Report only TF errors by default

import gym
import numpy as np
import tensorflow as tf

import wrappers

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--render_each", default=0, type=int, help="Render some episodes.")
parser.add_argument("--seed", default=None, type=int, help="Random seed.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--env", default="Pendulum-v0", type=str, help="Environment.")
parser.add_argument("--evaluate_each", default=100, type=int, help="Evaluate each number of episodes.")
parser.add_argument("--evaluate_for", default=100, type=int, help="Evaluate the given number of episodes.")
parser.add_argument("--gamma", default=0.99, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--noise_sigma", default=0.2, type=float, help="UB noise sigma.")
parser.add_argument("--noise_theta", default=0.15, type=float, help="UB noise theta.")
parser.add_argument("--target_tau", default=0.005, type=float, help="Target network update weight.")


class Network:
    def __init__(self, env, args):
        # TODO: Create:
        # - an actor, which starts with states and returns actions.
        #   Usually, one or two hidden layers are employed. As in the
        #   paac_continuous, to keep the actions in the required range, you
        #   should apply properly scaled `tf.tanh` activation.
        #
        # - a target actor as the copy of the actor using `tf.keras.models.clone_model`.
        #
        # - a critic, starting with given states and actions producing predicted
        #   returns.  Usually, states are fed through a hidden layer first, and
        #   then concatenated with action and fed through two more hidden
        #   layers, before computing the returns.
        #
        # - a target critic as the copy of the critic using `tf.keras.models.clone_model`.
        self.target_tau = args.target_tau
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        actor_in = tf.keras.Input(shape=env.observation_space.shape)
        x = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(actor_in)
        actor_out = tf.keras.layers.Dense(env.action_space.shape[0], activation='tanh', kernel_initializer=last_init)(x)

        actor_out *= 2.0

        self.actor = tf.keras.Model(inputs=actor_in, outputs=actor_out)
        self.actor.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate)
        )

        self.target_actor = tf.keras.models.clone_model(self.actor)

        critic_in = tf.keras.Input(shape=env.observation_space.shape)
        actions_in = tf.keras.Input(shape=(env.action_space.shape[0], ))
        states_hidden = tf.keras.layers.Dense(args.hidden_layer_size // 2, activation='relu')(critic_in)
        x = tf.keras.layers.Concatenate()([states_hidden, actions_in])
        x = tf.keras.layers.Dense(args.hidden_layer_size // 2, activation='relu')(x)
        x = tf.keras.layers.Dense(args.hidden_layer_size // 2, activation='relu')(x)
        critic_out = tf.keras.layers.Dense(1)(x)

        self.critic = tf.keras.Model(inputs=[critic_in, actions_in], outputs=critic_out)
        self.critic.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=2 * args.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )

        self.target_critic = tf.keras.models.clone_model(self.critic)

    @wrappers.typed_np_function(np.float32, np.float32, np.float32)
    @tf.function
    def train(self, states, actions, returns):
        # TODO: Separately train:
        # - the actor using the DPG loss,
        # - the critic using MSE loss.
        #
        # Furthermore, update the target actor and critic networks by
        # exponential moving average with weight `args.target_tau`. A possible
        # way to implement it inside a `tf.function` is the following:
        #   for var, target_var in zip(network.trainable_variables, target_network.trainable_variables):
        #       target_var.assign(target_var * (1 - target_tau) + var * target_tau)
        # Train critic
        self.critic.optimizer.minimize(
            lambda: self.critic.loss(returns, self.critic((states, actions), training=True)),
            var_list=self.critic.trainable_variables
        )

        # Train actor
        with tf.GradientTape() as actor_tape:
            predicted_actions = self.actor(states, training=True)
            predicted_values = self.critic((states, predicted_actions), training=True)
            predicted_values = tf.reshape(predicted_values, [-1])
            dpg_loss = - tf.math.reduce_mean(predicted_values)

        actor_grads = actor_tape.gradient(dpg_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # Update targets
        for var, target_var in zip(self.actor.trainable_variables, self.target_actor.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)
        for var, target_var in zip(self.critic.trainable_variables, self.target_critic.trainable_variables):
            target_var.assign(target_var * (1 - self.target_tau) + var * self.target_tau)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_actions(self, states):
        return self.actor(states, training=False)

    @wrappers.typed_np_function(np.float32)
    @tf.function
    def predict_values(self, states):
        predicted_actions = self.target_actor(states, training=True)
        return self.target_critic((states, predicted_actions), training=True)[:, 0]


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, mu, theta, sigma):
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        self.state += self.theta * (self.mu - self.state) + np.random.normal(scale=self.sigma, size=self.state.shape)
        return self.state


def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Replay memory; maxlen parameter can be passed to deque for a size limit,
    # which we however do not need in this simple task.
    replay_buffer = collections.deque()
    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "next_state"])

    def evaluate_episode(start_evaluation=False):
        rewards, state, done = 0, env.reset(start_evaluation), False
        while not done:
            if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                env.render()

            # TODO: Predict the action using the greedy policy
            action = network.predict_actions([state])[0]
            state, reward, done, _ = env.step(action)
            rewards += reward
        return rewards

    noise = OrnsteinUhlenbeckNoise(env.action_space.shape[0], 0, args.noise_theta, args.noise_sigma)
    training = True
    while training:
        # Training
        for _ in range(args.evaluate_each):
            state, done = env.reset(), False
            noise.reset()
            while not done:
                # TODO: Predict actions by calling `network.predict_actions`
                # and adding the Ornstein-Uhlenbeck noise. As in paac_continuous,
                # clip the actions to the `env.action_space.{low,high}` range.
                predicted_action = network.predict_actions([state])[0] + noise.sample()
                action = np.clip(predicted_action, env.action_space.low, env.action_space.high)

                next_state, reward, done, _ = env.step(action)
                replay_buffer.append(Transition(state, action, reward, done, next_state))
                state = next_state

                if len(replay_buffer) >= args.batch_size:
                    batch = np.random.choice(len(replay_buffer), size=args.batch_size, replace=False)
                    states, actions, rewards, dones, next_states = map(np.array, zip(*[replay_buffer[i] for i in batch]))
                    # TODO: Perform the training
                    predicted_returns = network.predict_values(next_states)
                    returns = rewards + args.gamma * predicted_returns
                    network.train(states, actions, returns)

        # Periodic evaluation
        total_reward = []
        for _ in range(args.evaluate_for):
            total_reward.append(evaluate_episode())
        if np.mean(total_reward) > -200:
            training = False

    # Final evaluation
    while True:
        evaluate_episode(start_evaluation=True)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make(args.env), args.seed)

    main(env, args)
