#!/usr/bin/env python3
import argparse
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
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=4, type=int, help="Maximum number of threads to use.")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--batch_size", default=5, type=int, help="Batch size.")
parser.add_argument("--episodes", default=600, type=int, help="Training episodes.")
parser.add_argument("--gamma", default=0.9999, type=float, help="Discounting factor.")
parser.add_argument("--hidden_layer_size", default=None, type=int, help="Size of hidden layer.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate.")

def typed_np_function(*types):
    """Typed NumPy function decorator.

    Can be used to wrap a function expecting NumPy inputs.

    It converts input positional arguments to NumPy arrays of the given types,
    and passes the result through `np.array` before returning.
    """
    def check_typed_np_function(wrapped, args):
        if len(types) != len(args):
            while hasattr(wrapped, "__wrapped__"): wrapped = wrapped.__wrapped__
            raise AssertionError("The typed_np_function decorator for {} expected {} arguments, but got {}".format(wrapped, len(types), len(args)))

    class TypedNpFunctionWrapperMethod:
        def __init__(self, instance, func):
            self._instance, self.__wrapped__ = instance, func
        def __call__(self, *args, **kwargs):
            check_typed_np_function(self.__wrapped__, args)
            return np.array(self.__wrapped__(*[np.asarray(arg, typ) for arg, typ in zip(args, types)], **kwargs))

    class TypedNpFunctionWrapper:
        def __init__(self, func):
            self.__wrapped__ = func
        def __call__(self, *args, **kwargs):
            check_typed_np_function(self.__wrapped__, args)
            return np.array(self.__wrapped__(*[np.asarray(arg, typ) for arg, typ in zip(args, types)], **kwargs))
        def __get__(self, instance, cls):
            return TypedNpFunctionWrapperMethod(instance, self.__wrapped__.__get__(instance, cls))

    return TypedNpFunctionWrapper

class Network:
    def __init__(self, env, args):
        # TODO: Create a suitable model.
        #
        # Apart from the model defined in `reinforce`, define also another
        # model for computing baseline (with one output, using a dense layer
        # without activation).
        #
        # Using Adam optimizer with given `args.learning_rate` for both models
        # is a good default.
        inputs = tf.keras.Input(shape=env.observation_space.shape)
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(1024, activation=tf.keras.activations.tanh,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)
        outputs = tf.keras.layers.Dense(env.action_space.n, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)

        baseline_in = tf.keras.Input(shape=env.observation_space.shape)
        x = tf.keras.layers.Flatten()(baseline_in)
        x = tf.keras.layers.Dense(512, activation=tf.keras.activations.tanh,
                                  kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)
        baseline_out = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.00001))(x)

        decay_rate = 0.6
        decay_steps = args.episodes // args.batch_size
        lr_schedule = tf.optimizers.schedules.ExponentialDecay(0.01, decay_steps=decay_steps, decay_rate=decay_rate)

        self.model = tf.keras.Model(inputs, outputs)
        self.model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
        )

        self.baseline = tf.keras.Model(baseline_in, baseline_out)
        self.baseline.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=0.5)
        )

    # TODO: Define a training method.
    #
    # Note that we need to use @tf.function for efficiency (using `train_on_batch`
    # on extremely small batches/networks has considerable overhead).
    #
    # The `wrappers.typed_np_function` automatically converts input arguments
    # to NumPy arrays of given type, and converts the result to a NumPy array.
    @typed_np_function(np.float32, np.int32, np.float32)
    @tf.function(experimental_relax_shapes=True)
    def train(self, states, actions, returns):
        # You should:
        # - compute the predicted baseline using the baseline model
        # - train the policy model, using `returns - predicted_baseline` as
        #   advantage estimate
        # - train the baseline model to predict `returns`

        # Note to self: The dimensions are weird
        predicted_baseline = self.baseline(states)[0]
        predicted_baseline = tf.reshape(predicted_baseline, (predicted_baseline.shape[0],))
        advantage = returns - predicted_baseline
        self.baseline.optimizer.minimize(
            lambda: self.baseline.loss(advantage, [self.baseline(states)], sample_weight=advantage),
            var_list=self.baseline.trainable_variables
        )
        predicted_baseline = self.baseline(states)[0]
        sample_weights = returns - predicted_baseline
        self.model.optimizer.minimize(
            lambda: self.model.loss(actions, self.model(states), sample_weight=sample_weights),
            var_list=self.model.trainable_variables
        )


    # Predict method, again with manual @tf.function for efficiency.
    @typed_np_function(np.float32)
    @tf.function
    def predict(self, states):
        return self.model(states)

def main(env, args):
    # Fix random seeds and number of threads
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Construct the network
    network = Network(env, args)

    # Training
    for _ in range(args.episodes // args.batch_size):
        batch_states, batch_actions, batch_returns = [], [], []
        for _ in range(args.batch_size):
            # Perform episode
            states, actions, rewards = [], [], []
            state, done = env.reset(), False
            while not done:
                if args.render_each and env.episode > 0 and env.episode % args.render_each == 0:
                    env.render()

                # TODO(reinforce): Choose `action` according to probabilities
                # distribution (see `np.random.choice`), which you
                # can compute using `network.predict` and current `state`.
                probabilities = network.predict([state])[0]
                action = np.random.choice(env.action_space.n, p=probabilities)

                next_state, reward, done, _ = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            # TODO(reinforce): Compute returns by summing rewards (with discounting)
            returns, running = [], 0
            for reward in rewards[::-1]:
                running = args.gamma * running + reward
                returns.insert(0, running)

            # TODO(reinforce): Add states, actions and returns to the training batch
            batch_states += states
            batch_actions += actions
            batch_returns += returns

        network.train(batch_states, batch_actions, batch_returns)

    # Final evaluation
    while True:
        state, done = env.reset(True), False
        while not done:
            # TODO(reinforce): Choose greedy action
            probabilities = network.predict([state])[0]
            action = np.argmax(probabilities)
            state, reward, done, _ = env.step(action)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    # Create the environment
    env = wrappers.EvaluationWrapper(gym.make("CartPole-v1"), args.seed)

    main(env, args)
