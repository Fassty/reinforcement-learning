import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model


class Network:
    def __init__(self, env, args):
        inputs = Input(shape=(4,))

        x = Dense(64, activation='relu')(inputs)
        x = Dense(128, activation='relu')(x)
        outputs = Dense(env.action_space.n, activation='linear')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss=tf.keras.losses.MeanSquaredError()
        )

        self.target = tf.keras.models.clone_model(self.model)

    @tf.function
    def train(self, states, q_values):
        self.model.optimizer.minimize(
            lambda: self.model.loss(q_values, self.model(states, training=True)),
            var_list=self.model.trainable_variables
        )

    @tf.function
    def predict(self, states):
        return self.model(states)

    @tf.function
    def predict_target(self, states):
        return self.target(states)

    def copy_weights_from(self, other):
        for var, other_var in zip(self.model.variables, other._model.variables):
            var.assign(other_var)

    def update_target_weights(self):
        self.target.set_weights(self.model.get_weights())

    def save(self):
        name = './dqn.model'
        self.model.save(name)
