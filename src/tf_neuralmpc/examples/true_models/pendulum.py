"""
tf_neuralmpc/examples/true_models/pendulum.py
============================================
"""
from tf_neuralmpc.dynamics_functions.deterministic_models.deterministic_dynamics_function_base \
    import DeterministicDynamicsFunctionBase
import tensorflow as tf
import numpy as np


class PendulumTrueModel(DeterministicDynamicsFunctionBase):
    """This is the pendulum true model for the gym environment"""
    def __init__(self, name=None):
        """
        This is the initializer function for the pendulum true model.


        Parameters
        ---------
        name: String
            Defines the name of the block of the pendulum true model.
        """
        super(PendulumTrueModel, self).__init__(name=name)
        self.g = tf.constant(10, dtype=tf.float32)
        self.max_torque = tf.constant(2.0, dtype=tf.float32)
        self.max_speed = tf.constant(8.0, dtype=tf.float32)
        self.m = tf.constant(1., dtype=tf.float32)
        self.l = tf.constant(1., dtype=tf.float32)
        self.dt = tf.constant(.05, dtype=tf.float32)
        self.pi = tf.constant(float(np.pi), dtype=tf.float32)

    @tf.function
    def __call__(self, x, train): #cos(theta), sin(theta), dtheta, u
        """
        This is the call function for the pendulum true model.


        Parameters
        ---------
        x: tf.float32
            Defines the (s_t, a_t) which is the state and action stacked on top of each other,
            (dims = Batch X (dim_S + dim_U)) [cos(theta), sin(theta), dtheta, u]


        Returns
        -------
        output: tf.float32
            Defines the next state (s_t+1) with (dims = Batch X dim_S), [cos(theta), sin(theta), dtheta]
        """
        u = x[:, 3]
        thdot = x[:, 2]
        theta_cos = x[:, 0]
        theta_sin = x[:, 1]
        theta = tf.math.atan2(theta_sin, theta_cos)
        newthdot = thdot + (-tf.constant(3, dtype=tf.float32)*self.g/ (tf.constant(2, dtype=tf.float32)*self.l)
                            * tf.math.sin(theta + self.pi) + tf.constant(3, dtype=tf.float32) /
                            (self.m * self.l ** tf.constant(2, dtype=tf.float32)) * u) * self.dt
        newth = theta + newthdot * self.dt
        newthdot = tf.clip_by_value(newthdot, -self.max_speed, self.max_speed)
        new_state = tf.concat([tf.expand_dims(tf.math.cos(newth), -1),
                          tf.expand_dims(tf.math.sin(newth), -1),
                          tf.expand_dims(newthdot, -1)], axis=1)
        deviation = new_state - x[:, :3]
        return deviation

    @tf.function
    def get_loss(self, expected_output, predictions):
        """
        This is the training loss function for the pendulum true model.


        Parameters
        ---------
        expected_output: tf.float32
            Defines the next state (s_t+1) ground truth with (dims = Batch X dim_S)
        predictions: tf.float32
            Defines the next state (s_t+1) predicted values with (dims = Batch X dim_S)


        Returns
        -------
        train_loss: tf.float32
            Defines the training loss as a scalar for the whole batch
        """
        return tf.constant(0, dtype=tf.float32)

    @tf.function
    def get_validation_loss(self, expected_output, predictions):
        """
        This is the validation loss function for the pendulum true model.


        Parameters
        ---------
        expected_output: tf.float32
            Defines the next state (s_t+1) ground truth with (dims = Batch X dim_S)
        predictions: tf.float32
            Defines the next state (s_t+1) predicted values with (dims = Batch X dim_S)


        Returns
        -------
        validation_loss: tf.float32
            Defines the validation loss as a scalar for the whole batch
        """
        return tf.constant(0, dtype=tf.float32)