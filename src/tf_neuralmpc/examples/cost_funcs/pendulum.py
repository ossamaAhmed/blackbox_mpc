"""
tf_neuralmpc/examples/cost_funcs/pendulum.py
============================================
"""

import tensorflow as tf
import numpy as np


@tf.function
def _pendulum_angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)


@tf.function
def pendulum_state_reward_function(current_state, next_state):
    """
        The pendulum state reward function

       Parameters
       ---------
       current_state: tf.float32
            represents the current state of the system (Bxdim_S)
       next_state: tf.float32
            represents the next state of the system (Bxdim_S)

       Returns
        -------
        rewards: tf.float32
            The reward corresponding to each of the pairs current_state, next_state
       """
    return -((_pendulum_angle_normalize(tf.math.atan2(current_state[:, 1], current_state[:, 0]))
              ** tf.constant(2, dtype=tf.float32)) + tf.constant(0.1, dtype=tf.float32) * current_state[:, 2]
             ** tf.constant(2, dtype=tf.float32))


@tf.function
def pendulum_actions_reward_function(actions):
    """
       The pendulum actions reward function

      Parameters
      ---------
      actions: tf.float32
           represents the current actions applied to the system (Bxdim_U)

      Returns
       -------
       rewards: tf.float32
           The reward corresponding to each of the actions.
      """
    return -(tf.constant(0.001, dtype=tf.float32) * tf.reduce_sum(tf.square(actions), axis=1))


