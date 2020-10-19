import tensorflow as tf


# #@tf.function
def reward_function(current_state, actions, next_state):
    heading_penalty_factor = -10
    rewards = tf.zeros((current_state.shape[0],), dtype=tf.float32)
    front_leg = current_state[:, 5]
    my_range = tf.constant(0.2, dtype=tf.float32)
    rewards = tf.where(front_leg >= my_range, x=rewards + heading_penalty_factor, y=rewards)

    front_shin = current_state[:, 6]
    my_range = tf.constant(0, dtype=tf.float32)
    rewards = tf.where(front_shin >= my_range, x=rewards + heading_penalty_factor, y=rewards)

    front_foot = current_state[:, 7]
    my_range = tf.constant(0, dtype=tf.float32)
    rewards = tf.where(front_foot >= my_range, x=rewards + heading_penalty_factor, y=rewards)

    rewards = rewards + ((next_state[:, 17] - current_state[:, 17]) / 0.01)
    rewards = rewards  -(tf.constant(0.0, dtype=tf.float32) * tf.reduce_sum(tf.square(actions), axis=1))
    return rewards
