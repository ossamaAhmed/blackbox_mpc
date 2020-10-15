import tensorflow as tf


@tf.function
def default_transform_targets(current_state, next_state):
    """
    This is the default transform targets function used, which preprocesses the targets of the network before training the
    dynamics function using the inputs and targets. The default one is (target = next_state - current_state).

    Parameters
    ---------
    current_state: tf.float32
        The current_state has a shape of (Batch X dim_S)
    next_state: tf.float32
        The next_state has a shape of (Batch X dim_S)
    """
    return next_state - current_state


@tf.function
def default_inverse_transform_targets(current_state, delta):
    """
    This is the default inverse transform targets function used, which reverses the preprocessing of  the targets of
    the dynamics function to obtain the real current_state not the relative one,
    The default one is (current_state = target + current_state).

    Parameters
    ---------
    current_state: tf.float32
        The current_state has a shape of (Batch X dim_S)
    delta: tf.float32
        The delta has a shape of (Batch X dim_S) which is equivilant to the target of the network.
    """
    return delta + current_state
