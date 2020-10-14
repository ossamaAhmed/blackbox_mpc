"""
tf_neuralmpc/dynamics_functions/deterministic_models/deterministic_dynamics_function_base.py
=============================================================================================
"""
import tensorflow as tf


class DeterministicDynamicsFunctionBase(tf.Module):
    """This is the deterministic dynamics function base class for (s_t, a_t) - > (s_t+1)"""
    def __init__(self, name=None):
        """
        This is the initializer function for the deterministic dynamics function base class.


        Parameters
        ---------
        name: String
            Defines the name of the block of the deterministic dynamics function.
        """
        super(DeterministicDynamicsFunctionBase, self).__init__(name=name)
        return

    def __call__(self, x, train):
        """
        This is the call function for the deterministic dynamics function base class.


        Parameters
        ---------
        x: tf.float32
            Defines the (s_t, a_t) which is the state and action stacked on top of each other,
            (dims = Batch X (dim_S + dim_U))
        train: tf.bool
            Defines whether the network should run in train mode or not.


        Returns
        -------
        output: tf.float32
            Defines the next state (s_t+1) with (dims = Batch X dim_S)
        """
        raise Exception("__call__ function is not implemented yet")

    def get_loss(self, expected_output, predictions):
        """
        This is the training loss function for the deterministic dynamics function base class.


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
        raise Exception("get_loss function is not implemented yet")

    def get_validation_loss(self, expected_output, predictions):
        """
        This is the validation loss function for the deterministic dynamics function base class.


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
        raise Exception("get_validation_loss function is not implemented yet")

