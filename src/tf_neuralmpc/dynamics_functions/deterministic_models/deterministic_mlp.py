"""
tf_neuralmpc/dynamics_functions/deterministic_models/deterministic_mlp.py
=========================================================================
"""
import tensorflow as tf
from tf_neuralmpc.dynamics_functions.deterministic_models.deterministic_dynamics_function_base \
    import DeterministicDynamicsFunctionBase


class DeterministicMLP(DeterministicDynamicsFunctionBase):
    """This is the deterministic fully connected MLP dynamics function class for (s_t, a_t) - > (s_t+1)"""
    def __init__(self, name=None):
        """
        This is the initializer function for the deterministic fully connected MLP dynamics function class.


        Parameters
        ---------
        name: String
            Defines the name of the block of the deterministic MLP dynamics function.
        """
        super(DeterministicMLP, self).__init__(name=name)
        self.finalized = False
        self.layers = []
        self.decays = []
        self.optvars = []
        self.nonoptvars = []
        self.optimizer = None
        self.layers = []
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def add_layer(self, input_dim, output_dim, activation_function=None):
        """
        This is the add layer function for the deterministic fully connected MLP dynamics function class, which
        is used to add a new layer at the end of the so far constructed MLP.


        Parameters
        ---------
        input_dim: Int
            Defines the input dimensions of the layer.
        output_dim: Int
            Defines the output dimensions of the layer.
        activation_function: tf.nn
            Defines the activation non-linearity to be used for this layer.
        """
        self.layers.append(tf.keras.layers.Dense(output_dim, input_shape=(input_dim,), activation=activation_function,
                                                 dtype='float32'))
        return

    @tf.function
    def __call__(self, x, train):
        """
        This is the call function for the deterministic fully connected MLP dynamics function class.


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
        for layer in self.layers.layers:
            x = layer(x)
        return x

    @tf.function
    def get_loss(self, expected_output, predictions):
        """
        This is the training loss function for the deterministic fully connected MLP dynamics function class.


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
        return self.loss_fn(expected_output, predictions)

    @tf.function
    def get_validation_loss(self, expected_output, predictions):
        """
        This is the validation loss function for the deterministic fully connected MLP dynamics function class.


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
        return self.loss_fn(expected_output, predictions)
