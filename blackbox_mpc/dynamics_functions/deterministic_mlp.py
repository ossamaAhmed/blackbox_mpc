import tensorflow as tf


class DeterministicMLP(tf.Module):
    def __init__(self, layers, activation_functions,
                 loss_fn=tf.keras.losses.MeanSquaredError(),
                 name=None):
        """
        A deterministic fully connected MLP dynamics function class for
        (s_t, a_t) - > (s_t+1)

        Parameters
        ---------
        name: str
            Defines the name of the block of the deterministic MLP dynamics
            function.
        """
        super(DeterministicMLP, self).__init__(name=name)
        self.layers = []
        for i in range(1, len(layers)):
            self.layers.append(
                tf.keras.layers.Dense(layers[i], input_shape=(layers[i-1],),
                                      activation=activation_functions[i-1],
                                      dtype='float32'))
        self.loss_fn = loss_fn

    @tf.function
    def __call__(self, x, train):
        """
        This is the call function for the deterministic fully connected MLP
        dynamics function class.


        Parameters
        ---------
        x: tf.float32
            Defines the (s_t, a_t) which is the state and action stacked on top
            of each other, (dims = Batch X (dim_S + dim_U))

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
        This is the training loss function for the deterministic fully
        connected MLP dynamics function class.


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
        This is the validation loss function for the deterministic fully
        connected MLP dynamics function class.


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
