"""
tf_neuralmpc/dynamics_handlers/system_dynamics_handler.py
==========================================================
"""
import numpy as np
import tensorflow as tf
import logging


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


class SystemDynamicsHandler(object):
    """This is the system dynamics handler class that is reponsible for training the dynamics functions
    , storing the rollouts as well as prepocessing and postprocessing of the MDP elements"""
    def __init__(self, dim_O, dim_U, dynamics_function=None, num_of_agents=1,
                 log_dir=None, tf_writer=None, saved_model_dir=None,
                 transform_targets_func=default_transform_targets,
                 inverse_transform_targets_func=default_inverse_transform_targets,
                 save_model_frequency=1, load_saved_model=False,
                 true_model=False, normalization=True):
        """
        This is the initializer function for the system dynamics handler class.


        Parameters
        ----------
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        dim_U: tf.int32
            Defines the dimensions of the input/ action space.
        dim_O: tf.int32
            Defines the dimensions of the observations space.
        tf_writer: tf.summary
            Defines a tensorflow writer to be used for logging
        log_dir: string
            Defines the log directory to save the normalization statistics in.
        saved_model_dir: string
            Defines the saved model directory where the model is saved in, in case of loading the model.
        num_of_agents: Int
            Defines the number of runner running in parallel
        dynamics_function: DeterministicDynamicsFunctionBase
            Defines the dynamics_function of the nn dynamics function itself
        transform_targets_func: tf_function
            Defines a tf function to transform the next states as targets (output of the nn dynamics), by default
            this is the deviation function which is (target = next_state - current_state).
        inverse_transform_targets_func: tf_function
            Defines a tf function to inverse transform the targets (output of the nn dynamics), by default
            this is the inverse of the deviation function which is (next_state = target + current_state).
        save_model_frequency: Int
            Defines how often the model should be saved (defined relative to the number of refining iters)
        load_saved_model: bool
            Defines if a model should be loaded or not.
        true_model: bool
            Defines if the dynamics function is a non trainable model or not.
        normalization: bool
            Defines if the dynamics function should be trained with normalization or not.
        """
        self.num_of_agents = num_of_agents
        self.is_true_model = true_model
        self.dim_O = dim_O
        self.dim_U = dim_U
        self.dynamics_function = dynamics_function
        self.save_model_frequency = save_model_frequency
        self.log_dir = log_dir
        if self.log_dir is not None and tf_writer is None:
            self.tf_writer = tf.summary.create_file_writer(self.log_dir)
        else:
            self.tf_writer = tf_writer
        self.load_saved_model = load_saved_model
        self.saved_model_dir = saved_model_dir
        self.normalization = normalization
        self.transform_targets = transform_targets_func
        self.inverse_transform_targets = inverse_transform_targets_func
        self.model_training_in = np.array([], dtype=np.float32).reshape(0, self.dim_U + self.dim_O)
        self.model_validation_in = np.array([], dtype=np.float32).reshape(0, self.dim_U + self.dim_O)
        self.model_training_out = np.array([], dtype=np.float32).reshape(0, self.dim_O)
        self.model_validation_out = np.array([], dtype=np.float32).reshape(0, self.dim_O)
        self.training_iter = 0
        self.refining_model_iter = 0
        self.first_time = True
        if self.load_saved_model:
            logging.info("Loading the saved model now....")
            if self.saved_model_dir is None:
                raise Exception("you need to provide a saved model dir")
            self.dynamics_function = tf.saved_model.load(self.saved_model_dir)
            #load the stats and set first time to false
            self.first_time = False
            if self.normalization:
                self.mean_states = np.load(self.saved_model_dir + '/mean_states.npy')
                self.std_states = np.load(self.saved_model_dir + '/std_states.npy')
                self.mean_actions = np.load(self.saved_model_dir + '/mean_actions.npy')
                self.std_actions = np.load(self.saved_model_dir + '/std_actions.npy')
                self.mean_targets = np.load(self.saved_model_dir + '/mean_targets.npy')
                self.std_targets = np.load(self.saved_model_dir + '/std_targets.npy')

    def process_input(self, states, actions):
        """
        This is the process_input function, which takes in the states and actions and preprocesses them for the dynamics
        function, (normalization..etc)

        Parameters
        ---------
        states: tf.float32
            The current states has a shape of (Batch Xdim_S)
        actions: tf.float32
            The current actions has a shape of (Batch Xdim_U)

        Returns
        -------
        result: tf.float32
            concatenated states and actions after preprocessing them.
        """
        if self.is_true_model:
            return tf.concat([states, actions], axis=-1)
        else:
            return self._process_input(states, actions)

    @tf.function
    def _process_input(self, states, actions):
        if self.normalization:
            new_states = (states - self.mean_states) / (self.std_states + 1e-7)
            new_actions = (actions - self.mean_actions) / (self.std_actions + 1e-7)
            return tf.concat([new_states, new_actions], axis=-1)
        else:
            return tf.concat([states, actions], axis=-1)

    def process_state_output(self, old_states, normalized_states_deviation):
        """
        This is the process_state_output function, which takes in the previous states predicted target/delta and processes
        them to get the predicted absolute next state.

        Parameters
        ---------
        old_states: tf.float32
            The previous states has a shape of (Batch Xdim_S)
        normalized_states_deviation: tf.float32
            The predicted normalized delta as received from the dynamics function has a shape of (Batch Xdim_U).

        Returns
        -------
        result: tf.float32
            absolute predicted next state.
        """
        if self.is_true_model:
            new_states_transformed = self.inverse_transform_targets(old_states, normalized_states_deviation)
            return new_states_transformed
        else:
            return self._process_state_output(old_states, normalized_states_deviation)

    @tf.function
    def _process_state_output(self, old_states, normalized_states_deviation):
        if self.normalization:
            new_states_deviation = self.mean_targets + (normalized_states_deviation * (self.std_targets + 1e-7))
        else:
            new_states_deviation = normalized_states_deviation
        new_states = self.inverse_transform_targets(old_states, new_states_deviation)
        return new_states

    def train(self, observations_trajectories, actions_trajectories,
              rewards_trajectories, validation_split=0.2,
              batch_size=128, learning_rate=1e-3, epochs=30,
              nn_optimizer=tf.keras.optimizers.Adam):
        """
        This is the train function, which takes in the data of the MDP to train the dynamics model on it.

        Parameters
        ---------
        observations_trajectories: [np.float32]
            A list of observations of each of the episodes for the n agents.
        actions_trajectories: [np.float32]
            A list of actions of each of the episodes for the n agents.
        rewards_trajectories: [np.float32]
            A list of rewards of each of the episodes for the n agents.
        learning_rate: float
            Learning rate to be used in training the dynamics function.
        epochs: Int
            Number of epochs to be used in training the dynamics function everytime train is called.
        validation_split: float32
            Defines the validation split to be used of the rollouts collected.
        batch_size: int
            Defines the batch size to be used for training the model.
        nn_optimizer: tf.keras.optimizers
            Defines the optimizer to use with the neural network.
        """
        self._append_to_training_dataset(observations_trajectories, actions_trajectories,
                                         rewards_trajectories, validation_split=validation_split)
        if self.first_time: #TODO: for some reason this is needed
            if self.normalization:
                self._compute_normalization()
                self.first_time = False
            else:
                self.first_time = False
        model_training_in, model_training_out = self._normalize_data(self.model_training_in, self.model_training_out)
        train_dataset = tf.data.Dataset.from_tensor_slices((model_training_in,
                                                            model_training_out)).\
            shuffle(model_training_in.shape[0]).batch(batch_size, drop_remainder=True)
        model_validation_in, model_validation_out = self._normalize_data(self.model_validation_in,
                                                                         self.model_validation_out)
        validation_dataset = tf.data.Dataset.from_tensor_slices((model_validation_in,
                                                                 model_validation_out)).\
            batch(batch_size, drop_remainder=True)
        logging.info("Started the system training")
        self._training_algorithm(train_dataset, validation_dataset,
                                 learning_rate=learning_rate, epochs=epochs,
                                 nn_optimizer=nn_optimizer)
        self.training_iter += 1
        #TODO: just commented for now because of a bug in TF 2.0.0
        if self.training_iter%self.save_model_frequency == 0 and self.log_dir is not None:
            logging.info("Saving the model now....")
            call = self.dynamics_function.__call__.get_concrete_function(tf.TensorSpec([None, self.dim_U + self.dim_O],
                                                                                    tf.float32),
                                                                         tf.TensorSpec([],
                                                                                       tf.bool)
                                                                         )
            tf.saved_model.save(self.dynamics_function, self.log_dir + '/saved_model/', signatures=call)
            #save the means and std devs
            np.save(self.log_dir + "/saved_model/mean_states", self.mean_states)
            np.save(self.log_dir + "/saved_model/std_states", self.std_states)
            np.save(self.log_dir + "/saved_model/mean_actions", self.mean_actions)
            np.save(self.log_dir + "/saved_model/std_actions", self.std_actions)
            np.save(self.log_dir + "/saved_model/mean_targets", self.mean_targets)
            np.save(self.log_dir + "/saved_model/std_targets", self.std_targets)
        return

    def _training_algorithm(self, train_dataset, validation_dataset,
                            learning_rate=1e-3, epochs=30,
                            nn_optimizer=tf.keras.optimizers.Adam):
        """
        This is the train_model function for the dynamics trainer base class.


        Parameters
        ---------
        train_dataset: tf.data.Dataset
            training dataset that is composed of (input, expected_output) after processing.
        validation_dataset: tf.data.Dataset
            validation dataset that is composed of (input, expected_output) after processing.
        """
        optimizer = nn_optimizer(learning_rate=learning_rate)
        training_loss = np.zeros(epochs)
        validation_loss = np.zeros(epochs)
        for i in range(0, epochs):  # run through epochs
            average_loss = 0
            number_of_batches = 0
            for x, y in train_dataset:
                number_of_batches += 1
                with tf.GradientTape() as tape:
                    prediction = self.dynamics_function(x, train=tf.constant(True, dtype=tf.bool))
                    loss = self.dynamics_function.get_loss(expected_output=y, predictions=prediction)
                    average_loss += loss
                trainable_variables = tape.watched_variables()
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

            training_loss[i] = average_loss / number_of_batches
            average_loss = 0
            number_of_batches = 0
            for x, y in validation_dataset:
                number_of_batches += 1
                prediction = self.dynamics_function(x, train=tf.constant(False, dtype=tf.bool))
                loss = self.dynamics_function.get_validation_loss(expected_output=y, predictions=prediction)
                average_loss += loss
            validation_loss[i] = average_loss / number_of_batches
            if self.tf_writer is not None:
                with self.tf_writer.as_default():
                    tf.summary.scalar('system_model_val/loss', validation_loss[i],
                                      step=(self.refining_model_iter * epochs) + i)
        self.refining_model_iter += 1

    def _append_to_training_dataset(self, observations_trajectories, actions_trajectories,
                                    rewards_trajectories, validation_split=0.2):
        new_data_in, new_data_targs = [], []
        for obs, acs in zip(observations_trajectories, actions_trajectories):
            #switch number of runner with traj length
            for agent in range(self.num_of_agents):
                states = obs[:-1, agent]
                actions = acs[:, agent]
                new_data_in.append(np.concatenate([states, actions], axis=-1))
                next_states = obs[1:, agent]
                new_data_targs.append(self.transform_targets(states, next_states).numpy())
        #now split data into val and train
        new_data_in = np.array(new_data_in, dtype=np.float32).reshape(-1, self.dim_U + self.dim_O)
        new_data_targs = np.array(new_data_targs, dtype=np.float32).reshape(-1, self.dim_O)
        training_indicies = np.random.choice([False, True],
                                             size=new_data_in.shape[0],
                                             p=[validation_split, 1.0 - validation_split])
        new_train_in = new_data_in[training_indicies]
        new_train_targs = new_data_targs[training_indicies]
        new_validation_in = new_data_in[~training_indicies]
        new_validation_targs = new_data_targs[~training_indicies]
        N_train = new_train_in.shape[0]
        N_val = new_validation_in.shape[0]
        self.model_training_in = np.concatenate([self.model_training_in, new_train_in], axis=0)
        self.model_training_out = np.concatenate([self.model_training_out, new_train_targs], axis=0)
        self.model_validation_in = np.concatenate([self.model_validation_in, new_validation_in], axis=0)
        self.model_validation_out = np.concatenate([self.model_validation_out, new_validation_targs], axis=0)
        return

    def _normalize_data(self, data_in, data_out):
        states_normalized = (data_in[:, :self.dim_O] - self.mean_states) / (self.std_states + 1e-7)
        actions_normalized = (data_in[:, self.dim_O:] - self.mean_actions) / (self.std_actions + 1e-7)
        targets_normalized = (data_out - self.mean_targets) / (self.std_targets + 1e-7)
        return np.concatenate([states_normalized, actions_normalized], axis=1), targets_normalized

    def _compute_normalization(self):
        self.mean_states = np.mean(self.model_training_in[:, :self.dim_O], axis=0)
        self.std_states = np.std(self.model_training_in[:, :self.dim_O], axis=0)

        self.mean_actions = np.mean(self.model_training_in[:, self.dim_O:], axis=0)
        self.std_actions = np.std(self.model_training_in[:, self.dim_O:], axis=0)

        self.mean_targets = np.mean(self.model_training_out, axis=0)
        self.std_targets = np.std(self.model_training_out, axis=0)
        return
