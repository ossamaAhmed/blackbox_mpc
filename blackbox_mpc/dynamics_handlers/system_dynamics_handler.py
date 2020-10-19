import numpy as np
import tensorflow as tf
import logging
from blackbox_mpc.utils.transforms import default_transform_targets, \
    default_inverse_transform_targets


class SystemDynamicsHandler(object):
    def __init__(self, env_action_space, env_observation_space,
                 dynamics_function=None, true_model=False,
                 is_normalized=True,
                 log_dir=None, tf_writer=None,
                 save_model_frequency=1,
                 saved_model_dir=None,
                 transform_targets_func=default_transform_targets,
                 inverse_transform_targets_func=
                 default_inverse_transform_targets):
        """
        This is the system dynamics handler class that is reponsible
        for training the dynamics functions, storing the rollouts as well as
         prepocessing and postprocessing of the MDP elements


        Parameters
        ----------
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
        env_observation_space: tf.int32
            Defines the observation space of the gym environment.
        tf_writer: tf.summary
            Defines a tensorflow writer to be used for logging
        log_dir: string
            Defines the log directory to save the normalization statistics in.
        saved_model_dir: string
            Defines the saved model directory where the model is saved in, in
            case of loading the model.
        dynamics_function: DeterministicDynamicsFunctionBase
            Defines the dynamics_function of the nn dynamics function itself
        transform_targets_func: tf_function
            Defines a tf function to transform the next states as targets
            (output of the nn dynamics), by default
            this is the deviation function which is
            (target = next_state - current_state).
        inverse_transform_targets_func: tf_function
            Defines a tf function to inverse transform the targets
            (output of the nn dynamics), by default
            this is the inverse of the deviation function which is
            (next_state = target + current_state).
        save_model_frequency: Int
            Defines how often the model should be saved (defined relative to
            the number of refining iters)
        true_model: bool
            Defines if the dynamics function is a non trainable model or not.
        is_normalized: bool
            Defines if the dynamics function should be trained with
            normalization or not.
        """
        self._is_true_model = true_model
        self._dim_S = tf.constant(env_observation_space.shape[0], dtype=tf.int32)
        self._dim_U = tf.constant(env_action_space.shape[0], dtype=tf.int32)
        self._dynamics_function = dynamics_function
        self._save_model_frequency = save_model_frequency
        self._log_dir = log_dir
        self._tf_writer = tf_writer
        self._saved_model_dir = saved_model_dir
        self._is_normalized = is_normalized
        self._transform_targets = transform_targets_func
        self._inverse_transform_targets = inverse_transform_targets_func
        self._model_training_in = np.array([], dtype=np.float32).reshape(0, self._dim_U + self._dim_S)
        self._model_validation_in = np.array([], dtype=np.float32).reshape(0, self._dim_U + self._dim_S)
        self._model_training_out = np.array([], dtype=np.float32).reshape(0, self._dim_S)
        self._model_validation_out = np.array([], dtype=np.float32).reshape(0, self._dim_S)
        self._training_iter = 0
        self._refining_model_iter = 0
        self._first_time = True
        if self._saved_model_dir is not None:
            logging.info("Loading the saved model now....")
            self._dynamics_function = tf.saved_model.load(self._saved_model_dir)
            #load the stats and set first time to false
            self._first_time = False
            if self._is_normalized:
                self._mean_states = np.load(self._saved_model_dir +
                                            '/mean_states.npy')
                self._std_states = np.load(self._saved_model_dir +
                                           '/std_states.npy')
                self._mean_actions = np.load(self._saved_model_dir +
                                             '/mean_actions.npy')
                self._std_actions = np.load(self._saved_model_dir +
                                            '/std_actions.npy')
                self._mean_targets = np.load(self._saved_model_dir +
                                             '/mean_targets.npy')
                self._std_targets = np.load(self._saved_model_dir +
                                            '/std_targets.npy')

    @tf.function
    def process_input(self, states, actions):
        """
        This is the process_input function, which takes in the states and
        actions and preprocesses them for the dynamics
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
        if self._is_true_model:
            return tf.concat([states, actions], axis=-1)
        else:
            if self._is_normalized:
                new_states = (states - self._mean_states) / (
                            self._std_states + 1e-7)
                new_actions = (actions - self._mean_actions) / (
                            self._std_actions + 1e-7)
                return tf.concat([new_states, new_actions], axis=-1)
            else:
                return tf.concat([states, actions], axis=-1)

    @tf.function
    def process_output(self, inputs_states, raw_output):
        """
        This is the process_state_output function, which takes in the
        previous states predicted target/delta and processes
        them to get the predicted absolute next state.

        Parameters
        ---------
        inputs_states: tf.float32
            The previous states has a shape of (Batch Xdim_S)
        raw_output: tf.float32
            The predicted normalized delta as received from the dynamics function
            has a shape of (Batch Xdim_U).

        Returns
        -------
        result: tf.float32
            absolute predicted next state.
        """
        if self._is_true_model:
            new_states_transformed = \
                self._inverse_transform_targets(inputs_states, raw_output)
            return new_states_transformed
        else:
            if self._is_normalized:
                new_states_deviation = self._mean_targets + (
                        raw_output * (
                                self._std_targets + 1e-7))
            else:
                new_states_deviation = raw_output
            new_states = self._inverse_transform_targets(inputs_states,
                                                         new_states_deviation)
            return new_states

    def train(self, observations_trajectories, actions_trajectories,
              rewards_trajectories, validation_split=0.2,
              batch_size=128, learning_rate=1e-3, epochs=30,
              nn_optimizer=tf.keras.optimizers.Adam):
        """
        This is the train function, which takes in the data of the MDP to
        train the dynamics model on it.

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
            Number of epochs to be used in training the dynamics function
            everytime train is called.
        validation_split: float32
            Defines the validation split to be used of the rollouts collected.
        batch_size: int
            Defines the batch size to be used for training the model.
        nn_optimizer: tf.keras.optimizers
            Defines the optimizer to use with the neural network.
        """
        self._append_to_training_dataset(observations_trajectories, actions_trajectories,
                                         rewards_trajectories, validation_split=validation_split)
        if self._first_time: #TODO: for some reason this is needed
            if self._is_normalized:
                self._recompute_normalization()
                self._first_time = False
            else:
                self._first_time = False
        model_training_in, model_training_out = self._normalize_data(self._model_training_in, self._model_training_out)
        train_dataset = tf.data.Dataset.from_tensor_slices((model_training_in,
                                                            model_training_out)).\
            shuffle(model_training_in.shape[0]).batch(batch_size, drop_remainder=True)
        model_validation_in, model_validation_out = self._normalize_data(self._model_validation_in,
                                                                         self._model_validation_out)
        validation_dataset = tf.data.Dataset.from_tensor_slices((model_validation_in,
                                                                 model_validation_out)).\
            batch(batch_size, drop_remainder=True)
        logging.info("Started the system training")
        self._training_algorithm(train_dataset, validation_dataset,
                                 learning_rate=learning_rate, epochs=epochs,
                                 nn_optimizer=nn_optimizer)
        self._training_iter += 1
        if self._training_iter % self._save_model_frequency == 0 and \
                self._log_dir is not None:
            logging.info("Saving the model now....")
            call = self._dynamics_function.__call__.get_concrete_function(
                tf.TensorSpec([None, self._dim_U + self._dim_S], tf.float32),
                tf.TensorSpec([], tf.bool))
            tf.saved_model.save(self._dynamics_function, self._log_dir +
                                '/saved_model_' +
                                str(self._refining_model_iter) + '/',
                                signatures=call)
            #save the means and std devs
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/mean_states',
                    self._mean_states)
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/std_states',
                    self._std_states)
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/mean_actions',
                    self._mean_actions)
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/std_actions',
                    self._std_actions)
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/mean_targets',
                    self._mean_targets)
            np.save(self._log_dir + '/saved_model_' +
                                str(self._refining_model_iter) + '/std_targets',
                    self._std_targets)
        logging.info("Ended the system training")
        return

    def _training_algorithm(self, train_dataset, validation_dataset,
                            learning_rate=1e-3, epochs=30,
                            nn_optimizer=tf.keras.optimizers.Adam):
        """
        This is the train_model function for the dynamics trainer base class.


        Parameters
        ---------
        train_dataset: tf.data.Dataset
            training dataset that is composed of (input, expected_output)
            after processing.
        validation_dataset: tf.data.Dataset
            validation dataset that is composed of (input, expected_output)
            after processing.
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
                    prediction = \
                        self._dynamics_function(x,
                                                train=tf.constant(
                                                    True,
                                                    dtype=tf.bool))
                    loss = \
                        self._dynamics_function.get_loss(expected_output=y,
                                                         predictions=prediction)
                    average_loss += loss
                trainable_variables = tape.watched_variables()
                gradients = tape.gradient(loss, trainable_variables)
                optimizer.apply_gradients(zip(gradients, trainable_variables))

            training_loss[i] = average_loss / number_of_batches
            average_loss = 0
            number_of_batches = 0
            for x, y in validation_dataset:
                number_of_batches += 1
                prediction = self._dynamics_function(x,
                                                     train=tf.constant(False, dtype=tf.bool))
                loss = self._dynamics_function.get_validation_loss(
                    expected_output=y, predictions=prediction)
                average_loss += loss
            validation_loss[i] = average_loss / number_of_batches
            if self._tf_writer is not None:
                with self._tf_writer.as_default():
                    tf.summary.scalar('system_model_val/loss', validation_loss[i],
                                      step=(self._refining_model_iter * epochs) + i)
        self._refining_model_iter += 1

    def _append_to_training_dataset(self, observations_trajectories,
                                    actions_trajectories,
                                    rewards_trajectories,
                                    validation_split=0.2):
        actions_trajectories = np.array(actions_trajectories)
        num_of_agents = actions_trajectories.shape[2]
        new_data_in, new_data_targs = [], []
        for obs, acs in zip(observations_trajectories, actions_trajectories):
            #switch number of runner with traj length
            for agent in range(num_of_agents):
                states = obs[:-1, agent]
                actions = acs[:, agent]
                new_data_in.append(np.concatenate([states, actions], axis=-1))
                next_states = obs[1:, agent]
                new_data_targs.append(self._transform_targets(states,
                                                              next_states).numpy())
        #now split data into val and train
        new_data_in = np.array(new_data_in, dtype=np.float32).reshape(-1, self._dim_U + self._dim_S)
        new_data_targs = np.array(new_data_targs, dtype=np.float32).reshape(-1, self._dim_S)
        training_indicies = np.random.choice([False, True],
                                             size=new_data_in.shape[0],
                                             p=[validation_split, 1.0 - validation_split])
        new_train_in = new_data_in[training_indicies]
        new_train_targs = new_data_targs[training_indicies]
        new_validation_in = new_data_in[~training_indicies]
        new_validation_targs = new_data_targs[~training_indicies]
        N_train = new_train_in.shape[0]
        N_val = new_validation_in.shape[0]
        self._model_training_in = np.concatenate([self._model_training_in, new_train_in], axis=0)
        self._model_training_out = np.concatenate([self._model_training_out, new_train_targs], axis=0)
        self._model_validation_in = np.concatenate([self._model_validation_in, new_validation_in], axis=0)
        self._model_validation_out = np.concatenate([self._model_validation_out, new_validation_targs], axis=0)
        return

    def _normalize_data(self, data_in, data_out):
        states_normalized = (data_in[:, :self._dim_S] - self._mean_states) / (self._std_states + 1e-7)
        actions_normalized = (data_in[:, self._dim_S:] - self._mean_actions) / (self._std_actions + 1e-7)
        targets_normalized = (data_out - self._mean_targets) / (self._std_targets + 1e-7)
        return np.concatenate([states_normalized, actions_normalized], axis=1), targets_normalized

    def _recompute_normalization(self):
        self._mean_states = np.mean(self._model_training_in[:, :self._dim_S], axis=0)
        self._std_states = np.std(self._model_training_in[:, :self._dim_S], axis=0)

        self._mean_actions = np.mean(self._model_training_in[:, self._dim_S:], axis=0)
        self._std_actions = np.std(self._model_training_in[:, self._dim_S:], axis=0)

        self._mean_targets = np.mean(self._model_training_out, axis=0)
        self._std_targets = np.std(self._model_training_out, axis=0)
        return

    def get_dynamics_function(self):
        """
        returns the dynamics function used by the system handler.

        :return:
        """
        return self._dynamics_function
