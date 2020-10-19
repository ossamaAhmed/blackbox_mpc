import tensorflow as tf
from blackbox_mpc.trajectory_evaluators.evaluator_base import EvaluatorBase


class DeterministicTrajectoryEvaluator(EvaluatorBase):
    def __init__(self, reward_function, system_dynamics_handler):
        """
        This is the trajectory evaluator class for a deterministic dynamics function


        Parameters
        ---------
        reward_function: tf_function
            Defines the reward function with the prototype: tf_func_name(current_state, current_actions, next_state),
            where current_state is BatchXdim_S, next_state is BatchXdim_S and  current_actions is BatchXdim_U.
        system_dynamics_handler: SystemDynamicsHandler
            Defines the system dynamics handler class with its own trainer and observations and actions
             preprocessing functions.
        """
        super(DeterministicTrajectoryEvaluator, self).__init__(
            reward_function=reward_function,
            system_dynamics_handler=system_dynamics_handler,
            name=None)
        return

    @tf.function
    def __call__(self, current_states, action_sequences, time_step):
        """
          This is the call function for the Deterministic Trajectory Evaluator Class.
          It is used to calculate the rewards corresponding to each of the action sequences starting
          from the current state.

          Parameters
          ---------
          current_states: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          action_sequences: tf.float32
             Defines the action sequences to be evaluated, (dims = population X num_of_agents X planning_horizon X dim_U)
          time_step: tf.float32
              Defines the current timestep of the episode.


          Returns
          -------
          rewards: tf.float32
              The rewards corresponding to each action sequence (dims = 1 X population)
          """
        init_t = tf.constant(0, dtype=tf.int32)
        nopt = tf.shape(action_sequences)[0]
        n_agents = tf.shape(action_sequences)[1]
        planning_horizon = tf.shape(action_sequences)[2]
        init_rewards = tf.zeros([nopt*n_agents], dtype=tf.float32)
        action_sequences = tf.reshape(action_sequences,
                                      [-1, planning_horizon,
                                       tf.shape(action_sequences)[3]])
        action_sequences = tf.transpose(action_sequences, [1, 0, 2])
        init_states = tf.tile(current_states, [nopt, 1])

        def continue_prediction(t, total_reward, current_state):
            return tf.less(t, planning_horizon)

        def iterate(t, total_reward, current_state):
            current_actions = action_sequences[t]
            next_state = self.predict_next_state(current_state, current_actions)
            delta_reward = self._reward_function(current_state,
                                                 current_actions, next_state)
            return t + tf.constant(1, dtype=tf.int32), \
                   total_reward + delta_reward, next_state

        _, rewards, _ = tf.while_loop(
            cond=continue_prediction, body=iterate, loop_vars=[init_t, init_rewards,
                                                               init_states]
        )
        rewards = tf.reshape(rewards, [nopt, n_agents])
        return tf.where(tf.math.is_nan(rewards),
                        tf.constant(-1e6, dtype=tf.float32) *
                        tf.ones_like(rewards), rewards)

    @tf.function
    def predict_next_state(self, current_states, current_actions):
        """
          This is the function used to predict the next state using the internal dynamics handler.

          Parameters
          ---------
          current_states: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          current_actions: tf.float32
             Defines the current action to be applied, (dims = num_of_agents X dim_U)


          Returns
          -------
          next_state: tf.float32
              Defines the next state of the system, (dims=num_of_agents X dim_S)
          """
        sys_model_inputs = self._system_dynamics_handler.process_input(
            current_states, current_actions)
        raw_next_states = self._system_dynamics_handler._dynamics_function(
            sys_model_inputs, train=tf.constant(False, dtype=tf.bool))
        next_states = self._system_dynamics_handler.process_output(
            current_states, raw_next_states)
        return next_states

    def evaluate_next_reward(self, current_states,
                             next_states, current_actions):
        """
          This is the function used to predict the next reward using the internal dynamics handler.

          Parameters
          ---------
          current_states: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          next_states: tf.float32
              Defines the next state of the system, (dims=num_of_agents X dim_S)
          current_actions: tf.float32
             Defines the current action to be applied, (dims = num_of_agents X dim_U)


          Returns
          -------
          reward: tf.float32
              returns the predicted reward using the action, current state and the next one,
              (dims=num_of_agents X 1)
          """
        return self._reward_function(current_states, current_actions,
                                     next_states)
