"""
tf_neuralmpc/trajectory_evaluator/deterministic.py
===================================================
"""
import tensorflow as tf
from tf_neuralmpc.trajectory_evaluators.evaluator_base import EvaluatorBase


class DeterministicTrajectoryEvaluator(EvaluatorBase):
    """This is the trajectory evaluator class for a deterministic dynamics function"""
    def __init__(self, state_reward_function, actions_reward_function, planning_horizon,
                 dim_U, dim_O, system_dynamics_handler):
        """
        This is the initializer function for the Deterministic Trajectory Evaluator Class.


        Parameters
        ---------
        state_reward_function: tf_function
            Defines the state reward function with the prototype: tf_func_name(current_state, next_state),
            where current_state is BatchXdim_S and next_state is BatchXdim_S.
        actions_reward_function: tf_function
            Defines the action reward function with the prototype: tf_func_name(current_actions),
            where current_actions is BatchXdim_U.
        planning_horizon: tf.int32
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        dim_U: tf.int32
            Defines the dimensions of the input/action space.
        dim_O: tf.int32
            Defines the dimensions of the observations space.
        system_dynamics_handler: SystemDynamicsHandler
            Defines the system dynamics handler class with its own trainer and observations and actions
             preprocessing functions.
        """
        super(DeterministicTrajectoryEvaluator, self).__init__(name=None)
        self.state_reward_function = state_reward_function
        self.actions_reward_function = actions_reward_function
        self.planning_horizon = planning_horizon
        self.system_dynamics_handler = system_dynamics_handler
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O

    @tf.function
    def __call__(self, current_states, action_sequences, time_step):
        """
          This is the call function for the Deterministic Trajectory Evaluator Class.
          It is used to calculate the rewards corresponding to each of the action sequences starting
          from the current state.

          Parameters
          ---------
          current_state: tf.float32
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
        init_rewards = tf.zeros([nopt*n_agents], dtype=tf.float32)
        action_sequences = tf.reshape(action_sequences,
                                      [-1, self.planning_horizon,
                                       self.dim_U])
        action_sequences = tf.transpose(action_sequences, [1, 0, 2])
        init_states = tf.tile(current_states, [nopt, 1])

        def continue_prediction(t, total_reward, current_state):
            return tf.less(t, self.planning_horizon)

        def iterate(t, total_reward, current_state):
            current_actions = action_sequences[t]
            next_state = self._predict_next_state(current_state, current_actions)
            delta_reward = self.state_reward_function(current_state, next_state) + \
                           self.actions_reward_function(current_actions)
            return t + tf.constant(1, dtype=tf.int32), total_reward + delta_reward, next_state

        _, rewards, _ = tf.while_loop(
            cond=continue_prediction, body=iterate, loop_vars=[init_t, init_rewards,
                                                               init_states]
        )
        #return rewards for each agent
        rewards = tf.reshape(rewards, [nopt, n_agents])
        return tf.where(tf.math.is_nan(rewards), tf.constant(-1e6, dtype=tf.float32) *
                        tf.ones_like(rewards), rewards)

    @tf.function
    def _predict_next_state(self, current_state, current_actions):
        # make sure its a feasible action
        current_actions = current_actions
        sys_model_inputs = self.system_dynamics_handler.process_input(current_state, current_actions)
        next_state_prediction_delta = self.system_dynamics_handler.dynamics_function(sys_model_inputs,
                                                                                     train=tf.constant(False, dtype=tf.bool))
        next_state = self.system_dynamics_handler.process_state_output(current_state, next_state_prediction_delta)
        return next_state

    def predict_next_state(self, current_state, current_action):
        """
          This is the function used to predict the next state using the internal dynamics handler.

          Parameters
          ---------
          current_state: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          current_action: tf.float32
             Defines the current action to be applied, (dims = num_of_agents X dim_U)


          Returns
          -------
          next_state: tf.float32
              Defines the next state of the system, (dims=num_of_agents X dim_S)
          """
        return self._predict_next_state(current_state, current_action)

    def evaluate_next_reward(self, current_state, next_state, current_action):
        """
          This is the function used to predict the next reward using the internal dynamics handler.

          Parameters
          ---------
          current_state: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          next_state: tf.float32
              Defines the next state of the system, (dims=num_of_agents X dim_S)
          current_action: tf.float32
             Defines the current action to be applied, (dims = num_of_agents X dim_U)


          Returns
          -------
          reward: tf.float32
              returns the predicted reward using the action, current state and the next one,
              (dims=num_of_agents X 1)
          """
        return self.state_reward_function(current_state, next_state) + \
               self.actions_reward_function(current_action)
