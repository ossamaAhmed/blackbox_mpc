import tensorflow as tf


class EvaluatorBase(tf.Module):
    """This is the base class of the trajectory evaluators"""
    def __init__(self, reward_function,
                 system_dynamics_handler,
                 name=None):
        """
        This is the initializer function for the Evaluator Base Class.


        Parameters
        ---------
        name: String
            Defines the name of the block of the evaluator.
        """
        super(EvaluatorBase, self).__init__(name=name)
        self._reward_function = reward_function
        self._system_dynamics_handler = system_dynamics_handler

    @tf.function
    def __call__(self, current_states, action_sequences, time_step):
        """
          This is the call function for the Evaluator Base Class.
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
        raise Exception("__call__ function is not implemented yet")

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
        raise Exception("predict_next_state function is not implemented yet")

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
        raise Exception("evaluate_next_reward function is not implemented yet")
