import tensorflow as tf
import numpy as np


class OptimizerBase(tf.Module):
    def __init__(self, name, planning_horizon, max_iterations, num_agents,
                 env_action_space, env_observation_space):
        """
        This is the base class of the optimizers


        Parameters
        ---------
        name: String
            Defines the name of the block of the optimizer.
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        max_iterations: tf.int32
            Defines the maximimum iterations for the CEM optimizer to refine its guess for the optimal solution.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
        env_observation_space: tf.int32
            Defines the observation space of the gym environment.
        """
        super(OptimizerBase, self).__init__(name=name)
        self._planning_horizon = planning_horizon
        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space
        self._dim_U = tf.constant(env_action_space.shape[0], dtype=tf.int32)
        self._dim_S = tf.constant(env_observation_space.shape[0], dtype=tf.int32)
        self._action_upper_bound = tf.constant(env_action_space.high,
                                               dtype=tf.float32)
        self._action_lower_bound = tf.constant(env_action_space.low,
                                               dtype=tf.float32)
        self._action_upper_bound_horizon = tf.tile(
            np.expand_dims(self._action_upper_bound, 0),
            [self._planning_horizon, 1])
        self._action_lower_bound_horizon = tf.tile(
            np.expand_dims(self._action_lower_bound, 0),
            [self._planning_horizon, 1])
        self._num_agents = num_agents
        self._max_iterations = max_iterations
        self._trajectory_evaluator = None
        self._exploration_variance = (np.square(self._action_lower_bound -
                                                self._action_upper_bound) /
                                      16) * 0.05
        self._exploration_mean = (self._action_upper_bound +
                                  self._action_lower_bound) / 2

    def _optimize(self, current_state, time_step):
        raise Exception("__call__ function is not implemented yet")

    @tf.function
    def __call__(self, current_state, time_step, add_exploration_noise):
        """
       This is the call function for the Base Optimizer Class.
       It is used to calculate the optimal solution for action at the current timestep given the current state.

       Parameters
       ---------
       current_state: tf.float32
           Defines the current state of the system, (dims=num_of_agents X dim_S)
       time_step: tf.float32
           Defines the current timestep of the episode.
       exploration_noise: tf.bool
           Define if the optimal action should have some noise added to it before returning it.


       Returns
       -------
       resulting_action: tf.float32
            The optimal solution for the first action to be applied in the current time step.
       next_state: tf.float32
            The next state predicted using the dynamics model in the trajectory evaluator.
       rewards_of_next_state: tf.float32
            The predicted reward achieved after applying the action given by the optimizer.
       """
        resulting_action = \
            self._optimize(current_state, time_step)
        if add_exploration_noise:
            noise = tf.random.truncated_normal([self._num_agents, self._dim_U],
                                               self._exploration_mean,
                                               tf.sqrt(self._exploration_variance),
                                               dtype=tf.float32)
            resulting_action = resulting_action + noise
            resulting_action = tf.clip_by_value(resulting_action,
                                                self._action_lower_bound,
                                                self._action_upper_bound)
        next_state = self._trajectory_evaluator.predict_next_state(
            current_state, resulting_action)
        rewards_of_next_state = self._trajectory_evaluator.\
            evaluate_next_reward(current_state, next_state, resulting_action)
        return resulting_action, next_state, rewards_of_next_state

    def reset(self):
        """
          This method resets the optimizer to its default state at the
          beginning of the trajectory/episode.
          """

        raise Exception("reset function is not implemented yet")

    def set_trajectory_evaluator(self, trajectory_evaluator):
        """
        Sets the trajectory evaluator to be used by the optimizer.

        :param trajectory_evaluator: (EvaluatorBaseClass) Defines the
                trajectory evaluator to be used to evaluate the reward of a
                sequence of actions.
        :return:
        """
        self._trajectory_evaluator = trajectory_evaluator
        return
