"""
blackbox_mpc/optimizers/random_search.py
=========================================
"""
import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class RandomSearchOptimizer(OptimizerBase):
    """This class is responsible for performing random shooting and choosing the best
       possible predicted trajectory and returning the first action of this trajectory."""
    def __init__(self, planning_horizon, population_size, dim_U, dim_O,
                 action_upper_bound, action_lower_bound, trajectory_evaluator, num_agents):
        """
        This is the initializer function for the Cross-Entropy Method Optimizer.


        Parameters
        ---------
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        population_size: tf.int32
            Defines the population size of the particles evaluated at each iteration.
        dim_U: tf.int32
            Defines the dimensions of the input/ action space.
        dim_O: tf.int32
            Defines the dimensions of the observations space.
        action_upper_bound: tf.float32
            Defines the actions upper bound that could be applied, shape should be 1xdim_U.
        action_lower_bound: tf.float32
            Defines the actions lower bound that could be applied, shape should be 1xdim_U.
        trajectory_evaluator: EvaluatorBaseClass
            Defines the trajectory evaluator to be used to evaluate the reward of a sequence of actions.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        """
        super(RandomSearchOptimizer, self).__init__(name=None)
        self.planning_horizon = planning_horizon
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O
        self.num_agents = num_agents
        self.trajectory_evaluator = trajectory_evaluator
        self.solution_dim = [self.num_agents, self.planning_horizon, self.dim_U]
        self.population_size = population_size
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound_horizon = tf.tile(np.expand_dims(self.action_upper_bound, 0),
                                                                [self.planning_horizon, 1])
        self.action_lower_bound_horizon = tf.tile(np.expand_dims(self.action_lower_bound, 0),
                                                  [self.planning_horizon, 1])
        #exploration
        self.exploration_variance = (np.square(self.action_lower_bound - self.action_upper_bound) / 16)*0.05
        self.exploration_mean = (self.action_upper_bound + self.action_lower_bound)/2

    @tf.function
    def __call__(self, current_state, time_step, exploration_noise):
        """
          This is the call function for the Random Shooting Method Optimizer.
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
        samples = tf.random.uniform([self.population_size, *self.solution_dim], self.action_lower_bound_horizon,
                                    self.action_upper_bound_horizon, dtype=tf.float32)
        rewards = self.trajectory_evaluator(current_state, samples, time_step)
        best_particle_index = tf.cast(tf.math.argmax(rewards), dtype=tf.int32)
        samples = tf.transpose(samples, [1, 0, 2, 3])
        best_particle_index = best_particle_index + tf.range(0, samples.shape[0], dtype=tf.int32)*samples.shape[1]
        samples = tf.reshape(samples, [-1, *samples.shape[2:]])
        resulting_action = tf.gather(samples, best_particle_index)[:, 0]
        if exploration_noise:
            noise = tf.random.truncated_normal([self.num_agents, self.dim_U],
                                               self.exploration_mean,
                                               tf.sqrt(self.exploration_variance),
                                               dtype=tf.float32)
            resulting_action = resulting_action + noise
            resulting_action = tf.clip_by_value(resulting_action, self.action_lower_bound,
                                                self.action_upper_bound)
        next_state = self.trajectory_evaluator.predict_next_state(current_state, resulting_action)
        rewards_of_next_state = self.trajectory_evaluator.evaluate_next_reward(current_state,
                                                                               next_state,
                                                                               resulting_action)
        return resulting_action, next_state, rewards_of_next_state

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        return
