import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class RandomSearchOptimizer(OptimizerBase):
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, population_size=1024, num_agents=5):
        """
        This class is responsible for performing random shooting and choosing the best
        possible predicted trajectory and returning the first action of this trajectory.


        Parameters
        ---------
        env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
        env_observation_space: gym.ObservationSpace
            Defines the observation space of the gym environment.
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        population_size: tf.int32
            Defines the population size of the particles evaluated at each iteration.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        """
        super(RandomSearchOptimizer, self).__init__(name=None,
                                                    planning_horizon=planning_horizon,
                                                    max_iterations=None,
                                                    num_agents=num_agents,
                                                    env_action_space=env_action_space,
                                                    env_observation_space=
                                                    env_observation_space)
        self._solution_dim = [self._num_agents, self._planning_horizon, self._dim_U]
        self._population_size = population_size
        return

    @tf.function
    def _optimize(self, current_state, time_step):
        samples = tf.random.uniform([self._population_size, *self._solution_dim], self._action_lower_bound_horizon,
                                    self._action_upper_bound_horizon, dtype=tf.float32)
        rewards = self._trajectory_evaluator(current_state, samples, time_step)
        best_particle_index = tf.cast(tf.math.argmax(rewards), dtype=tf.int32)
        samples = tf.transpose(samples, [1, 0, 2, 3])
        best_particle_index = best_particle_index + tf.range(0, samples.shape[0], dtype=tf.int32)*samples.shape[1]
        samples = tf.reshape(samples, [-1, *samples.shape[2:]])
        resulting_action = tf.gather(samples, best_particle_index)[:, 0]
        return resulting_action

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        return
