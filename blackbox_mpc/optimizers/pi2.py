import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase
tfd = tfp.distributions


class PI2Optimizer(OptimizerBase):
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, max_iterations=5, population_size=500,
                 num_agents=5, lamda=tf.constant(1.0, dtype=tf.float32)):
        """
        This class defines the information theortic MPC based on path intergral methods.
        (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202)


        Parameters
        ---------
        env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
        env_observation_space: gym.ObservationSpace
            Defines the observation space of the gym environment.
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        max_iterations: tf.int32
            Defines the maximimum iterations for the CMAES optimizer to refine its guess for the optimal solution.
        population_size: tf.int32
            Defines the population size of the particles evaluated at each iteration.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        lamda: tf.float32
            Defines the lamda used the energy function.
        """
        super(PI2Optimizer, self).__init__(name=None,
                                           planning_horizon=planning_horizon,
                                           max_iterations=max_iterations,
                                           num_agents=num_agents,
                                           env_action_space=env_action_space,
                                           env_observation_space=
                                           env_observation_space)
        self._solution_dim = [self._num_agents, self._planning_horizon, self._dim_U]
        self._solution_size = tf.reduce_prod(self._solution_dim)
        self._population_size = population_size
        previous_solution_values = np.tile((self._action_lower_bound + self._action_upper_bound) / 2,
                                           [self._planning_horizon * self._num_agents, 1])
        previous_solution_values = previous_solution_values.reshape([self._num_agents, self._planning_horizon, -1])
        self._previous_solution = tf.Variable(tf.zeros(shape=previous_solution_values.shape,
                                                       dtype=tf.float32))
        self._previous_solution.assign(previous_solution_values)
        solution_variance_values = np.tile(np.square(self._action_lower_bound - self._action_upper_bound) / 16,
                                           [self._planning_horizon * self._num_agents, 1])
        solution_variance_values = solution_variance_values.reshape([self._num_agents, self._planning_horizon, -1])
        self._solution_variance = tf.Variable(tf.zeros(shape=solution_variance_values.shape,
                                                       dtype=tf.float32))
        self._solution_variance.assign(solution_variance_values)
        self._lamda = lamda

    @tf.function
    def _optimize(self, current_state, time_step):
        def continue_condition(t, mean):
            result = tf.less(t, self._max_iterations)
            return result

        def iterate(t, mean):
            samples = tf.random.truncated_normal([self._population_size,
                                                  *self._solution_dim],
                                                 mean,
                                                 tf.sqrt(self._solution_variance),
                                                 dtype=tf.float32)
            samples_feasible = tf.clip_by_value(samples, self._action_lower_bound_horizon,
                                                self._action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(samples - samples_feasible, [self._population_size,
                                                                      self._num_agents,
                                                                      -1]),
                              axis=2) ** 2
            samples = samples_feasible

            rewards = self._trajectory_evaluator(current_state, samples, time_step) - penalty
            costs = -rewards
            costs = tf.transpose(costs, [1, 0])
            beta = tf.reduce_min(costs, axis=1)
            prob = tf.math.exp(-(1 / self._lamda) * (costs - tf.expand_dims(beta, -1)))
            eta = tf.reduce_sum(prob, axis=1)
            #compute weights now
            omega = tf.expand_dims(1 / eta, -1) * prob
            samples = tf.transpose(samples, [1, 0, 2, 3])
            new_mean = tf.reduce_sum(tf.multiply(samples, tf.expand_dims(tf.expand_dims(omega, -1), -1)), axis=1)
            return t + tf.constant(1, dtype=tf.int32), new_mean

        _, new_mean = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32),
                                                                                      self._previous_solution])
        self._previous_solution.assign(tf.concat([new_mean[:, 1:],
                                                  tf.expand_dims(new_mean[:, -1], 1)], 1))

        resulting_action = new_mean[:, 0]
        return resulting_action

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        previous_solution_values = np.tile((self._action_lower_bound + self._action_upper_bound) / 2,
                                           [self._planning_horizon * self._num_agents, 1])
        previous_solution_values = previous_solution_values.reshape([self._num_agents, self._planning_horizon, -1])
        self._previous_solution.assign(previous_solution_values)
