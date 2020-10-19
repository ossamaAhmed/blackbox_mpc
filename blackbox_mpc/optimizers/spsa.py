import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class SPSAOptimizer(OptimizerBase):
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, max_iterations=5, population_size=500,
                 num_agents=5, alpha=tf.constant(0.602, dtype=tf.float32),
                 gamma=tf.constant(0.101, dtype=tf.float32),
                 a_par=tf.constant(0.01, dtype=tf.float32),
                 noise_parameter=tf.constant(0.3, dtype=tf.float32)):
        """
           This class defines the simultaneous perturbation stochastic approximation optimizer.
           (https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_Stochastic_Optimization.PDF)


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
           alpha: tf.float32
               Defines the alpha used.
           gamma: tf.float32
               Defines the gamma used.
           a_par: tf.float32
               Defines the a_par used.
           noise_parameter: tf.float32
               Defines the noise_parameter used.
        """
        super(SPSAOptimizer, self).__init__(name=None,
                                           planning_horizon=planning_horizon,
                                           max_iterations=max_iterations,
                                           num_agents=num_agents,
                                           env_action_space=env_action_space,
                                           env_observation_space=
                                           env_observation_space)
        self._solution_dim = [self._num_agents, self._planning_horizon, self._dim_U]
        self._population_size = population_size
        current_params = np.tile((self._action_lower_bound + self._action_upper_bound) / 2,
                                 [self._planning_horizon * self._num_agents, 1])
        current_params = current_params.reshape([self._num_agents, self._planning_horizon, -1])
        self._alpha = alpha
        self._gamma = gamma
        self._a_par = a_par
        self._big_a_par = tf.cast(self._max_iterations, dtype=tf.float32) / tf.constant(10., dtype=tf.float32)
        self._noise_parameter = noise_parameter
        self._current_parameters = tf.Variable(tf.constant(current_params,
                                                           dtype=tf.float32))

    @tf.function
    def _optimize(self, current_state, time_step):
        def continue_condition(t, solution):
            result = tf.less(t, self._max_iterations)
            return result

        def iterate(t, solution):
            #TODO: early termination and checking if we are always doing better or we diverged
            ak = self._a_par / (tf.cast(t, tf.float32) + 1 + self._big_a_par) ** self._alpha
            ck = self._noise_parameter / (tf.cast(t, tf.float32) + 1) ** self._gamma

            #_sample delta for half of the population
            delta = (tf.random.uniform(shape=[self._population_size, *self._solution_dim],
                                       minval=0, maxval=2, dtype=tf.int32)*2) - 1
            delta = tf.cast(delta, dtype=tf.float32)

            parameters_plus = solution + ck * delta
            parameters_minus = solution - ck * delta

            parameters_plus_feasible = tf.clip_by_value(parameters_plus, self._action_lower_bound_horizon,
                                                        self._action_upper_bound_horizon)
            parameters_minus_feasible = tf.clip_by_value(parameters_minus, self._action_lower_bound_horizon,
                                                         self._action_upper_bound_horizon)
            parameters_plus_penalty = tf.norm(tf.reshape(parameters_plus - parameters_plus_feasible,
                                                         [self._population_size, self._num_agents, -1]),
                                              axis=2) ** 2
            parameters_minus_penalty = tf.norm(tf.reshape(parameters_minus - parameters_minus_feasible,
                                                          [self._population_size, self._num_agents, -1]),
                                               axis=2) ** 2
            parameters_plus = parameters_plus_feasible
            parameters_minus = parameters_minus_feasible

            #concat both for faster implementation
            actions_inputs = tf.concat([parameters_plus, parameters_minus], axis=0)
            full_rewards = self._trajectory_evaluator(current_state, actions_inputs,
                                                      time_step)
            #evaluate the costs
            rewards_parameters_plus = full_rewards[0:self._population_size] - parameters_plus_penalty
            rewards_parameters_minus = full_rewards[self._population_size:] - parameters_minus_penalty
            # Estimate the gradient
            ghat = tf.reduce_mean(tf.expand_dims(tf.expand_dims(rewards_parameters_plus -
                                                               rewards_parameters_minus, -1), -1) / (2. * ck * delta),
                                  axis=0)
            #update now the parameters
            new_solution = solution + ak * ghat
            new_solution = tf.clip_by_value(new_solution, self._action_lower_bound_horizon,
                                            self._action_upper_bound_horizon)

            return t + tf.constant(1, dtype=tf.int32), new_solution

        _, solution = tf.while_loop(cond=continue_condition, body=iterate,
                                    loop_vars=[tf.constant(0, dtype=tf.int32), self._current_parameters])
        #shift the solution for next iteration start
        self._current_parameters.assign(tf.concat([solution[:, 1:],
                                                   tf.expand_dims(solution[:, -1], 1)], 1))
        resulting_action = solution[:, 0]
        return resulting_action

    def reset(self):
        """
         This method resets the optimizer to its default state at the beginning of the trajectory/episode.
         """
        current_params = np.tile((self._action_lower_bound + self._action_upper_bound) / 2,
                                 [self._planning_horizon * self._num_agents, 1])
        current_params = current_params.reshape([self._num_agents, self._planning_horizon, -1])
        self._current_parameters.assign(current_params)
        return
