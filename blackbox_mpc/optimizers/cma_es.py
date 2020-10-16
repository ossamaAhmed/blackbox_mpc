import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class CMAESOptimizer(OptimizerBase):
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, max_iterations=5, population_size=500,
                 num_elite=50, num_agents=5, alpha_cov=tf.constant(2.0, dtype=tf.float32),
                 h_sigma=tf.constant(1.0, dtype=tf.float32)):
        """
        This class defines a Covariance Matrix Adaptation Evolutionary-Strategy.
        (https://arxiv.org/pdf/1604.00772.pdf) Note: this optimzer is not optimized for more than one agent

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
        num_elite: tf.int32
            Defines the number of elites kept for the next iteration from the population.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        alpha_cov: tf.float32
            Defines the alpha covariance to be used.
        h_sigma: tf.float32
            Defines the h sigma to be used.
        """
        super(CMAESOptimizer, self).__init__(name=None,
                                             planning_horizon=planning_horizon,
                                             max_iterations=max_iterations,
                                             num_agents=num_agents,
                                             env_action_space=env_action_space,
                                             env_observation_space=
                                             env_observation_space)
        self._solution_dim = [self._num_agents,
                              self._planning_horizon,
                              self._dim_U]
        self._population_size = population_size
        self._num_elite = num_elite
        previous_solution_values = tf.constant(np.tile((self._action_lower_bound +
                                                        self._action_upper_bound) / 2,
                                                       [self._planning_horizon *
                                                        self._num_agents, 1]),
                                               dtype=tf.float32)
        previous_solution_values = tf.reshape(previous_solution_values, [-1])
        solution_variance_values = tf.constant(np.tile(np.square(self._action_lower_bound
                                                                 - self._action_upper_bound) / 16,
                                                       [self._planning_horizon *
                                                        self._num_agents, 1]),
                                               dtype=tf.float32)
        solution_variance_values = tf.reshape(solution_variance_values, [-1])

        # Recombination weights
        self._weights = tf.concat([
            tf.math.log(tf.cast(self._num_elite, dtype=tf.float32) + 0.5) -
            tf.math.log(tf.range(1, tf.cast(self._num_elite, dtype=tf.float32) + 1)),
            tf.zeros(shape=(self._population_size - self._num_elite,), dtype=tf.float32),
        ], axis=0)
        # Normalize weights such as they sum to one and reshape into a column matrix
        self._weights = (self._weights / tf.reduce_sum(self._weights))[:, tf.newaxis]
        self._mu_eff = tf.reduce_sum(self._weights) ** 2 / \
                       tf.reduce_sum(self._weights ** 2)
        self._solution_size = tf.reduce_prod(self._solution_dim)
        #step_size_control
        self._c_sigma = (self._mu_eff + 2) / (tf.cast(self._solution_size,
                                                      dtype=tf.float32) +
                                              self._mu_eff + 5)
        self._d_sigma = 1 + 2 * tf.maximum(0, tf.sqrt((self._mu_eff - 1) /
                                                      (tf.cast(self._solution_size,
                                                               dtype=tf.float32) + 1)) - 1) \
                        + self._c_sigma
        #Covariance Matrix Adaptation
        self._cc = (4 + self._mu_eff / tf.cast(self._solution_size, dtype=tf.float32)) / \
                    (tf.cast(self._solution_size, dtype=tf.float32) + 4 + 2 * self._mu_eff /
                    tf.cast(self._solution_size, dtype=tf.float32))
        self._alpha_cov = alpha_cov
        self._h_sigma = h_sigma
        self._c1 = self._alpha_cov / ((tf.cast(self._solution_size,
                                               dtype=tf.float32) + 1.3) ** 2 +
                                      self._mu_eff)
        c_mu_option_two = self._alpha_cov * (self._mu_eff - 2 + 1 / self._mu_eff) / \
                          ((tf.cast(self._solution_size, dtype=tf.float32) + 2)
                           ** 2 + self._alpha_cov * self._mu_eff / 2)
        self._c_mu = tf.minimum(1 - self._c1, c_mu_option_two)
        #define trainable parameters
        # Mean
        self._m = tf.Variable(previous_solution_values)
        # Step-size
        self._sigma = tf.Variable(tf.math.sqrt(solution_variance_values))
        # Covariance matrix
        self._C = tf.Variable(tf.eye(num_rows=tf.cast(self._solution_size,
                                                      dtype=tf.float32),
                                     dtype=tf.float32))
        # Evolution path for σ
        self._p_sigma = tf.Variable(tf.zeros((tf.cast(self._solution_size,
                                                      dtype=tf.float32),),
                                             dtype=tf.float32))
        # Evolution path for C
        self._p_C = tf.Variable(tf.zeros((tf.cast(self._solution_size,
                                                  dtype=tf.float32),),
                                         dtype=tf.float32))
        # Coordinate system (normalized eigenvectors)
        self._B = tf.Variable(tf.eye(num_rows=tf.cast(self._solution_size,
                                                      dtype=tf.float32),
                                     dtype=tf.float32))
        # Scaling (square root of eigenvalues)
        self._D = tf.Variable(tf.eye(num_rows=tf.cast(self._solution_size,
                                                      dtype=tf.float32),
                                     dtype=tf.float32))
        self._expectation_of_normal = tf.sqrt(tf.cast(self._solution_size,
                                                      dtype=tf.float32) *
                                              (1 - 1 / (4 * tf.cast(
                                                  self._solution_size,
                                                  dtype=tf.float32)) +
                                              1 / (21 * tf.cast(
                                                          self._solution_size,
                                                          dtype=tf.float32)
                                                   ** 2)))
        return

    @tf.function
    def _optimize(self, current_state, time_step):
        def continue_condition(t, mean):
            result = tf.less(t, self._max_iterations)
            return result

        def iterate(t, mean):
            # -----------------------------------------------------
            # (1) Sample a new population of solutions ∼ N(m, σ²C)
            # -----------------------------------------------------
            z = tf.random.normal([self._population_size, self._solution_size], dtype=tf.float32)
            y = tf.matmul(z, tf.matmul(self._B, self._D))
            samples = self._m + self._sigma * y
            samples = tf.reshape(samples, [self._population_size, *self._solution_dim])
            # -------------------------------------------------
            # (2) Selection and Recombination: Moving the Mean
            # -------------------------------------------------
            # Evaluate and sort solutions
            samples_feasible = tf.clip_by_value(samples, self._action_lower_bound_horizon,
                                                self._action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(samples - samples_feasible,
                                         [self._population_size, self._num_agents, -1]),
                              axis=2) ** 2
            samples = samples_feasible
            # -------------------------------------------------
            # (2) Selection and Recombination: Moving the Mean
            # -------------------------------------------------
            # Evaluate and sort solutions
            rewards = self._trajectory_evaluator(current_state, samples, time_step) - penalty
            rewards = tf.reduce_sum(rewards, axis=1) #TODO: double check this, very flaky
            self._x_sorted = tf.gather(samples, tf.argsort(rewards, direction='DESCENDING'))
            # The new mean is a weighted average of the top-μ solutions
            x_diff = (tf.reshape(self._x_sorted, [self._population_size, self._solution_size]) - self._m)
            x_mean = tf.reduce_sum(tf.multiply(x_diff, self._weights), axis=0)
            m = self._m + x_mean
            # ----------------------
            # (3) Step-size control
            # ----------------------
            y_mean = x_mean / self._sigma
            D_inv = tf.linalg.tensor_diag(tf.math.reciprocal(tf.linalg.diag_part(self._D)))
            C_inv_half = tf.matmul(tf.matmul(self._B, D_inv), tf.transpose(self._B))
            p_sigma = ((1 - self._c_sigma) * self._p_sigma) + (tf.math.sqrt(self._c_sigma * (2 - self._c_sigma) * self._mu_eff) *
                                                               tf.squeeze(tf.matmul(C_inv_half, y_mean[:, tf.newaxis])))
            sigma = self._sigma * tf.exp((self._c_sigma / self._d_sigma) * ((tf.norm(p_sigma) /
                                                                             self._expectation_of_normal) - 1))
            # -----------------------------------
            # (4) Adapting the Covariance Matrix
            # -----------------------------------
            p_C = ((1 - self._cc) * self._p_C + (self._h_sigma * tf.sqrt(self._cc * (2 - self._cc) * self._mu_eff) * y_mean))

            p_C_matrix = p_C[:, tf.newaxis]
            y_mean_unweighted = x_diff / self._sigma
            y_mean_unweighted_squared = tf.map_fn(fn=lambda e: e * tf.transpose(e), elems=y_mean_unweighted[:, tf.newaxis])
            y_s = tf.reduce_sum(tf.multiply(y_mean_unweighted_squared, self._weights[:, tf.newaxis]), axis=0)
            C = ((1 - self._c1 - self._c_mu) * self._C + self._c1 * p_C_matrix * tf.transpose(p_C_matrix) +
                 self._c_mu * y_s)
            # -----------------------------------
            # (5) Ensure the symmetry of the covariance matrix here
            # -----------------------------------
            C_upper = tf.linalg.band_part(C, 0, -1)
            C_upper_no_diag = C_upper - tf.linalg.tensor_diag(tf.linalg.diag_part(C_upper))
            C = C_upper + tf.transpose(C_upper_no_diag)

            # -----------------------------------
            # (6)Update the values
            # -----------------------------------
            u, B, _ = tf.linalg.svd(C)

            diag_D = tf.sqrt(u)
            D = tf.linalg.tensor_diag(diag_D)
            # Assign values
            self._p_C.assign(p_C)
            self._p_sigma.assign(p_sigma)
            self._C.assign(C)
            self._sigma.assign(sigma)
            self._B.assign(B)
            self._D.assign(D)
            self._m.assign(m)
            return t + tf.constant(1, dtype=tf.int32), m

        _ = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32),
                                                                            self._m])
        solution = tf.reshape(self._m, self._solution_dim)
        resulting_action = solution[:, 0]
        return resulting_action

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        previous_solution_values = tf.constant(np.tile((self._action_lower_bound + self._action_upper_bound) / 2,
                                                       [self._planning_horizon * self._num_agents, 1]), dtype=tf.float32)
        previous_solution_values = tf.reshape(previous_solution_values, [-1])
        solution_variance_values = tf.constant(
            np.tile(np.square(self._action_lower_bound - self._action_upper_bound) / 16,
                    [self._planning_horizon * self._num_agents, 1]), dtype=tf.float32)
        solution_variance_values = tf.reshape(solution_variance_values, [-1])
        self._m.assign(previous_solution_values)
        self._sigma.assign(tf.math.sqrt(solution_variance_values))
