"""
blackbox_mpc/optimizers/cma_es.py
=================================
"""
import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class CMAESOptimizer(OptimizerBase):
    """This class defines a Covariance Matrix Adaptation Evolutionary-Strategy.
    (https://arxiv.org/pdf/1604.00772.pdf) Note: this optimzer is not optimized for more than one agent"""
    def __init__(self, planning_horizon, max_iterations, population_size, num_elite,
                 dim_U, dim_O, action_upper_bound, action_lower_bound, num_agents,
                 trajectory_evaluator, alpha_cov=tf.constant(2.0, dtype=tf.float32),
                 h_sigma=tf.constant(1.0, dtype=tf.float32)):
        """
        This is the initializer function for the Covariance Matrix Adaptation Evolutionary-Strategy Optimizer.


        Parameters
        ---------
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        max_iterations: tf.int32
            Defines the maximimum iterations for the CMAES optimizer to refine its guess for the optimal solution.
        population_size: tf.int32
            Defines the population size of the particles evaluated at each iteration.
        num_elite: tf.int32
            Defines the number of elites kept for the next iteration from the population.
        dim_O: tf.int32
            Defines the dimensions of the observations space.
        dim_U: tf.int32
            Defines the dimensions of the input space.
        action_upper_bound: tf.float32
            Defines the actions upper bound that could be applied, shape should be 1xdim_U.
        action_lower_bound: tf.float32
            Defines the actions lower bound that could be applied, shape should be 1xdim_U.
        trajectory_evaluator: EvaluatorBaseClass
            Defines the trajectory evaluator to be used to evaluate the reward of a sequence of actions.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        alpha_cov: tf.float32
            Defines the alpha covariance to be used.
        h_sigma: tf.float32
            Defines the h sigma to be used.
        """
        super(CMAESOptimizer, self).__init__(name=None)
        self.planning_horizon = planning_horizon
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O
        self.trajectory_evaluator = trajectory_evaluator
        self.num_agents = num_agents
        self.solution_dim = [self.num_agents, self.planning_horizon, self.dim_U]
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.num_elite = num_elite
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound

        self.action_upper_bound_horizon = tf.tile(np.expand_dims(self.action_upper_bound, 0),
                                                                [self.planning_horizon, 1])

        self.action_lower_bound_horizon = tf.tile(np.expand_dims(self.action_lower_bound, 0),
                                                  [self.planning_horizon, 1])
        previous_solution_values = tf.constant(np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                                       [self.planning_horizon * self.num_agents, 1]), dtype=tf.float32)
        previous_solution_values = tf.reshape(previous_solution_values, [-1])
        solution_variance_values = tf.constant(np.tile(np.square(self.action_lower_bound - self.action_upper_bound) / 16,
                                               [self.planning_horizon * self.num_agents, 1]), dtype=tf.float32)
        solution_variance_values = tf.reshape(solution_variance_values, [-1])

        # Recombination weights
        self.weights = tf.concat([
            tf.math.log(tf.cast(self.num_elite, dtype=tf.float32) + 0.5) -
            tf.math.log(tf.range(1, tf.cast(self.num_elite, dtype=tf.float32) + 1)),
            tf.zeros(shape=(self.population_size - self.num_elite,), dtype=tf.float32),
        ], axis=0)
        # Normalize weights such as they sum to one and reshape into a column matrix
        self.weights = (self.weights / tf.reduce_sum(self.weights))[:, tf.newaxis]
        self.mu_eff = tf.reduce_sum(self.weights) ** 2 / tf.reduce_sum(self.weights ** 2)
        self.solution_size = tf.reduce_prod(self.solution_dim)
        #step_size_control
        self.c_sigma = (self.mu_eff + 2) / (tf.cast(self.solution_size, dtype=tf.float32) + self.mu_eff + 5)
        self.d_sigma = 1 + 2 * tf.maximum(0, tf.sqrt((self.mu_eff - 1) / (tf.cast(self.solution_size, dtype=tf.float32) + 1)) - 1) \
                       + self.c_sigma
        #Covariance Matrix Adaptation
        self.cc = (4 + self.mu_eff / tf.cast(self.solution_size, dtype=tf.float32)) / \
                  (tf.cast(self.solution_size, dtype=tf.float32) + 4 + 2 * self.mu_eff /
                   tf.cast(self.solution_size, dtype=tf.float32))
        self.alpha_cov = alpha_cov
        self.h_sigma = h_sigma
        self.c1 = self.alpha_cov / ((tf.cast(self.solution_size, dtype=tf.float32) + 1.3) ** 2 + self.mu_eff)
        c_mu_option_two = self.alpha_cov * (self.mu_eff - 2 + 1 / self.mu_eff) / \
                            ((tf.cast(self.solution_size, dtype=tf.float32) + 2) ** 2 + self.alpha_cov * self.mu_eff / 2)
        self.c_mu = tf.minimum(1 - self.c1, c_mu_option_two)
        #define trainable parameters
        # Mean
        self.m = tf.Variable(previous_solution_values)
        # Step-size
        self.sigma = tf.Variable(tf.math.sqrt(solution_variance_values))
        # Covariance matrix
        self.C = tf.Variable(tf.eye(num_rows=tf.cast(self.solution_size, dtype=tf.float32), dtype=tf.float32))
        # Evolution path for σ
        self.p_sigma = tf.Variable(tf.zeros((tf.cast(self.solution_size, dtype=tf.float32),), dtype=tf.float32))
        # Evolution path for C
        self.p_C = tf.Variable(tf.zeros((tf.cast(self.solution_size, dtype=tf.float32),), dtype=tf.float32))
        # Coordinate system (normalized eigenvectors)
        self.B = tf.Variable(tf.eye(num_rows=tf.cast(self.solution_size, dtype=tf.float32), dtype=tf.float32))
        # Scaling (square root of eigenvalues)
        self.D = tf.Variable(tf.eye(num_rows=tf.cast(self.solution_size, dtype=tf.float32), dtype=tf.float32))
        #TODO: double check the below expression
        self.expectation_of_normal = tf.sqrt(tf.cast(self.solution_size, dtype=tf.float32) *
                                             (1 - 1 / (4 * tf.cast(self.solution_size, dtype=tf.float32)) +
                                              1 / (21 * tf.cast(self.solution_size, dtype=tf.float32)**2)))
        # exploration
        self.exploration_variance = (np.square(self.action_lower_bound - self.action_upper_bound) / 16) * 0.05
        self.exploration_mean = (self.action_upper_bound + self.action_lower_bound) / 2
        return

    @tf.function
    def __call__(self, current_state, time_step, exploration_noise):
        """
          This is the call function for the Covariance Matrix Adaptation Evolutionary-Strategy Optimizer.
          It is used to calculate the optimal solution for action at the current timestep given the current state.

          Parameters
          ---------
          current_state: tf.float32
              Defines the current state of the system, (dims=num_of_agents X dim_S)
          time_step: tf.float32
              Defines the current timestep of the episode.
          exploration_noise: tf.bool
              Specifies if the optimal action should have some noise added to it before returning it.


          Returns
           -------
           resulting_action: tf.float32
               The optimal solution for the first action to be applied in the current time step.
           next_state: tf.float32
               The next state predicted using the dynamics model in the trajectory evaluator.
           rewards_of_next_state: tf.float32
               The predicted reward achieved after applying the action given by the optimizer.
          """
        def continue_condition(t, mean):
            result = tf.less(t, self.max_iterations)
            return result

        def iterate(t, mean):
            # -----------------------------------------------------
            # (1) Sample a new population of solutions ∼ N(m, σ²C)
            # -----------------------------------------------------
            z = tf.random.normal([self.population_size, self.solution_size], dtype=tf.float32)
            y = tf.matmul(z, tf.matmul(self.B, self.D))
            samples = self.m + self.sigma * y
            samples = tf.reshape(samples, [self.population_size, *self.solution_dim])
            # -------------------------------------------------
            # (2) Selection and Recombination: Moving the Mean
            # -------------------------------------------------
            # Evaluate and sort solutions
            samples_feasible = tf.clip_by_value(samples, self.action_lower_bound_horizon,
                                                self.action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(samples - samples_feasible, [self.population_size, self.num_agents, -1]), axis=2) ** 2
            samples = samples_feasible
            # -------------------------------------------------
            # (2) Selection and Recombination: Moving the Mean
            # -------------------------------------------------
            # Evaluate and sort solutions
            rewards = self.trajectory_evaluator(current_state, samples, time_step) - penalty
            rewards = tf.reduce_sum(rewards, axis=1) #TODO: double check this, very flaky
            self.x_sorted = tf.gather(samples, tf.argsort(rewards, direction='DESCENDING'))
            # The new mean is a weighted average of the top-μ solutions
            x_diff = (tf.reshape(self.x_sorted, [self.population_size, self.solution_size]) - self.m)
            x_mean = tf.reduce_sum(tf.multiply(x_diff, self.weights), axis=0)
            m = self.m + x_mean
            # ----------------------
            # (3) Step-size control
            # ----------------------
            y_mean = x_mean / self.sigma
            D_inv = tf.linalg.tensor_diag(tf.math.reciprocal(tf.linalg.diag_part(self.D)))
            C_inv_half = tf.matmul(tf.matmul(self.B, D_inv), tf.transpose(self.B))
            p_sigma = ((1 - self.c_sigma) * self.p_sigma) + (tf.math.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) *
                                                             tf.squeeze(tf.matmul(C_inv_half, y_mean[:, tf.newaxis])))
            sigma = self.sigma * tf.exp((self.c_sigma / self.d_sigma) * ((tf.norm(p_sigma) /
                                                                          self.expectation_of_normal) - 1))
            # -----------------------------------
            # (4) Adapting the Covariance Matrix
            # -----------------------------------
            p_C = ((1 - self.cc) * self.p_C + (self.h_sigma * tf.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_mean))

            p_C_matrix = p_C[:, tf.newaxis]
            y_mean_unweighted = x_diff / self.sigma
            y_mean_unweighted_squared = tf.map_fn(fn=lambda e: e * tf.transpose(e), elems=y_mean_unweighted[:, tf.newaxis])
            y_s = tf.reduce_sum(tf.multiply(y_mean_unweighted_squared, self.weights[:, tf.newaxis]), axis=0)
            C = ((1 - self.c1 - self.c_mu) * self.C + self.c1 * p_C_matrix * tf.transpose(p_C_matrix) +
                self.c_mu * y_s)
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
            self.p_C.assign(p_C)
            self.p_sigma.assign(p_sigma)
            self.C.assign(C)
            self.sigma.assign(sigma)
            self.B.assign(B)
            self.D.assign(D)
            self.m.assign(m)
            return t + tf.constant(1, dtype=tf.int32), m

        _ = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32),
                                                                            self.m])
        solution = tf.reshape(self.m, self.solution_dim)
        resulting_action = solution[:, 0]
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
        previous_solution_values = tf.constant(np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                                       [self.planning_horizon * self.num_agents, 1]), dtype=tf.float32)
        previous_solution_values = tf.reshape(previous_solution_values, [-1])
        solution_variance_values = tf.constant(
            np.tile(np.square(self.action_lower_bound - self.action_upper_bound) / 16,
                    [self.planning_horizon * self.num_agents, 1]), dtype=tf.float32)
        solution_variance_values = tf.reshape(solution_variance_values, [-1])
        self.m.assign(previous_solution_values)
        self.sigma.assign(tf.math.sqrt(solution_variance_values))
