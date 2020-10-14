"""
tf_neuralmpc/optimizers/spsa.py
===============================
"""
import tensorflow as tf
import numpy as np
from tf_neuralmpc.optimizers.optimizer_base import OptimizerBase


class SPSAOptimizer(OptimizerBase):
    """This class defines the simultaneous perturbation stochastic approximation optimizer.
    (https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_Stochastic_Optimization.PDF)"""
    def __init__(self, planning_horizon, max_iterations, population_size,  dim_U, dim_O, num_agents,
                 action_upper_bound, action_lower_bound, trajectory_evaluator, alpha=tf.constant(0.602, dtype=tf.float32),
                 gamma=tf.constant(0.101, dtype=tf.float32), a_par=tf.constant(0.01, dtype=tf.float32),
                 noise_parameter=tf.constant(0.3, dtype=tf.float32)):
        """
       This is the initializer function for the simulataneous perturbation stochastic approximation optimizer.


       Parameters
       ---------
       planning_horizon: Int
           Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
       max_iterations: tf.int32
           Defines the maximimum iterations for the CMAES optimizer to refine its guess for the optimal solution.
       population_size: tf.int32
           Defines the population size of the particles evaluated at each iteration.
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
       alpha: tf.float32
           Defines the alpha used.
       gamma: tf.float32
           Defines the gamma used.
       a_par: tf.float32
           Defines the a_par used.
       noise_parameter: tf.float32
           Defines the noise_parameter used.
       """
        super(SPSAOptimizer, self).__init__(name=None)
        self.planning_horizon = planning_horizon
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O
        self.trajectory_evaluator = trajectory_evaluator
        self.num_agents = num_agents
        self.solution_dim = [self.num_agents, self.planning_horizon, self.dim_U]
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound_horizon = tf.tile(np.expand_dims(self.action_upper_bound, 0),
                                                                [self.planning_horizon, 1])
        self.action_lower_bound_horizon = tf.tile(np.expand_dims(self.action_lower_bound, 0),
                                                  [self.planning_horizon, 1])
        current_params = np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                 [self.planning_horizon*self.num_agents, 1])
        current_params = current_params.reshape([self.num_agents, self.planning_horizon, -1])
        self.alpha = alpha
        self.gamma = gamma
        self.a_par = a_par
        self.big_a_par = tf.cast(self.max_iterations, dtype=tf.float32) / tf.constant(10., dtype=tf.float32)
        self.noise_parameter = noise_parameter
        self.current_parameters = tf.Variable(tf.constant(current_params,
                                                          dtype=tf.float32))
        # exploration
        self.exploration_variance = (np.square(self.action_lower_bound - self.action_upper_bound) / 16) * 0.05
        self.exploration_mean = (self.action_upper_bound + self.action_lower_bound) / 2

    @tf.function
    def __call__(self, current_state, time_step, exploration_noise):
        """
      This is the call function for the simultaneous perturbation stochastic approximation optimizer.
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
        def continue_condition(t, solution):
            result = tf.less(t, self.max_iterations)
            return result

        def iterate(t, solution):
            #TODO: early termination and checking if we are always doing better or we diverged
            ak = self.a_par / (tf.cast(t, tf.float32) + 1 + self.big_a_par) ** self.alpha
            ck = self.noise_parameter / (tf.cast(t, tf.float32) + 1) ** self.gamma

            #sample delta for half of the population
            delta = (tf.random.uniform(shape=[self.population_size, *self.solution_dim],
                                       minval=0, maxval=2, dtype=tf.int32)*2) - 1
            delta = tf.cast(delta, dtype=tf.float32)

            parameters_plus = solution + ck * delta
            parameters_minus = solution - ck * delta

            parameters_plus_feasible = tf.clip_by_value(parameters_plus, self.action_lower_bound_horizon,
                                                        self.action_upper_bound_horizon)
            parameters_minus_feasible = tf.clip_by_value(parameters_minus, self.action_lower_bound_horizon,
                                                         self.action_upper_bound_horizon)
            parameters_plus_penalty = tf.norm(tf.reshape(parameters_plus - parameters_plus_feasible,
                                                         [self.population_size, self.num_agents, -1]),
                                              axis=2) ** 2
            parameters_minus_penalty = tf.norm(tf.reshape(parameters_minus - parameters_minus_feasible,
                                                          [self.population_size, self.num_agents, -1]),
                                               axis=2) ** 2
            parameters_plus = parameters_plus_feasible
            parameters_minus = parameters_minus_feasible

            #concat both for faster implementation
            actions_inputs = tf.concat([parameters_plus, parameters_minus], axis=0)
            full_rewards = self.trajectory_evaluator(current_state, actions_inputs,
                                                     time_step)
            #evaluate the costs
            rewards_parameters_plus = full_rewards[0:self.population_size] - parameters_plus_penalty
            rewards_parameters_minus = full_rewards[self.population_size:] - parameters_minus_penalty
            # Estimate the gradient
            ghat = tf.reduce_mean(tf.expand_dims(tf.expand_dims(rewards_parameters_plus -
                                                               rewards_parameters_minus, -1), -1) / (2. * ck * delta),
                                  axis=0)
            #update now the parameters
            new_solution = solution + ak * ghat
            new_solution = tf.clip_by_value(new_solution, self.action_lower_bound_horizon,
                                            self.action_upper_bound_horizon)

            return t + tf.constant(1, dtype=tf.int32), new_solution

        _, solution = tf.while_loop(cond=continue_condition, body=iterate,
                                    loop_vars=[tf.constant(0, dtype=tf.int32), self.current_parameters])
        #shift the solution for next iteration start
        self.current_parameters.assign(tf.concat([solution[:, 1:],
                                                 tf.expand_dims(solution[:, -1], 1)], 1))
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
        current_params = np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                 [self.planning_horizon*self.num_agents, 1])
        current_params = current_params.reshape([self.num_agents, self.planning_horizon, -1])
        self.current_parameters.assign(current_params)
