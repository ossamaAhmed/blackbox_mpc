"""
blackbox_mpc/optimizers/pi2.py
==============================
"""
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase
tfd = tfp.distributions


class PI2Optimizer(OptimizerBase):
    """This class defines the information theortic MPC based on path intergral methods.
        (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7989202)"""
    def __init__(self, planning_horizon, max_iterations, population_size, dim_U, dim_O,
                 action_upper_bound, action_lower_bound, trajectory_evaluator, num_agents,
                 lamda=tf.constant(1.0, dtype=tf.float32)):
        """
        This is the initializer function for the information theortic MPC optimizer.


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
        lamda: tf.float32
            Defines the lamda used the energy function.
        """
        super(PI2Optimizer, self).__init__(name=None)
        self.planning_horizon = planning_horizon
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O
        self.num_agents = num_agents
        self.trajectory_evaluator = trajectory_evaluator
        self.solution_dim = [self.num_agents, self.planning_horizon, self.dim_U]
        self.solution_size = tf.reduce_prod(self.solution_dim)
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound_horizon = tf.tile(np.expand_dims(self.action_upper_bound, 0),
                                                  [self.planning_horizon, 1])
        self.action_lower_bound_horizon = tf.tile(np.expand_dims(self.action_lower_bound, 0),
                                                  [self.planning_horizon, 1])
        previous_solution_values = np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                           [self.planning_horizon*self.num_agents, 1])
        previous_solution_values = previous_solution_values.reshape([self.num_agents, self.planning_horizon, -1])
        self.previous_solution = tf.Variable(tf.zeros(shape=previous_solution_values.shape,
                                                      dtype=tf.float32))
        self.previous_solution.assign(previous_solution_values)
        solution_variance_values = np.tile(np.square(self.action_lower_bound - self.action_upper_bound) / 16,
                                           [self.planning_horizon*self.num_agents, 1])
        solution_variance_values = solution_variance_values.reshape([self.num_agents, self.planning_horizon, -1])
        self.solution_variance = tf.Variable(tf.zeros(shape=solution_variance_values.shape,
                                                      dtype=tf.float32))
        self.solution_variance.assign(solution_variance_values)
        #TODO: to be pushed to a config file
        self.lamda = lamda
        # exploration
        self.exploration_variance = (np.square(self.action_lower_bound - self.action_upper_bound) / 16) * 0.05
        self.exploration_mean = (self.action_upper_bound + self.action_lower_bound) / 2

    @tf.function
    def __call__(self, current_state, time_step, exploration_noise):
        """
          This is the call function for the information theortic MPC optimizer.
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
            samples = tf.random.truncated_normal([self.population_size,
                                                  *self.solution_dim],
                                                 mean,
                                                 tf.sqrt(self.solution_variance),
                                                 dtype=tf.float32)
            samples_feasible = tf.clip_by_value(samples, self.action_lower_bound_horizon,
                                                self.action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(samples - samples_feasible, [self.population_size,
                                                                      self.num_agents,
                                                                      -1]),
                              axis=2) ** 2
            samples = samples_feasible

            rewards = self.trajectory_evaluator(current_state, samples, time_step) - penalty
            costs = -rewards
            costs = tf.transpose(costs, [1, 0])
            beta = tf.reduce_min(costs, axis=1)
            prob = tf.math.exp(-(1/self.lamda) * (costs - tf.expand_dims(beta, -1)))
            eta = tf.reduce_sum(prob, axis=1)
            #compute weights now
            omega = tf.expand_dims(1 / eta, -1) * prob
            samples = tf.transpose(samples, [1, 0, 2, 3])
            new_mean = tf.reduce_sum(tf.multiply(samples, tf.expand_dims(tf.expand_dims(omega, -1), -1)), axis=1)
            return t + tf.constant(1, dtype=tf.int32), new_mean

        _, new_mean = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32),
                                                                                      self.previous_solution])
        #assign it to the previous solution for the next uinit
        self.previous_solution.assign(tf.concat([new_mean[:, 1:],
                                                 tf.expand_dims(new_mean[:, -1], 1)], 1))

        resulting_action = new_mean[:, 0]
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
        previous_solution_values = np.tile((self.action_lower_bound + self.action_upper_bound) / 2,
                                           [self.planning_horizon * self.num_agents, 1])
        previous_solution_values = previous_solution_values.reshape([self.num_agents, self.planning_horizon, -1])
        self.previous_solution.assign(previous_solution_values)
