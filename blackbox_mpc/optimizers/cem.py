import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class CEMOptimizer(OptimizerBase):
    """This Class defines a Cross-Entropy Method optimizer.
       (http://web.mit.edu/6.454/www/www_fall_2003/gew/CEtutorial.pdf)"""
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, max_iterations=5, population_size=500,
                 num_elite=50, num_agents=5,
                 epsilon=0.001, alpha=0.25):
        """
        This is the initializer function for the Cross-Entropy Method Optimizer.


        Parameters
        ---------
        planning_horizon: Int
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        max_iterations: tf.int32
            Defines the maximimum iterations for the CEM optimizer to refine its guess for the optimal solution.
        population_size: tf.int32
            Defines the population size of the particles evaluated at each iteration.
        num_elite: tf.int32
            Defines the number of elites kept for the next iteration from the population.
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
        epsilon: tf.float32
            Defines the epsilon threshold for the difference between iterations solutions so that
            the optimizer returns the solution earlier than max iterations.
        alpha: tf.float32
            Defines the weight of the solution at t-1 in determining the solution at t,
            ex: mean = alpha*old_mean + (1-alpha)*new_mean.
        """
        super(CEMOptimizer, self).__init__(name=None,
                                           planning_horizon=planning_horizon,
                                           max_iterations=max_iterations,
                                           num_agents=num_agents,
                                           env_action_space=env_action_space,
                                           env_observation_space=
                                           env_observation_space)
        self._solution_dim = [self._num_agents, self._planning_horizon,
                              self._dim_U]
        self._elites_dim = [int(self._num_agents), int(num_elite),
                            int(self._planning_horizon),
                            int(self._dim_U)]
        self._population_size = population_size
        self._num_elite = num_elite
        self._epsilon = epsilon
        self._alpha = alpha
        previous_solution_values = np.tile((self._action_lower_bound +
                                            self._action_upper_bound) / 2,
                                           [self._planning_horizon *
                                            self._num_agents, 1])
        previous_solution_values = previous_solution_values.reshape(
            [self._num_agents, self._planning_horizon, -1])
        self._previous_solution = tf.Variable(
            tf.zeros(shape=previous_solution_values.shape, dtype=tf.float32))
        self._previous_solution.assign(previous_solution_values)
        solution_variance_values = np.tile(np.square(self._action_lower_bound -
                                                     self._action_upper_bound) / 16,
                                           [self._planning_horizon *
                                            self._num_agents, 1])
        solution_variance_values = solution_variance_values.reshape(
            [self._num_agents, self._planning_horizon, -1])
        self._solution_variance = tf.Variable(tf.zeros(
            shape=solution_variance_values.shape, dtype=tf.float32))
        self._solution_variance.assign(solution_variance_values)

    @tf.function
    def _optimize(self, current_state, time_step):
        """
       This is the call function for the Cross-Entropy Method Optimizer.
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
        def continue_condition(t, mean, variance):
            result = tf.less(t, self._max_iterations)
            return result

        def iterate(t, mean, variance):
            lower_bound_dist = mean - self._action_lower_bound_horizon
            upper_bound_dist = self._action_upper_bound_horizon - mean
            constrained_variance = tf.minimum(tf.minimum(
                tf.square(lower_bound_dist /
                          tf.constant(2, dtype=tf.float32)),
                tf.square(upper_bound_dist /
                          tf.constant(2, dtype=tf.float32))),
                                              variance)
            #sample population
            samples = tf.random.truncated_normal([self._population_size,
                                                  *self._solution_dim],
                                                  mean,
                                                  tf.sqrt(constrained_variance),
                                                 dtype=tf.float32)
            rewards = self._trajectory_evaluator(current_state, samples,
                                                 time_step)
            rewards = tf.transpose(rewards, [1, 0])
            values, indices = tf.nn.top_k(rewards, k=self._num_elite,
                                          sorted=True)
            samples = tf.transpose(samples, [1, 0, 2, 3])
            elites = tf.zeros(self._elites_dim, dtype=tf.float32)
            for agent in tf.range(self._num_agents):
                elites = tf.concat([elites[:agent],
                                    tf.expand_dims(tf.gather(samples[agent],
                                                             indices[agent]), 0),
                                    tf.zeros([samples.shape[0] - 1 - agent,
                                              self._num_elite,
                                              *samples.shape[2:]],
                                             dtype=tf.float32)],
                                   axis=0)
                elites.set_shape(self._elites_dim)
            new_mean = tf.reduce_mean(elites, axis=1)
            new_variance = tf.reduce_mean(tf.square(elites -
                                                    tf.tile(
                                                        tf.expand_dims(
                                                            new_mean, 1),
                                                        [1, tf.shape(elites)
                                                        [1], 1, 1])),
                                          axis=1)

            mean = self._alpha * mean + (tf.constant(
                1, dtype=tf.float32) - self._alpha) * new_mean
            variance = self._alpha * variance + (
                    tf.constant(1, dtype=tf.float32) - self._alpha) * \
                       new_variance

            return t + tf.constant(1, dtype=tf.int32), mean, variance

        num_optimization_iters, mean, variance = tf.while_loop(
            cond=continue_condition, body=iterate,
            loop_vars=[tf.constant(0, dtype=tf.int32), self._previous_solution,
                       self._solution_variance])
        #TODO: the below line is causing problems with cheetah env
        # self.previous_solution.assign(mean)
        resulting_action = mean[:, 0]
        return resulting_action

    def reset(self):
        """
          This method resets the optimizer to its default state at the beginning of the trajectory/episode.
          """
        previous_solution_values = np.tile((self._action_lower_bound +
                                            self._action_upper_bound) / 2,
                                           [self._planning_horizon *
                                            self._num_agents, 1])
        previous_solution_values = \
            previous_solution_values.reshape([self._num_agents,
                                              self._planning_horizon, -1])
        self._previous_solution.assign(previous_solution_values)
