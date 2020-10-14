"""
tf_neuralmpc/optimizers/pso.py
==============================
"""
import tensorflow as tf
import numpy as np
from tf_neuralmpc.optimizers.optimizer_base import OptimizerBase


class PSOOptimizer(OptimizerBase):
    """This class defines the particle swarm optimizer.
    (https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)"""
    def __init__(self, planning_horizon, max_iterations, population_size,
                 dim_U, dim_O, action_upper_bound, action_lower_bound, num_agents,
                 trajectory_evaluator, c1=tf.constant(0.3, dtype=tf.float32),
                 c2=tf.constant(0.5, dtype=tf.float32), w=tf.constant(0.2, dtype=tf.float32),
                 initial_velocity_fraction=tf.constant(0.01, dtype=tf.float32)):
        """
        This is the initializer function for the particle swarm optimizer.


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
        c1: tf.float32
            Defines the fraction of the local best known position direction.
        c2: tf.float32
            Defines the fraction of the global best known position direction.
        w: tf.float32
            Defines the fraction of the current velocity to use.
        initial_velocity_fraction: tf.float32
           Defines the initial velocity fraction out of the action space.
        """
        super(PSOOptimizer, self).__init__(name=None)
        self.planning_horizon = planning_horizon
        self.dim_U = dim_U
        self.dim_O = dim_O
        self.dim_S = dim_O
        self.trajectory_evaluator = trajectory_evaluator
        self.num_agents = num_agents
        self.solution_dim = [self.num_agents, tf.constant(self.planning_horizon, dtype=tf.int32), self.dim_U]
        self.solution_size = tf.reduce_prod(self.solution_dim)
        self.max_iterations = max_iterations
        self.population_size = population_size
        self.action_upper_bound = action_upper_bound
        self.action_lower_bound = action_lower_bound
        self.action_upper_bound_horizon = tf.tile(np.expand_dims(self.action_upper_bound, 0),
                                                                [self.planning_horizon, 1])
        self.action_lower_bound_horizon = tf.tile(np.expand_dims(self.action_lower_bound, 0),
                                                  [self.planning_horizon, 1])
        self.particle_positions = tf.Variable(tf.zeros([self.population_size, *self.solution_dim], dtype=tf.float32))
        self.particle_velocities = tf.Variable(tf.zeros([self.population_size, *self.solution_dim], dtype=tf.float32))
        self.particle_best_known_position = tf.Variable(tf.zeros([self.population_size, *self.solution_dim],
                                                        dtype=tf.float32))
        self.particle_best_known_reward = tf.Variable(tf.zeros([self.population_size, self.num_agents],
                                                      dtype=tf.float32))

        #global
        self.global_best_known_position = tf.Variable(tf.zeros([*self.solution_dim], dtype=tf.float32))
        self.global_best_known_reward = tf.Variable(tf.zeros([self.num_agents], dtype=tf.float32))
        solution_variance_values = np.tile(np.square(self.action_lower_bound - self.action_upper_bound) / 16,
                                           [self.planning_horizon*self.num_agents, 1])
        solution_variance_values = solution_variance_values.reshape([self.num_agents, self.planning_horizon, -1])
        self.solution_variance = tf.constant(solution_variance_values, dtype=tf.float32)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.initial_velocity_fraction = initial_velocity_fraction
        self.solution = tf.Variable(tf.zeros([self.num_agents, self.dim_U], dtype=tf.float32))
        # exploration
        self.exploration_variance = (np.square(self.action_lower_bound - self.action_upper_bound) / 16) * 0.05
        self.exploration_mean = (self.action_upper_bound + self.action_lower_bound) / 2

    @tf.function
    def __call__(self, current_state, time_step, exploration_noise):
        """
          This is the call function for the particle swarm optimizer.
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
        def continue_condition(t, position):
            result = tf.less(t, self.max_iterations)
            return result

        def iterate(t, position):
            #evaluate each of the particles
            # Evaluate and sort solutions
            feasible_particle_positions = tf.clip_by_value(self.particle_positions, self.action_lower_bound_horizon,
                                                           self.action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(self.particle_positions - feasible_particle_positions, [self.population_size, self.num_agents, -1]),
                              axis=2) ** 2
            self.particle_positions.assign(feasible_particle_positions)

            rewards = self.trajectory_evaluator(current_state, self.particle_positions, time_step) - penalty
            #set the best local known position
            condition = tf.less(self.particle_best_known_reward, rewards)

            new_particle_best_known_position = tf.where(tf.expand_dims(tf.expand_dims(condition, -1), -1), self.particle_positions,
                                                        self.particle_best_known_position)
            self.particle_best_known_position.assign(new_particle_best_known_position)
            new_particle_best_known_reward = tf.where(condition, rewards,
                                                      self.particle_best_known_reward)
            self.particle_best_known_reward.assign(new_particle_best_known_reward)
            #get the global best now

            global_best_known_position_index = tf.math.argmax(self.particle_best_known_reward)
            samples = tf.transpose(self.particle_best_known_position, [1, 0, 2, 3])
            global_best_known_position_index = tf.cast(global_best_known_position_index, dtype=tf.int32) + tf.range(0, samples.shape[0], dtype=tf.int32) * samples.shape[1]
            samples = tf.reshape(samples, [-1, *samples.shape[2:]])
            self.global_best_known_position.assign(tf.gather(samples, global_best_known_position_index))
            samples = tf.reshape(self.particle_best_known_reward, [-1])
            self.global_best_known_reward.assign(tf.gather(samples, global_best_known_position_index))


            #calculate the velocity now
            adapted_particle_velocities = (self.particle_velocities*self.w) + \
                                          (self.particle_best_known_position - self.particle_positions) * self.c1 * tf.random.normal(shape=[], dtype=tf.float32) + \
                                          (self.global_best_known_position - self.particle_positions) * self.c2 * tf.random.normal(shape=[], dtype=tf.float32)
            self.particle_velocities.assign(adapted_particle_velocities)
            self.particle_positions.assign(self.particle_positions + self.particle_velocities)
            return t + tf.constant(1, dtype=tf.int32), self.global_best_known_position
        _ = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32), self.global_best_known_position])
        self.solution.assign(self.global_best_known_position[:, 0, :])
        # update the particles position for the next iteration
        lower_bound_dist = self.global_best_known_position - self.action_lower_bound_horizon
        upper_bound_dist = self.action_upper_bound_horizon - self.global_best_known_position
        constrained_variance = tf.minimum(tf.minimum(tf.square(lower_bound_dist / tf.constant(2, dtype=tf.float32)),
                                                     tf.square(upper_bound_dist / tf.constant(2, dtype=tf.float32))),
                                          self.solution_variance)
        samples_positions = tf.random.truncated_normal([self.population_size,
                                                        *self.solution_dim],
                                                       tf.concat([self.global_best_known_position[:, 1:],
                                                                  tf.expand_dims(self.global_best_known_position[:, -1],
                                                                                 1)], 1),
                                                       tf.sqrt(constrained_variance),
                                                       dtype=tf.float32)
        action_space = self.action_upper_bound_horizon - self.action_lower_bound_horizon
        initial_velocity = self.initial_velocity_fraction * action_space
        samples_velocities = tf.random.uniform([self.population_size, *self.solution_dim], -initial_velocity,
                                               initial_velocity, dtype=tf.float32)
        self.particle_positions.assign(samples_positions)
        self.particle_velocities.assign(samples_velocities)
        self.particle_best_known_position.assign(samples_positions)
        self.particle_best_known_reward.assign(tf.fill([self.population_size, self.num_agents],
                                                       tf.constant(-np.inf, dtype=tf.float32)))
        self.global_best_known_reward.assign(tf.fill([self.num_agents],
                                                     tf.constant(-np.inf, dtype=tf.float32)))
        #end update particles
        resulting_action = self.solution
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
        samples_positions = tf.random.uniform([self.population_size, *self.solution_dim], self.action_lower_bound_horizon,
                                               self.action_upper_bound_horizon, dtype=tf.float32)
        action_space = self.action_upper_bound_horizon - self.action_lower_bound_horizon
        initial_velocity = self.initial_velocity_fraction * action_space
        samples_velocities = tf.random.uniform([self.population_size, *self.solution_dim], -initial_velocity,
                                               initial_velocity, dtype=tf.float32)
        self.particle_positions.assign(samples_positions)
        self.particle_velocities.assign(samples_velocities)
        self.particle_best_known_position.assign(samples_positions)
        self.particle_best_known_reward.assign(tf.fill([self.population_size, self.num_agents],
                                                       tf.constant(-np.inf, dtype=tf.float32)))
        self.global_best_known_reward.assign(tf.fill([self.num_agents],
                                                     tf.constant(-np.inf, dtype=tf.float32)))


