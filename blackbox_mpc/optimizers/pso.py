import tensorflow as tf
import numpy as np
from blackbox_mpc.optimizers.optimizer_base import OptimizerBase


class PSOOptimizer(OptimizerBase):
    def __init__(self, env_action_space, env_observation_space,
                 planning_horizon=50, max_iterations=5, population_size=500,
                 num_agents=5, c1=tf.constant(0.3, dtype=tf.float32),
                 c2=tf.constant(0.5, dtype=tf.float32), w=tf.constant(0.2, dtype=tf.float32),
                 initial_velocity_fraction=tf.constant(0.01, dtype=tf.float32)):
        """
        This class defines the particle swarm optimizer.
        (https://www.cs.tufts.edu/comp/150GA/homeworks/hw3/_reading6%201995%20particle%20swarming.pdf)


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
        c1: tf.float32
            Defines the fraction of the local best known position direction.
        c2: tf.float32
            Defines the fraction of the global best known position direction.
        w: tf.float32
            Defines the fraction of the current velocity to use.
        initial_velocity_fraction: tf.float32
           Defines the initial velocity fraction out of the action space.
        """
        super(PSOOptimizer, self).__init__(name=None,
                                           planning_horizon=planning_horizon,
                                           max_iterations=max_iterations,
                                           num_agents=num_agents,
                                           env_action_space=env_action_space,
                                           env_observation_space=
                                           env_observation_space)
        self._solution_dim = [self._num_agents, tf.constant(self._planning_horizon, dtype=tf.int32), self._dim_U]
        self._solution_size = tf.reduce_prod(self._solution_dim)
        self._population_size = population_size
        self._particle_positions = tf.Variable(tf.zeros([self._population_size, *self._solution_dim], dtype=tf.float32))
        self._particle_velocities = tf.Variable(tf.zeros([self._population_size, *self._solution_dim], dtype=tf.float32))
        self._particle_best_known_position = tf.Variable(tf.zeros([self._population_size, *self._solution_dim],
                                                                  dtype=tf.float32))
        self._particle_best_known_reward = tf.Variable(tf.zeros([self._population_size, self._num_agents],
                                                                dtype=tf.float32))

        #global
        self._global_best_known_position = tf.Variable(tf.zeros([*self._solution_dim], dtype=tf.float32))
        self._global_best_known_reward = tf.Variable(tf.zeros([self._num_agents], dtype=tf.float32))
        solution_variance_values = np.tile(np.square(self._action_lower_bound - self._action_upper_bound) / 16,
                                           [self._planning_horizon * self._num_agents, 1])
        solution_variance_values = solution_variance_values.reshape([self._num_agents, self._planning_horizon, -1])
        self._solution_variance = tf.constant(solution_variance_values, dtype=tf.float32)
        self._c1 = c1
        self._c2 = c2
        self._w = w
        self._initial_velocity_fraction = initial_velocity_fraction
        self._solution = tf.Variable(tf.zeros([self._num_agents, self._dim_U], dtype=tf.float32))

    @tf.function
    def _optimize(self, current_state, time_step):
        def continue_condition(t, position):
            result = tf.less(t, self._max_iterations)
            return result

        def iterate(t, position):
            #evaluate each of the particles
            # Evaluate and sort solutions
            feasible_particle_positions = tf.clip_by_value(self._particle_positions, self._action_lower_bound_horizon,
                                                           self._action_upper_bound_horizon)
            penalty = tf.norm(tf.reshape(self._particle_positions - feasible_particle_positions, [self._population_size, self._num_agents, -1]),
                              axis=2) ** 2
            self._particle_positions.assign(feasible_particle_positions)

            rewards = self._trajectory_evaluator(current_state, self._particle_positions, time_step) - penalty
            #set the best local known position
            condition = tf.less(self._particle_best_known_reward, rewards)

            new_particle_best_known_position = tf.where(tf.expand_dims(tf.expand_dims(condition, -1), -1), self._particle_positions,
                                                        self._particle_best_known_position)
            self._particle_best_known_position.assign(new_particle_best_known_position)
            new_particle_best_known_reward = tf.where(condition, rewards,
                                                      self._particle_best_known_reward)
            self._particle_best_known_reward.assign(new_particle_best_known_reward)
            #get the global best now

            global_best_known_position_index = tf.math.argmax(self._particle_best_known_reward)
            samples = tf.transpose(self._particle_best_known_position, [1, 0, 2, 3])
            global_best_known_position_index = tf.cast(global_best_known_position_index, dtype=tf.int32) + tf.range(0, samples.shape[0], dtype=tf.int32) * samples.shape[1]
            samples = tf.reshape(samples, [-1, *samples.shape[2:]])
            self._global_best_known_position.assign(tf.gather(samples, global_best_known_position_index))
            samples = tf.reshape(self._particle_best_known_reward, [-1])
            self._global_best_known_reward.assign(tf.gather(samples, global_best_known_position_index))


            #calculate the velocity now
            adapted_particle_velocities = (self._particle_velocities * self._w) + \
                                          (self._particle_best_known_position - self._particle_positions) * self._c1 * tf.random.normal(shape=[], dtype=tf.float32) + \
                                          (self._global_best_known_position - self._particle_positions) * self._c2 * tf.random.normal(shape=[], dtype=tf.float32)
            self._particle_velocities.assign(adapted_particle_velocities)
            self._particle_positions.assign(self._particle_positions + self._particle_velocities)
            return t + tf.constant(1, dtype=tf.int32), self._global_best_known_position
        _ = tf.while_loop(cond=continue_condition, body=iterate, loop_vars=[tf.constant(0, dtype=tf.int32), self._global_best_known_position])
        self._solution.assign(self._global_best_known_position[:, 0, :])
        # update the particles position for the next iteration
        lower_bound_dist = self._global_best_known_position - self._action_lower_bound_horizon
        upper_bound_dist = self._action_upper_bound_horizon - self._global_best_known_position
        constrained_variance = tf.minimum(tf.minimum(tf.square(lower_bound_dist / tf.constant(2, dtype=tf.float32)),
                                                     tf.square(upper_bound_dist / tf.constant(2, dtype=tf.float32))),
                                          self._solution_variance)
        samples_positions = tf.random.truncated_normal([self._population_size,
                                                        *self._solution_dim],
                                                       tf.concat([self._global_best_known_position[:, 1:],
                                                                  tf.expand_dims(self._global_best_known_position[:, -1],
                                                                                 1)], 1),
                                                       tf.sqrt(constrained_variance),
                                                       dtype=tf.float32)
        action_space = self._action_upper_bound_horizon - self._action_lower_bound_horizon
        initial_velocity = self._initial_velocity_fraction * action_space
        samples_velocities = tf.random.uniform([self._population_size, *self._solution_dim], -initial_velocity,
                                               initial_velocity, dtype=tf.float32)
        self._particle_positions.assign(samples_positions)
        self._particle_velocities.assign(samples_velocities)
        self._particle_best_known_position.assign(samples_positions)
        self._particle_best_known_reward.assign(tf.fill([self._population_size, self._num_agents],
                                                        tf.constant(-np.inf, dtype=tf.float32)))
        self._global_best_known_reward.assign(tf.fill([self._num_agents],
                                                      tf.constant(-np.inf, dtype=tf.float32)))
        #end update particles
        resulting_action = self._solution
        return resulting_action

    def reset(self):
        """
         This method resets the optimizer to its default state at the beginning of the trajectory/episode.
         """
        samples_positions = tf.random.uniform([self._population_size, *self._solution_dim], self._action_lower_bound_horizon,
                                              self._action_upper_bound_horizon, dtype=tf.float32)
        action_space = self._action_upper_bound_horizon - self._action_lower_bound_horizon
        initial_velocity = self._initial_velocity_fraction * action_space
        samples_velocities = tf.random.uniform([self._population_size, *self._solution_dim], -initial_velocity,
                                               initial_velocity, dtype=tf.float32)
        self._particle_positions.assign(samples_positions)
        self._particle_velocities.assign(samples_velocities)
        self._particle_best_known_position.assign(samples_positions)
        self._particle_best_known_reward.assign(tf.fill([self._population_size, self._num_agents],
                                                        tf.constant(-np.inf, dtype=tf.float32)))
        self._global_best_known_reward.assign(tf.fill([self._num_agents],
                                                      tf.constant(-np.inf, dtype=tf.float32)))
        return


