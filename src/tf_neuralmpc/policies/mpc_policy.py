"""
tf_neuralmpc/policies/mpc_policy.py
===================================
"""
from tf_neuralmpc.policies.model_based_base_policy import ModelBasedBasePolicy
import tensorflow as tf


class MPCPolicy(ModelBasedBasePolicy):
    """This is the model predictive control policy for controlling the agent"""
    def __init__(self, system_dynamics_handler,
                 optimizer, tf_writer=None):
        """
            This is the initializer function for the model predictive control policy.

        Parameters
        ---------
        system_dynamics_handler: SystemDynamicsHandler
            Defines the system dynamics handler to be used in the policy for preprocessing the observations and
            postprocessing the state to observations.
        optimizer: OptimizerBaseClass
            Optimizer to be used that optimizes for the best action sequence and returns the first action.
        tf_writer: tf.summary
            Tensorflow writer to be used in logging the data.
        """
        super(MPCPolicy, self).__init__(system_dynamics=system_dynamics_handler)
        self.system_dynamics_handler = system_dynamics_handler
        self.tf_writer = tf_writer
        self.optimizer = optimizer
        self.act_call_counter = 0
        return

    def act(self,  observations, t, exploration_noise=False, log_results=True):
        """
        This is the act function for the model predictive control policy, which should be called to provide the action
        to be executed at the current time step.


        Parameters
        ---------
        observations: tf.float32
            Defines the current observations received from the environment.
        t: tf.float32
            Defines the current timestep.
        exploration_noise: bool
            Defines if exploration noise should be added to the action to be executed.
        log_results: bool
            Defines if results should be logged to tensorboard or not.


        Returns
        -------
        action: tf.float32
            The action to be executed for each of the runner (dims = runner X dim_U)
        next_observations: tf.float32
            The next observations predicted using the dynamics function learned so far.
        rewards_of_next_state: tf.float32
            The predicted reward if the action was executed using the predicted observations.
        """
        t = tf.constant(t, dtype=tf.int32)
        current_state = tf.cast(observations, dtype=tf.float32)
        mean, next_state, rewards_of_next_state = self.optimizer(current_state, t,
                                                                 tf.constant(exploration_noise,
                                                                             dtype=tf.bool))
        if log_results:
            if self.tf_writer is not None:
                with self.tf_writer.as_default():
                    tf.summary.scalar('rewards/predicted_reward', tf.reduce_mean(rewards_of_next_state),
                                      step=self.act_call_counter)
        next_observations = next_state
        result_action = mean.numpy()
        next_observations = next_observations.numpy()
        self.act_call_counter += 1
        return result_action, next_observations, rewards_of_next_state

    def reset(self):
        """
        This is the reset function for the model predictive control policy, which should be called at the beginning of
        the episode.
        """
        self.optimizer.reset()

    def switch_optimizer(self, optimizer=None, optimizer_name=''):
        """
        This function is used to switch the optimizer of model predictive control policy.

        Parameters
        ----------
        optimizer: OptimizerBaseClass
            Optimizer to be used that optimizes for the best action sequence and returns the first action.
        optimizer_name: str
            optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].
        """
        if optimizer is None:
            dim_U = self.optimizer.dim_U
            dim_O = self.optimizer.dim_O
            dim_S = self.optimizer.dim_S
            action_upper_bound = self.optimizer.action_upper_bound
            action_lower_bound = self.optimizer.action_lower_bound
            num_agents = self.optimizer.num_agents
            deterministic_trajectory_evaluator = self.optimizer.trajectory_evaluator
            planning_horizon = self.optimizer.planning_horizon
            if optimizer_name == 'RandomSearch':
                # 6- define the corresponding optimizer
                from tf_neuralmpc.optimizers.random_search import RandomSearchOptimizer
                population_size = tf.constant(1024, dtype=tf.int32)
                self.optimizer = RandomSearchOptimizer(planning_horizon=planning_horizon,
                                                     population_size=population_size,
                                                     dim_U=dim_U,
                                                     dim_O=dim_O,
                                                     action_upper_bound=action_upper_bound,
                                                     action_lower_bound=action_lower_bound,
                                                     trajectory_evaluator=deterministic_trajectory_evaluator,
                                                     num_agents=num_agents)
            elif optimizer_name == 'CEM':
                from tf_neuralmpc.optimizers.cem import CEMOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha = 0.1
                epsilon = 0.001
                self.optimizer = CEMOptimizer(planning_horizon=planning_horizon,
                                            max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                            population_size=tf.constant(population_size, dtype=tf.int32),
                                            num_elite=tf.constant(num_elites, dtype=tf.int32),
                                            dim_U=dim_U,
                                            dim_O=dim_O,
                                            action_upper_bound=action_upper_bound,
                                            action_lower_bound=action_lower_bound,
                                            epsilon=tf.constant(epsilon, dtype=tf.float32),
                                            alpha=tf.constant(alpha, dtype=tf.float32),
                                            num_agents=num_agents,
                                            trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'CMA-ES':
                from tf_neuralmpc.optimizers.cma_es import CMAESOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha_cov = 2.0
                sigma = 1
                self.optimizer = CMAESOptimizer(planning_horizon=planning_horizon,
                                                max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                                population_size=tf.constant(population_size,
                                                                            dtype=tf.int32),
                                                num_elite=tf.constant(num_elites, dtype=tf.int32),
                                                h_sigma=tf.constant(sigma, dtype=tf.float32),
                                                alpha_cov=tf.constant(alpha_cov, dtype=tf.float32),
                                                dim_U=dim_U,
                                                dim_O=dim_O,
                                                action_upper_bound=action_upper_bound,
                                                action_lower_bound=action_lower_bound,
                                                num_agents=num_agents,
                                                trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'PI2':
                from tf_neuralmpc.optimizers.pi2 import PI2Optimizer
                max_iterations = 5
                population_size = 500
                lamda = 1.0
                self.optimizer = PI2Optimizer(planning_horizon=planning_horizon,
                                              max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                              population_size=tf.constant(population_size,
                                                                          dtype=tf.int32),
                                              lamda=tf.constant(lamda, dtype=tf.float32),
                                            dim_U=dim_U,
                                            dim_O=dim_O,
                                            action_upper_bound=action_upper_bound,
                                            action_lower_bound=action_lower_bound,
                                            num_agents=num_agents,
                                            trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'PSO':
                from tf_neuralmpc.optimizers.pso import PSOOptimizer
                max_iterations = 5
                population_size = 500
                c1 = 0.3
                c2 = 0.5
                w = 0.2
                initial_velocity_fraction = 0.01
                self.optimizer = PSOOptimizer(planning_horizon=planning_horizon,
                                            max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                            population_size=tf.constant(population_size, dtype=tf.int32),
                                            c1=tf.constant(c1, dtype=tf.float32),
                                            c2=tf.constant(c2, dtype=tf.float32),
                                            w=tf.constant(w, dtype=tf.float32),
                                            initial_velocity_fraction=tf.constant(
                                                  initial_velocity_fraction, dtype=tf.float32),
                                            dim_U=dim_U,
                                            dim_O=dim_O,
                                            action_upper_bound=action_upper_bound,
                                            action_lower_bound=action_lower_bound,
                                            num_agents=num_agents,
                                            trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'SPSA':
                from tf_neuralmpc.optimizers.spsa import SPSAOptimizer
                max_iterations = 5
                population_size = 500
                alpha = 0.602
                gamma = 0.101
                a_par = 0.01
                noise_parameter = 0.3
                self.optimizer = SPSAOptimizer(planning_horizon=planning_horizon,
                                               max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                               population_size=tf.constant(population_size,
                                                                           dtype=tf.int32),
                                               alpha=tf.constant(alpha, dtype=tf.float32),
                                               gamma=tf.constant(gamma, dtype=tf.float32),
                                               a_par=tf.constant(a_par, dtype=tf.float32),
                                               noise_parameter=tf.constant(noise_parameter,
                                                                           dtype=tf.float32),
                                             dim_U=dim_U,
                                             dim_O=dim_O,
                                             action_upper_bound=action_upper_bound,
                                             action_lower_bound=action_lower_bound,
                                             num_agents=num_agents,
                                             trajectory_evaluator=deterministic_trajectory_evaluator)
        else:
            self.optimizer = optimizer
        return
