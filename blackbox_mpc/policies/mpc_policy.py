from blackbox_mpc.policies.model_based_base_policy import ModelBasedBasePolicy
from blackbox_mpc.trajectory_evaluators.deterministic import \
    DeterministicTrajectoryEvaluator
from blackbox_mpc.dynamics_handlers.system_dynamics_handler import \
    SystemDynamicsHandler
import tensorflow as tf
import numpy as np


class MPCPolicy(ModelBasedBasePolicy):
    """This is the model predictive control policy for controlling the agent"""
    def __init__(self, trajectory_evaluator=None,
                 optimizer=None, tf_writer=None,
                 log_dir=None, reward_function=None,
                 env_action_space=None, env_observation_space=None,
                 dynamics_function=None, dynamics_handler=None,
                 true_model=False, optimizer_name=None,
                 num_agents=None,
                 save_model_frequency=1,
                 saved_model_dir=None,
                 **optimizer_args):
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
        if trajectory_evaluator is None:
            if dynamics_handler is None:
                trajectory_evaluator = \
                    DeterministicTrajectoryEvaluator(reward_function=reward_function,
                                                     system_dynamics_handler=
                        SystemDynamicsHandler(env_action_space=env_action_space,
                                              env_observation_space=env_observation_space,
                                              true_model=true_model,
                                              dynamics_function=dynamics_function,
                                              log_dir=log_dir,
                                              tf_writer=tf_writer,
                                              save_model_frequency=save_model_frequency,
                                              saved_model_dir=saved_model_dir))
            else:
                trajectory_evaluator = \
                    DeterministicTrajectoryEvaluator(
                        reward_function=reward_function,
                        system_dynamics_handler=dynamics_handler)
        super(MPCPolicy, self).__init__(trajectory_evaluator=
                                        trajectory_evaluator)
        if optimizer is None:
            if optimizer_name == 'CEM':
                from blackbox_mpc.optimizers.cem import CEMOptimizer
                if num_agents is None:
                    raise Exception("Please Specify Num Of Agents in the MPC")
                optimizer = CEMOptimizer(env_action_space=env_action_space,
                                         env_observation_space=env_observation_space,
                                         num_agents=num_agents,
                                         **optimizer_args)
        self._optimizer = optimizer
        self._tf_writer = tf_writer
        self._optimizer.set_trajectory_evaluator(trajectory_evaluator)
        self._act_call_counter = 0
        return

    def act(self,  observations, t, exploration_noise=False):
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
        batched_observations = np.array(observations)
        if len(observations.shape) == 1:
           batched_observations = np.tile(np.expand_dims(observations, 0),
                                          (self._optimizer._num_agents, 1))
        t = tf.constant(t, dtype=tf.int32)
        current_state = tf.cast(batched_observations, dtype=tf.float32)
        mean, next_state, rewards_of_next_state = self._optimizer(current_state, t,
                                                                  tf.constant(exploration_noise,
                                                                             dtype=tf.bool))
        log_results = False
        if log_results:
            if self._tf_writer is not None:
                with self._tf_writer.as_default():
                    tf.summary.scalar('rewards/predicted_reward', tf.reduce_mean(rewards_of_next_state),
                                      step=self._act_call_counter)
        next_observations = next_state
        result_action = mean.numpy()
        next_observations = next_observations.numpy()
        self._act_call_counter += 1
        if len(observations.shape) == 1:
            result_action = result_action[0]
            next_observations = next_observations[0]
            rewards_of_next_state = rewards_of_next_state[0]
        return result_action, next_observations, rewards_of_next_state

    def reset(self):
        """
        This is the reset function for the model predictive control policy, which should be called at the beginning of
        the episode.
        """
        self._optimizer.reset()

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
            dim_U = self._optimizer.dim_U
            dim_O = self._optimizer.dim_O
            dim_S = self._optimizer.dim_S
            action_upper_bound = self._optimizer.action_upper_bound
            action_lower_bound = self._optimizer.action_lower_bound
            num_agents = self._optimizer.num_agents
            deterministic_trajectory_evaluator = self._optimizer.trajectory_evaluator
            planning_horizon = self._optimizer.planning_horizon
            if optimizer_name == 'RandomSearch':
                # 6- define the corresponding optimizer
                from blackbox_mpc.optimizers.random_search import RandomSearchOptimizer
                population_size = tf.constant(1024, dtype=tf.int32)
                self._optimizer = RandomSearchOptimizer(planning_horizon=planning_horizon,
                                                        population_size=population_size,
                                                        dim_U=dim_U,
                                                        dim_O=dim_O,
                                                        action_upper_bound=action_upper_bound,
                                                        action_lower_bound=action_lower_bound,
                                                        trajectory_evaluator=deterministic_trajectory_evaluator,
                                                        num_agents=num_agents)
            elif optimizer_name == 'CEM':
                from blackbox_mpc.optimizers.cem import CEMOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha = 0.1
                epsilon = 0.001
                self._optimizer = CEMOptimizer(planning_horizon=planning_horizon,
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
                from blackbox_mpc.optimizers.cma_es import CMAESOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha_cov = 2.0
                sigma = 1
                self._optimizer = CMAESOptimizer(planning_horizon=planning_horizon,
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
                from blackbox_mpc.optimizers.pi2 import PI2Optimizer
                max_iterations = 5
                population_size = 500
                lamda = 1.0
                self._optimizer = PI2Optimizer(planning_horizon=planning_horizon,
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
                from blackbox_mpc.optimizers.pso import PSOOptimizer
                max_iterations = 5
                population_size = 500
                c1 = 0.3
                c2 = 0.5
                w = 0.2
                initial_velocity_fraction = 0.01
                self._optimizer = PSOOptimizer(planning_horizon=planning_horizon,
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
                from blackbox_mpc.optimizers.spsa import SPSAOptimizer
                max_iterations = 5
                population_size = 500
                alpha = 0.602
                gamma = 0.101
                a_par = 0.01
                noise_parameter = 0.3
                self._optimizer = SPSAOptimizer(planning_horizon=planning_horizon,
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
            self._optimizer = optimizer
        return
