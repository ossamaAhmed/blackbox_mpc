from blackbox_mpc.policies.model_based_base_policy import ModelBasedBasePolicy
from blackbox_mpc.trajectory_evaluators.deterministic import \
    DeterministicTrajectoryEvaluator
from blackbox_mpc.dynamics_handlers.system_dynamics_handler import \
    SystemDynamicsHandler
import tensorflow as tf
import numpy as np


class MPCPolicy(ModelBasedBasePolicy):
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
            This is the model predictive control policy for controlling the agent

        Parameters
        ---------
        trajectory_evaluator: EvaluatorBase
            Defines the trajectory evaluator to be used in the optimizer to
            evaluate trajectories.
        tf_writer: tf.summary
            Tensorflow writer to be used in logging the data.
        optimizer_name: str
            optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].
        env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
        env_observation_space: gym.ObservationSpace
            Defines the observation space of the gym environment.
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler is a handler of the state, actions and targets processing funcs as well
            as the dynamics function.
        reward_function: tf_function
            Defines the reward function with the prototype: tf_func_name(current_state, current_actions, next_state),
            where current_state is BatchXdim_S, next_state is BatchXdim_S and  current_actions is BatchXdim_U.
        true_model: bool
            boolean defining if its a true model dynamics or not.
        log_dir: string
            Defines the log directory to save the normalization statistics in.
        num_agents: tf.int32
            Defines the number of runner running in parallel
        saved_model_dir: string
            Defines the saved model directory where the model is saved in, in case of loading the model.
        save_model_frequency: Int
            Defines how often the model should be saved (defined relative to the number of refining iters)
        optimizer_args: args
            other arguments specific to the optimizer.
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
            if num_agents is None:
                raise Exception("Please Specify Num Of Agents in the MPC")
            if optimizer_name == 'CEM':
                from blackbox_mpc.optimizers.cem import CEMOptimizer
                optimizer = CEMOptimizer(env_action_space=env_action_space,
                                         env_observation_space=env_observation_space,
                                         num_agents=num_agents,
                                         **optimizer_args)
            elif optimizer_name == 'CMA-ES':
                from blackbox_mpc.optimizers.cma_es import CMAESOptimizer
                optimizer = CMAESOptimizer(env_action_space=env_action_space,
                                           env_observation_space=env_observation_space,
                                           num_agents=num_agents,
                                           **optimizer_args)
            elif optimizer_name == 'PI2':
                from blackbox_mpc.optimizers.pi2 import PI2Optimizer
                optimizer = PI2Optimizer(env_action_space=env_action_space,
                                         env_observation_space=env_observation_space,
                                         num_agents=num_agents,
                                         **optimizer_args)
            elif optimizer_name == 'PSO':
                from blackbox_mpc.optimizers.pso import PSOOptimizer
                optimizer = PSOOptimizer(env_action_space=env_action_space,
                                         env_observation_space=env_observation_space,
                                         num_agents=num_agents,
                                         **optimizer_args)
            elif optimizer_name == 'SPSA':
                from blackbox_mpc.optimizers.spsa import SPSAOptimizer
                optimizer = SPSAOptimizer(env_action_space=env_action_space,
                                          env_observation_space=env_observation_space,
                                          num_agents=num_agents,
                                          **optimizer_args)
            elif optimizer_name == 'RandomSearch':
                from blackbox_mpc.optimizers.random_search import RandomSearchOptimizer
                optimizer = RandomSearchOptimizer(env_action_space=env_action_space,
                                                  env_observation_space=env_observation_space,
                                                  num_agents=num_agents,
                                                  **optimizer_args)
        self._optimizer = optimizer
        self._tf_writer = tf_writer
        self._trajectory_evaluator = trajectory_evaluator
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

    def switch_optimizer(self, optimizer=None, optimizer_name='',
                         **optimizer_args):
        """
        This function is used to switch the optimizer of model predictive control policy.

        Parameters
        ----------
        optimizer: OptimizerBaseClass
            Optimizer to be used that optimizes for the best action sequence and returns the first action.
        optimizer_name: str
            optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].
        optimizer_args: args
            other arguments specific to the optimizer.
        """
        if optimizer is None:
            if optimizer_name == 'CEM':
                from blackbox_mpc.optimizers.cem import CEMOptimizer
                optimizer = CEMOptimizer(env_action_space=self._optimizer._env_action_space,
                                         env_observation_space=self._optimizer._env_observation_space,
                                         num_agents=self._optimizer._num_agents,
                                         **optimizer_args)
                self._optimizer = optimizer
            elif optimizer_name == 'CMA-ES':
                from blackbox_mpc.optimizers.cma_es import CMAESOptimizer
                optimizer = CMAESOptimizer(
                    env_action_space=self._optimizer._env_action_space,
                    env_observation_space=self._optimizer._env_observation_space,
                    num_agents=self._optimizer._num_agents,
                    **optimizer_args)
                self._optimizer = optimizer
            elif optimizer_name == 'PI2':
                from blackbox_mpc.optimizers.pi2 import PI2Optimizer
                optimizer = PI2Optimizer(env_action_space=self._optimizer._env_action_space,
                                         env_observation_space=self._optimizer._env_observation_space,
                                         num_agents=self._optimizer._num_agents,
                                         **optimizer_args)
                self._optimizer = optimizer
            elif optimizer_name == 'PSO':
                from blackbox_mpc.optimizers.pso import PSOOptimizer
                optimizer = PSOOptimizer(env_action_space=self._optimizer._env_action_space,
                                         env_observation_space=self._optimizer._env_observation_space,
                                         num_agents=self._optimizer._num_agents,
                                         **optimizer_args)
                self._optimizer = optimizer
            elif optimizer_name == 'SPSA':
                from blackbox_mpc.optimizers.spsa import SPSAOptimizer
                optimizer = SPSAOptimizer(
                    env_action_space=self._optimizer._env_action_space,
                    env_observation_space=self._optimizer._env_observation_space,
                    num_agents=self._optimizer._num_agents,
                    **optimizer_args)
                self._optimizer = optimizer
            elif optimizer_name == 'RandomSearch':
                from blackbox_mpc.optimizers.random_search import \
                    RandomSearchOptimizer
                optimizer = RandomSearchOptimizer(
                    env_action_space=self._optimizer._env_action_space,
                    env_observation_space=self._optimizer._env_observation_space,
                    num_agents=self._optimizer._num_agents,
                    **optimizer_args)
                self._optimizer = optimizer
        else:
            self._optimizer = optimizer
        self._optimizer.set_trajectory_evaluator(self._trajectory_evaluator)
        return
