"""
tf_neuralmpc/runner/runner.py
=============================
"""
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
import logging
import numpy as np
import tensorflow as tf
from tf_neuralmpc.policies.model_free_base_policy import ModelFreeBasePolicy
from tf_neuralmpc.policies.random_policy import RandomPolicy


class Runner(object):
    """This is the runner class which handles the interaction of the different agents with the enviornment,
       a runnner takes in a controller and possibly a dynamics model which are used further to iteract with
       the environment"""
    def __init__(self, env, log_path=None, num_of_agents=1):
        """
        This is the initializer function for the runner class.


        Parameters
        ---------
        env: [mujoco_env.MujocoEnv, SubprocVecEnv]
            Defines two environment_utils in a list, the first being a single environment that could possibly be used to
            record a rollout in a video and the second is a vectorized environment that could possibly be used for collecting the
            rollouts.
        log_path:
            Defines the log directory to be used to log the results while training and collecting samples.
        num_of_agents: Int
            The number of agents running in parallel in the environment.
        """
        self.recording_env, self.env = env
        self.log_path = log_path
        if log_path is not None:
            self.tf_writer = tf.summary.create_file_writer(self.log_path)
        else:
            self.tf_writer = None
        self.num_of_agents = num_of_agents
        self.act_counter = 0
        self.episode_counter = 0
        return

    def sample(self, horizon, policy, exploration_noise=False):
        """
        This is the sampling function for the runner class which samples one episode with a specified length
        using the provided policy.


        Parameters
        ---------
        horizon: Int
            The task horizon/ episode length.
        policy: ModelBasedBasePolicy or ModelFreeBasePolicy
            The policy to be used in collecting the episodes from the different agents.
        exploration_noise: bool
            If noise should be added to the actions to help in exploration.

        Returns
        -------
        result: dict
            returns the episode rollouts results for all the agents in the parallelized environment,
            it has the form of {observations, actions, rewards, reward_sum}
        """
        times, observations, actions, rewards, reward_sum, done = \
            [], [self.env.reset()], [], [], 0, False
        if not isinstance(policy, ModelFreeBasePolicy):
            predicted_reward = 0

        for t in range(horizon):
            start = time.time()
            if not isinstance(policy, ModelFreeBasePolicy):
                action_to_execute, expected_obs, expected_reward = policy.act(observations[t], t,
                                                                              exploration_noise)
                predicted_reward += expected_reward
            else:
                action_to_execute = policy.act(observations[t], t)
                action_to_execute = action_to_execute.numpy()
            actions.append(action_to_execute)
            times.append(time.time() - start)
            obs, reward, done, info = self.env.step(actions[t])
            if self.log_path is not None:
                if not isinstance(policy, RandomPolicy):
                    with self.tf_writer.as_default():
                        tf.summary.scalar('rewards/actual_reward', np.mean(reward), step=self.act_counter)
                if not isinstance(policy, ModelFreeBasePolicy):
                    with self.tf_writer.as_default():
                        tf.summary.scalar('states/predicted_observations_abs_error', np.mean(np.sum(np.abs(expected_obs - obs),
                                                                                             axis=1)),
                                          step=self.act_counter)
                        tf.summary.scalar('rewards/predicted_reward_abs_error', np.mean(np.abs(expected_reward - reward)),
                                          step=self.act_counter)
                        self.act_counter += 1
            observations.append(obs)
            rewards.append(reward)
            reward_sum += reward
            if t >= horizon - 1:
                if self.log_path is not None:
                    if not isinstance(policy, RandomPolicy):
                        with self.tf_writer.as_default():
                            tf.summary.scalar('rewards/actual_episode_reward', np.mean(reward_sum),
                                              step=self.episode_counter)
                    if not isinstance(policy, ModelFreeBasePolicy):
                        with self.tf_writer.as_default():
                            tf.summary.scalar('rewards/predicted_episode_reward', np.mean(predicted_reward),
                                              step=self.episode_counter)
                            self.episode_counter += 1
                break
        logging.info("Average action selection time: " + str(np.mean(times)))
        logging.info("Rollout length: " + str(len(actions)))

        return {"observations": np.array(observations),
                "actions": np.array(actions),
                "rewards": np.array(rewards),
                "reward_sum": reward_sum}

    def record_rollout(self, horizon, policy, record_file_path):
        """
        This is the recording function for the runner class which samples one episode with a specified length
        using the provided policy and records it in a video.


        Parameters
        ---------
        horizon: Int
            The task horizon/ episode length.
        policy: ModelBasedBasePolicy or ModelFreeBasePolicy
            The policy to be used in collecting the episodes from the different agents.
        record_file_path: String
            specified the file path to save the video that will be recorded in.
        """
        recorder = VideoRecorder(self.recording_env, record_file_path + '_episode_' + str(self.episode_counter) + '.mp4')
        observations = np.tile(np.expand_dims(self.recording_env.reset(), 0), (self.num_of_agents, 1))
        for t in range(horizon):
            recorder.capture_frame()
            if not isinstance(policy, ModelFreeBasePolicy):
                action_to_execute, expected_obs, expected_reward = policy.act(observations, t,
                                                                              exploration_noise=False,
                                                                              log_results=False)
            else:
                action_to_execute = policy.act(observations, t)
            obs, reward, done, info = self.recording_env.step(action_to_execute[0])
            observations = np.tile(np.expand_dims(obs, 0), (self.num_of_agents, 1))
        recorder.capture_frame()
        recorder.close()
        return

    def perform_rollouts(self, number_of_rollouts, task_horizon, policy, exploration_noise=False):
        """
        This is the perform_rollouts function for the runner class which samples n episodes with a specified length
        using the provided policy.


        Parameters
        ---------
        number_of_rollouts: Int
            Number of rollouts/ episodes to perform for each of the agents in the vectorized environment.
        task_horizon: Int
            The task horizon/ episode length.
        policy: ModelBasedBasePolicy or ModelFreeBasePolicy
            The policy to be used in collecting the episodes from the different agents.
        exploration_noise: bool
            If noise should be added to the actions to help in exploration.

        Returns
        -------
        traj_obs: [np.float32]
            List with length=number_of_rollouts which holds the observations starting from the reset observations.
        traj_acs: [np.float32]
            List with length=number_of_rollouts which holds the actions taken by the policy.
        traj_rews: [np.float32]
            List with length=number_of_rollouts which holds the rewards taken by the policy.
        """
        traj_obs, traj_acs, traj_rews = [], [], []
        samples = []
        logging.info("Started collecting samples for rollouts")
        for i in range(number_of_rollouts):
            samples.append(
                self.sample(
                    task_horizon, policy, exploration_noise=exploration_noise))
            traj_obs.append(samples[-1]["observations"])
            traj_acs.append(samples[-1]["actions"])
            traj_rews.append(samples[-1]["rewards"])
        logging.info("Finished collecting samples for rollout")
        return traj_obs, traj_acs, traj_rews

    def learn_dynamics_iteratively_w_mpc(self, number_of_initial_rollouts,
                                         number_of_rollouts_for_refinement,
                                         number_of_refinement_steps,
                                         task_horizon,
                                         initial_policy=None,
                                         mpc_policy=None,
                                         planning_horizon=None,
                                         state_reward_function=None,
                                         actions_reward_function=None,
                                         optimizer=None,
                                         optimizer_name='RandomSearch',
                                         nn_optimizer=tf.keras.optimizers.Adam,
                                         system_dynamics_handler=None,
                                         dynamics_function=None,
                                         exploration_noise=False,
                                         epochs=30, learning_rate=1e-3,
                                         validation_split=0.2, batch_size=128,
                                         normalization=True):
        """
        This is the learn dynamics function iteratively using mpc policy
        for the runner class which samples n rollouts using an initial policy and then
        uses these rollouts to learn a dynamics function for the system which is then used to sample further rollouts
        to refine the dynamics function.


        Parameters
        ---------
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        number_of_initial_rollouts: Int
            Number of initial rollouts/ episodes to perform for each of the agents in the vectorized environment.
        number_of_rollouts_for_refinement: Int
            Number of refinement rollouts/ episodes to perform for each of the agents in the vectorized environment.
        number_of_refinement_steps: Int
            Number of refinemnet steps train, collect, train..etc to run for.
        task_horizon: Int
            The task horizon/ episode length.
        initial_policy: ModelBasedBasePolicy or ModelFreeBasePolicy
            The policy to be used in collecting the initial episodes from the different agents.
        mpc_policy: ModelBasedBasePolicy
            The policy to be used in collecting the further episodes from the different agents to refine the model
            estimate.
        system_dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler is a handler of the state, actions and targets processing funcs as well
            as the dynamics function.
        exploration_noise: bool
            If noise should be added to the actions to help in exploration.
        learning_rate: float
            Learning rate to be used in training the dynamics function.
        epochs: Int
            Number of epochs to be used in training the dynamics function everytime train is called.
        validation_split: float32
            Defines the validation split to be used of the rollouts collected.
        batch_size: int
            Defines the batch size to be used for training the model.
        nn_optimizer: tf.keras.optimizers
            Defines the optimizer to use with the neural network.
        normalization: bool
            Defines if the dynamics function should be trained with normalization or not.
        state_reward_function: tf_function
            Defines the state reward function with the prototype: tf_func_name(current_state, next_state),
            where current_state is BatchXdim_S and next_state is BatchXdim_S.
        actions_reward_function: tf_function
            Defines the action reward function with the prototype: tf_func_name(current_actions),
            where current_actions is BatchXdim_U.
        planning_horizon: tf.int32
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        optimizer: OptimizerBaseClass
            Optimizer to be used that optimizes for the best action sequence and returns the first action.
        optimizer_name: str
            optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].

        Returns
        -------
        system_dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler holds the trained system dynamics.
        mpc_policy: ModelBasedBasePolicy
            The policy that was refined to be used as a control policy
        """
        if mpc_policy is None:
            if system_dynamics_handler is None and dynamics_function is None:
                raise Exception("you need to specify either a handler or a dynamics function")

            if system_dynamics_handler is not None and dynamics_function is not None:
                raise Exception("you cannot specify both a handler or a dynamics function")

        action_upper_bound = tf.constant(self.recording_env.action_space.high, dtype=tf.float32)
        action_lower_bound = tf.constant(self.recording_env.action_space.low, dtype=tf.float32)
        if initial_policy is None:
            from tf_neuralmpc.policies.random_policy import RandomPolicy
            initial_policy = RandomPolicy(number_of_agents=self.num_of_agents,
                                          action_lower_bound=action_lower_bound,
                                          action_upper_bound=action_upper_bound)
        number_of_agents = tf.constant(self.num_of_agents, dtype=tf.int32)
        state_size = self.recording_env.observation_space.shape[0]
        input_size = self.recording_env.action_space.shape[0]
        if mpc_policy is None:
            if system_dynamics_handler is None:
                from tf_neuralmpc.dynamics_handlers.system_dynamics_handler import SystemDynamicsHandler
                system_dynamics_handler = SystemDynamicsHandler(dynamics_function=dynamics_function,
                                                                dim_O=state_size,
                                                                dim_U=input_size,
                                                                num_of_agents=number_of_agents,
                                                                log_dir=self.log_path,
                                                                true_model=False,
                                                                normalization=normalization)
            mpc_policy = self.make_mpc_policy(state_reward_function,
                                              actions_reward_function,
                                              planning_horizon,
                                              optimizer_name=optimizer_name,
                                              optimizer=optimizer,
                                              system_dynamics_handler=system_dynamics_handler,
                                              true_model=False,
                                              normalization=normalization)
        traj_obs, traj_acs, traj_rews = \
            self.perform_rollouts(number_of_initial_rollouts, task_horizon, initial_policy,
                                  exploration_noise=exploration_noise)
        if number_of_initial_rollouts > 0:
            mpc_policy.system_dynamics_handler.train(traj_obs, traj_acs, traj_rews, validation_split=validation_split,
                                                      batch_size=batch_size, learning_rate=learning_rate,
                                                      epochs=epochs, nn_optimizer=nn_optimizer)
            logging.info("Trained initial system model")
        for i in range(number_of_refinement_steps):
            traj_obs, traj_acs, traj_rews = \
                self.perform_rollouts(number_of_rollouts_for_refinement, task_horizon, mpc_policy,
                                      exploration_noise=exploration_noise)
            mpc_policy.system_dynamics_handler.train(traj_obs, traj_acs, traj_rews, validation_split=validation_split,
                                                      batch_size=batch_size, learning_rate=learning_rate,
                                                      epochs=epochs, nn_optimizer=nn_optimizer)
        traj_obs, traj_acs, traj_rews = \
            self.perform_rollouts(number_of_rollouts_for_refinement, task_horizon, mpc_policy,
                                  exploration_noise=exploration_noise)
        return mpc_policy.system_dynamics_handler, mpc_policy

    def make_mpc_policy(self, state_reward_function,
                        actions_reward_function, planning_horizon,
                        optimizer_name='RandomSearch', system_dynamics_handler=None,
                        dynamics_function=None, true_model=True,
                        optimizer=None, normalization=False):
        """
        This is the make mpc policy which returns an mpc policy with the defined cost funcs, optimzer
        and the dynamics function.


        Parameters
        ---------
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        system_dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler is a handler of the state, actions and targets processing funcs as well
            as the dynamics function.
        normalization: bool
            Defines if the dynamics function should be trained with normalization or not.
        state_reward_function: tf_function
            Defines the state reward function with the prototype: tf_func_name(current_state, next_state),
            where current_state is BatchXdim_S and next_state is BatchXdim_S.
        actions_reward_function: tf_function
            Defines the action reward function with the prototype: tf_func_name(current_actions),
            where current_actions is BatchXdim_U.
        planning_horizon: tf.int32
            Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
        optimizer: OptimizerBaseClass
            Optimizer to be used that optimizes for the best action sequence and returns the first action.
        optimizer_name: str
            optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].
        true_model: bool
            boolean defining if its a true model dynamics or not.

        Returns
        -------
        mpc_policy: ModelBasedBasePolicy
            The mpc policy that will be used with the system dynamics.
        """
        if system_dynamics_handler is None and dynamics_function is None:
            raise Exception("you need to specify either a handler or a dynamics function")
        if system_dynamics_handler is not None and dynamics_function is not None:
            raise Exception("you cannot specify both a handler or a dynamics function")

        state_size = tf.constant(self.recording_env.observation_space.shape[0], dtype=tf.int32)
        input_size = tf.constant(self.recording_env.action_space.shape[0], dtype=tf.int32)
        number_of_agents = tf.constant(self.num_of_agents, dtype=tf.int32)
        if system_dynamics_handler is None:
            from tf_neuralmpc.dynamics_handlers.system_dynamics_handler import SystemDynamicsHandler
            number_of_agents = tf.constant(self.num_of_agents, dtype=tf.int32)
            system_dynamics_handler = SystemDynamicsHandler(dynamics_function=dynamics_function,
                                                            dim_O=state_size,
                                                            dim_U=input_size,
                                                            num_of_agents=number_of_agents,
                                                            log_dir=self.log_path,
                                                            true_model=true_model,
                                                            normalization=normalization)

        if optimizer is None:
            from tf_neuralmpc.trajectory_evaluators.deterministic import DeterministicTrajectoryEvaluator
            planning_horizon = tf.constant(planning_horizon, dtype=tf.int32)
            deterministic_trajectory_evaluator = DeterministicTrajectoryEvaluator(
                state_reward_function=state_reward_function,
                actions_reward_function=actions_reward_function,
                planning_horizon=planning_horizon,
                dim_U=input_size,
                dim_O=state_size,
                system_dynamics_handler=system_dynamics_handler)
            action_upper_bound = tf.constant(self.recording_env.action_space.high, dtype=tf.float32)
            action_lower_bound = tf.constant(self.recording_env.action_space.low, dtype=tf.float32)
            if optimizer_name == 'RandomSearch':
                # 6- define the corresponding optimizer
                from tf_neuralmpc.optimizers.random_search import RandomSearchOptimizer
                population_size = tf.constant(1024, dtype=tf.int32)
                my_optimizer = RandomSearchOptimizer(planning_horizon=planning_horizon,
                                                     population_size=population_size,
                                                     dim_U=input_size,
                                                     dim_O=state_size,
                                                     action_upper_bound=action_upper_bound,
                                                     action_lower_bound=action_lower_bound,
                                                     trajectory_evaluator=deterministic_trajectory_evaluator,
                                                     num_agents=number_of_agents)
            elif optimizer_name == 'CEM':
                from tf_neuralmpc.optimizers.cem import CEMOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha = 0.1
                epsilon = 0.001
                my_optimizer = CEMOptimizer(planning_horizon=planning_horizon,
                                            max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                            population_size=tf.constant(population_size, dtype=tf.int32),
                                            num_elite=tf.constant(num_elites, dtype=tf.int32),
                                            dim_U=input_size,
                                            dim_O=state_size,
                                            action_upper_bound=action_upper_bound,
                                            action_lower_bound=action_lower_bound,
                                            epsilon=tf.constant(epsilon, dtype=tf.float32),
                                            alpha=tf.constant(alpha, dtype=tf.float32),
                                            num_agents=number_of_agents,
                                            trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'CMA-ES':
                from tf_neuralmpc.optimizers.cma_es import CMAESOptimizer
                max_iterations = 5
                population_size = 500
                num_elites = 50
                alpha_cov = 2.0
                sigma = 1
                my_optimizer = CMAESOptimizer(planning_horizon=planning_horizon,
                                                max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                                population_size=tf.constant(population_size,
                                                                            dtype=tf.int32),
                                                num_elite=tf.constant(num_elites, dtype=tf.int32),
                                                h_sigma=tf.constant(sigma, dtype=tf.float32),
                                                alpha_cov=tf.constant(alpha_cov, dtype=tf.float32),
                                                dim_U=input_size,
                                                dim_O=state_size,
                                                action_upper_bound=action_upper_bound,
                                                action_lower_bound=action_lower_bound,
                                                num_agents=number_of_agents,
                                                trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'PI2':
                from tf_neuralmpc.optimizers.pi2 import PI2Optimizer
                max_iterations = 5
                population_size = 500
                lamda = 1.0
                my_optimizer = PI2Optimizer(planning_horizon=planning_horizon,
                                              max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                              population_size=tf.constant(population_size,
                                                                          dtype=tf.int32),
                                              lamda=tf.constant(lamda, dtype=tf.float32),
                                              dim_U=input_size,
                                              dim_O=state_size,
                                              action_upper_bound=action_upper_bound,
                                              action_lower_bound=action_lower_bound,
                                              num_agents=number_of_agents,
                                              trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'PSO':
                from tf_neuralmpc.optimizers.pso import PSOOptimizer
                max_iterations = 5
                population_size = 500
                c1 = 0.3
                c2 = 0.5
                w = 0.2
                initial_velocity_fraction = 0.01
                my_optimizer = PSOOptimizer(planning_horizon=planning_horizon,
                                            max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                            population_size=tf.constant(population_size, dtype=tf.int32),
                                            c1=tf.constant(c1, dtype=tf.float32),
                                            c2=tf.constant(c2, dtype=tf.float32),
                                            w=tf.constant(w, dtype=tf.float32),
                                            initial_velocity_fraction=tf.constant(
                                                  initial_velocity_fraction, dtype=tf.float32),
                                            dim_U=input_size,
                                            dim_O=state_size,
                                            action_upper_bound=action_upper_bound,
                                            action_lower_bound=action_lower_bound,
                                            num_agents=number_of_agents,
                                            trajectory_evaluator=deterministic_trajectory_evaluator)
            elif optimizer_name == 'SPSA':
                from tf_neuralmpc.optimizers.spsa import SPSAOptimizer
                max_iterations = 5
                population_size = 500
                alpha = 0.602
                gamma = 0.101
                a_par = 0.01
                noise_parameter = 0.3
                my_optimizer = SPSAOptimizer(planning_horizon=planning_horizon,
                                               max_iterations=tf.constant(max_iterations, dtype=tf.int32),
                                               population_size=tf.constant(population_size,
                                                                           dtype=tf.int32),
                                               alpha=tf.constant(alpha, dtype=tf.float32),
                                               gamma=tf.constant(gamma, dtype=tf.float32),
                                               a_par=tf.constant(a_par, dtype=tf.float32),
                                               noise_parameter=tf.constant(noise_parameter,
                                                                           dtype=tf.float32),
                                               dim_U=input_size,
                                               dim_O=state_size,
                                               action_upper_bound=action_upper_bound,
                                               action_lower_bound=action_lower_bound,
                                               num_agents=number_of_agents,
                                               trajectory_evaluator=deterministic_trajectory_evaluator)
            else:
                raise Exception("this optimizer name is not supported")
        from tf_neuralmpc.policies.mpc_policy import MPCPolicy
        mpc_policy = MPCPolicy(system_dynamics_handler=system_dynamics_handler,
                               optimizer=my_optimizer,
                               tf_writer=self.tf_writer)

        return mpc_policy

    def learn_dynamics_from_randomness(self, number_of_rollouts, task_horizon, dynamics_function,
                                       epochs=30, learning_rate=1e-3, validation_split=0.2,
                                       batch_size=128, normalization=True,
                                       nn_optimizer=tf.keras.optimizers.Adam):
        """
        This is the learn dynamics function for the runner class which samples n rollouts using a random policy and then
        uses these rollouts to learn a dynamics function for the system.


        Parameters
        ---------
        number_of_rollouts: Int
            Number of rollouts/ episodes to perform for each of the agents in the vectorized environment.
        task_horizon: Int
            The task horizon/ episode length.
        dynamics_function: DeterministicDynamicsFunctionBaseClass
            Defines the system dynamics function.
        learning_rate: float
            Learning rate to be used in training the dynamics function.
        epochs: Int
            Number of epochs to be used in training the dynamics function everytime train is called.
        validation_split: float32
            Defines the validation split to be used of the rollouts collected.
        batch_size: int
            Defines the batch size to be used for training the model.
        nn_optimizer: tf.keras.optimizers
            Defines the optimizer to use with the neural network.
        normalization: bool
            Defines if the dynamics function should be trained with normalization or not.

        Returns
        -------
        system_dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler holds the trained system dynamics.
        """
        from tf_neuralmpc.policies.random_policy import RandomPolicy
        action_upper_bound = tf.constant(self.recording_env.action_space.high, dtype=tf.float32)
        action_lower_bound = tf.constant(self.recording_env.action_space.low, dtype=tf.float32)
        policy = RandomPolicy(number_of_agents=self.num_of_agents,
                              action_lower_bound=action_lower_bound,
                              action_upper_bound=action_upper_bound)

        from tf_neuralmpc.dynamics_handlers.system_dynamics_handler import SystemDynamicsHandler
        number_of_agents = tf.constant(self.num_of_agents, dtype=tf.int32)
        state_size = self.recording_env.observation_space.shape[0]
        input_size = self.recording_env.action_space.shape[0]
        system_dynamics_handler = SystemDynamicsHandler(dynamics_function=dynamics_function,
                                                        dim_O=state_size,
                                                        dim_U=input_size,
                                                        num_of_agents=number_of_agents,
                                                        log_dir=self.log_path,
                                                        tf_writer=self.tf_writer,
                                                        true_model=False,
                                                        normalization=normalization)
        traj_obs, traj_acs, traj_rews = \
            self.perform_rollouts(number_of_rollouts, task_horizon, policy)
        system_dynamics_handler.train(traj_obs, traj_acs, traj_rews, validation_split=validation_split,
                                      batch_size=batch_size, learning_rate=learning_rate,
                                      epochs=epochs, nn_optimizer=nn_optimizer)
        return system_dynamics_handler
