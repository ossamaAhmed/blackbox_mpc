from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.dynamics_learning import learn_dynamics_from_policy
import logging
logging.getLogger().setLevel(logging.INFO)
import tensorflow as tf
from blackbox_mpc.dynamics_handlers.system_dynamics_handler import \
    SystemDynamicsHandler


def learn_dynamics_iteratively_w_mpc(env,
                                     number_of_initial_rollouts,
                                     number_of_rollouts_for_refinement,
                                     number_of_refinement_steps,
                                     task_horizon,
                                     env_action_space=None,
                                     env_observation_space=None,
                                     initial_policy=None,
                                     refinement_policy=None,
                                     planning_horizon=None,
                                     reward_function=None,
                                     is_normalized=True,
                                     optimizer_name='CEM',
                                     optimizer=None,
                                     num_agents=None,
                                     nn_optimizer=tf.keras.optimizers.Adam,
                                     dynamics_function=None,
                                     system_dynamics_handler=None,
                                     log_dir=None,
                                     tf_writer=None,
                                     save_model_frequency=1,
                                     saved_model_dir=None,
                                     exploration_noise=False,
                                     epochs=30, learning_rate=1e-3,
                                     validation_split=0.2, batch_size=128,
                                     start_episode=0,
                                     **optimizer_args):
    """
    This is the learn dynamics function iteratively using mpc policy
    for the runner class which samples n rollouts using an initial policy and then
    uses these rollouts to learn a dynamics function for the system which is then used to _sample further rollouts
    to refine the dynamics function.


    Parameters
    ---------
    env: parallelgymEnv
        a wrapped gym environment using blackbox.environment_utils.EnvironmentWrapper funcs
    env_action_space: gym.ActionSpace
            Defines the action space of the gym environment.
    env_observation_space: gym.ObservationSpace
        Defines the observation space of the gym environment.
    num_agents: tf.int32
            Defines the number of runner running in parallel
    dynamics_function: DeterministicDynamicsFunctionBaseClass
        Defines the system dynamics function.
    system_dynamics_handler: SystemDynamicsHandler
            The system_dynamics_handler is a handler of the state, actions and
            targets processing funcs as well.
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
    refinement_policy: ModelBasedBasePolicy
        The policy to be used in collecting the followup episodes to refine the policy.
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
    is_normalized: bool
        Defines if the dynamics function should be trained with normalization or not.
    reward_function: tf_function
            Defines the reward function with the prototype: tf_func_name(current_state, current_actions, next_state),
            where current_state is BatchXdim_S, next_state is BatchXdim_S and  current_actions is BatchXdim_U.
    planning_horizon: tf.int32
        Defines the planning horizon for the optimizer (how many steps to lookahead and optimize for).
    optimizer: OptimizerBaseClass
        Optimizer to be used that optimizes for the best action sequence and returns the first action.
    optimizer_name: str
        optimizer name between in ['CEM', 'CMA-ES', 'PI2', 'RandomSearch', 'PSO', 'SPSA'].
    saved_model_dir: string
            Defines the saved model directory where the model is saved in, in case of loading the model.
    save_model_frequency: Int
        Defines how often the model should be saved (defined relative to the number of refining iters)
    start_episode: Int
        the episode index for tensorflow logging purposes
    exploration_noise: bool
            Defines if exploration noise should be added to the action to be executed.
    log_dir: string
        Defines the log directory to save the normalization statistics in.
    tf_writer: tf.summary
            Tensorflow writer to be used in logging the data.

    Returns
    -------
    system_dynamics_handler: SystemDynamicsHandler
        The system_dynamics_handler holds the trained system dynamics.
    mpc_policy: ModelBasedBasePolicy
        The policy that was refined to be used as a control policy
    """
    if number_of_initial_rollouts > 0:
        system_dynamics_handler = learn_dynamics_from_policy(
            env=env,
            policy=initial_policy,
            number_of_rollouts=number_of_initial_rollouts,
            task_horizon=task_horizon,
            dynamics_function=dynamics_function,
            system_dynamics_handler=system_dynamics_handler,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            batch_size=batch_size,
            is_normalized=is_normalized,
            nn_optimizer=nn_optimizer,
            tf_writer=tf_writer,
            exploration_noise=exploration_noise,
            log_dir=log_dir,
            save_model_frequency=save_model_frequency,
            saved_model_dir=saved_model_dir)
        logging.info("Trained initial system model")
    else:
        if system_dynamics_handler is None:
            system_dynamics_handler = SystemDynamicsHandler(
                env_action_space=env_action_space,
                env_observation_space=env_observation_space,
                true_model=False,
                dynamics_function=dynamics_function,
                tf_writer=tf_writer,
                is_normalized=is_normalized,
                log_dir=log_dir,
                save_model_frequency=save_model_frequency,
                saved_model_dir=saved_model_dir)
    if refinement_policy is None:
        refinement_policy = MPCPolicy(reward_function=reward_function,
                               env_action_space=env_action_space,
                               env_observation_space=env_observation_space,
                               dynamics_handler=system_dynamics_handler,
                               optimizer=optimizer,
                               optimizer_name=optimizer_name,
                               num_agents=num_agents,
                               planning_horizon=planning_horizon,
                               tf_writer=tf_writer,
                               **optimizer_args)
    for i in range(number_of_refinement_steps):
        system_dynamics_handler = learn_dynamics_from_policy(
            env=env,
            policy=refinement_policy,
            number_of_rollouts=number_of_rollouts_for_refinement,
            task_horizon=task_horizon,
            system_dynamics_handler=system_dynamics_handler,
            epochs=epochs,
            learning_rate=learning_rate,
            validation_split=validation_split,
            batch_size=batch_size,
            is_normalized=is_normalized,
            nn_optimizer=nn_optimizer,
            tf_writer=tf_writer,
            exploration_noise=exploration_noise,
            start_episode=start_episode + (number_of_rollouts_for_refinement*i))
    return system_dynamics_handler, refinement_policy
