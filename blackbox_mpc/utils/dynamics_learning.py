import tensorflow as tf
from blackbox_mpc.utils.rollouts import perform_rollouts
from blackbox_mpc.dynamics_handlers.system_dynamics_handler import \
    SystemDynamicsHandler


def learn_dynamics_from_policy(env, policy,
                               number_of_rollouts, task_horizon,
                               dynamics_function=None,
                               system_dynamics_handler=None,
                               epochs=30, learning_rate=1e-3,
                               validation_split=0.2,
                               batch_size=128, is_normalized=True,
                               nn_optimizer=tf.keras.optimizers.Adam,
                               tf_writer=None,
                               exploration_noise=False,
                               log_dir=None,
                               save_model_frequency=1,
                               saved_model_dir=None):
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
    if system_dynamics_handler is None:
        system_dynamics_handler = SystemDynamicsHandler(env_action_space=env.action_space,
                                                        env_observation_space=env.observation_space,
                                                        true_model=False,
                                                        dynamics_function=dynamics_function,
                                                        tf_writer=tf_writer,
                                                        is_normalized=is_normalized,
                                                        log_dir=log_dir,
                                                        save_model_frequency=save_model_frequency,
                                                        saved_model_dir=saved_model_dir)
    traj_obs, traj_acs, traj_rews = \
        perform_rollouts(env, number_of_rollouts, task_horizon, policy,
                         exploration_noise=exploration_noise, tf_writer=None)
    system_dynamics_handler.train(traj_obs, traj_acs, traj_rews,
                                  validation_split=validation_split,
                                  batch_size=batch_size,
                                  learning_rate=learning_rate,
                                  epochs=epochs, nn_optimizer=nn_optimizer)
    return system_dynamics_handler
