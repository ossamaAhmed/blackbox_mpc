import logging
logging.getLogger().setLevel(logging.INFO)
from blackbox_mpc.policies.model_free_base_policy import ModelFreeBasePolicy
import time
import tensorflow as tf
import numpy as np
from blackbox_mpc.policies.random_policy import RandomPolicy


def perform_rollouts(env, number_of_rollouts, task_horizon, policy,
                     exploration_noise=False, tf_writer=None,
                     start_episode=0):
    """
    This is the perform_rollouts function for the runner class which samples n episodes with a specified length
    using the provided policy.


    Parameters
    ---------
    env: parallelgymEnv
        a wrapped gym environment using blackbox.environment_utils.EnvironmentWrapper funcs
    number_of_rollouts: Int
        Number of rollouts/ episodes to perform for each of the agents in the vectorized environment.
    task_horizon: Int
        The task horizon/ episode length.
    policy: ModelBasedBasePolicy or ModelFreeBasePolicy
        The policy to be used in collecting the episodes from the different agents.
    exploration_noise: bool
        If noise should be added to the actions to help in exploration.
    tf_writer: tf.summary
            Tensorflow writer to be used in logging the data.
    start_episode: Int
        the episode index for tensorflow logging purposes

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
            _sample(
                env, task_horizon, policy, exploration_noise=exploration_noise,
                tf_writer=tf_writer,
                episode_step=start_episode+i))
        traj_obs.append(samples[-1]["observations"])
        traj_acs.append(samples[-1]["actions"])
        traj_rews.append(samples[-1]["rewards"])
    logging.info("Finished collecting samples for rollout")
    return traj_obs, traj_acs, traj_rews


def _sample(env, horizon, policy, episode_step,
            exploration_noise=False, tf_writer=None):
    """
    This is the sampling function for the runner class which samples one episode with a specified length
    using the provided policy.


    Parameters
    ---------
    env: parallelgymEnv
        a wrapped gym environment using blackbox.environment_utils.EnvironmentWrapper funcs
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
    policy.reset()
    first_obs = env.reset()
    times, observations, actions, rewards, reward_sum, done = \
        [], [first_obs], [], [], 0, False
    if not isinstance(policy, ModelFreeBasePolicy):
        predicted_reward = 0

    for t in range(horizon):
        start = time.time()
        if not isinstance(policy, ModelFreeBasePolicy):
            action_to_execute, expected_obs, expected_reward = \
                policy.act(observations[t], t, exploration_noise)
            predicted_reward += expected_reward
        else:
            action_to_execute = policy.act(observations[t], t)
            action_to_execute = action_to_execute.numpy()
        actions.append(action_to_execute)
        times.append(time.time() - start)
        obs, reward, done, info = env.step(actions[t])
        if tf_writer is not None:
            if not isinstance(policy, RandomPolicy):
                with tf_writer.as_default():
                    tf.summary.scalar('rewards/actual_reward', np.mean(reward),
                                      step=(episode_step*horizon)+t)
            if not isinstance(policy, ModelFreeBasePolicy):
                with tf_writer.as_default():
                    tf.summary.scalar('states/predicted_observations_abs_error',
                                      np.mean(np.sum(np.abs(expected_obs - obs),
                                                     axis=1)),
                                      step=(episode_step*horizon)+t)
                    tf.summary.scalar('rewards/predicted_reward_abs_error',
                                      np.mean(np.abs(expected_reward - reward)),
                                      step=(episode_step * horizon) + t)
        observations.append(obs)
        rewards.append(reward)
        reward_sum += reward
        if t >= horizon - 1:
            if tf_writer is not None:
                if not isinstance(policy, RandomPolicy):
                    with tf_writer.as_default():
                        tf.summary.scalar('rewards/actual_episode_reward',
                                          np.mean(reward_sum),
                                          step=episode_step)
                if not isinstance(policy, ModelFreeBasePolicy):
                    with tf_writer.as_default():
                        tf.summary.scalar('rewards/predicted_episode_reward',
                                          np.mean(predicted_reward),
                                          step=episode_step)
            break
    logging.info("Average action selection time: " + str(np.mean(times)))
    logging.info("Rollout length: " + str(len(actions)))

    return {"observations": np.array(observations),
            "actions": np.array(actions),
            "rewards": np.array(rewards),
            "reward_sum": reward_sum}
