from gym.wrappers.monitoring.video_recorder import VideoRecorder
from blackbox_mpc.policies.model_free_base_policy import ModelFreeBasePolicy


def record_rollout(env, horizon, policy, record_file_path):
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
    recorder = VideoRecorder(env,
                             record_file_path + '.mp4')
    observations = env.reset()
    for t in range(horizon):
        recorder.capture_frame()
        if not isinstance(policy, ModelFreeBasePolicy):
            action_to_execute, expected_obs, expected_reward = policy.act(
                observations, t)
        else:
            action_to_execute = policy.act(observations, t)
        observations, reward, done, info = env.step(action_to_execute)
    recorder.capture_frame()
    recorder.close()
    return
