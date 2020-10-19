"""
- instantiate an env for a pendulum
- instantiate an MPC by loading a previosuluy saved dynamics model.
- render the result.
"""
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.pendulum import pendulum_reward_function
import gym

env = gym.make("Pendulum-v0")
mpc_policy = MPCPolicy(reward_function=pendulum_reward_function,
                       env_action_space=env.action_space,
                       env_observation_space=env.observation_space,
                       optimizer_name='CEM',
                       saved_model_dir='./saved_model',
                       num_agents=1)

current_obs = env.reset()
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_policy.act(
        current_obs, t)
    current_obs, reward, _, info = env.step(action_to_execute)
    env.render()
