"""
- instantiate an env for a pendulum
- instantiate an MPC controller using the true known analytical model
- define cost/reward functions as used in the openAI gym env.
- render the resulting MPC afterwards
- switch optimizer (try CEM, CMA-ES, PSO, RandomSearch, SPSA or PI2)
- render the resulting MPC afterwards and compare
"""
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.pendulum import PendulumTrueModel, \
    pendulum_reward_function
import gym

env = gym.make("Pendulum-v0")
mpc_policy = MPCPolicy(reward_function=pendulum_reward_function,
                       env_action_space=env.action_space,
                       env_observation_space=env.observation_space,
                       true_model=True,
                       dynamics_function=PendulumTrueModel(),
                       optimizer_name='CEM',
                       num_agents=1)

current_obs = env.reset()
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_policy.act(
        current_obs, t)
    current_obs, reward, _, info = env.step(action_to_execute)
    env.render()

mpc_policy.switch_optimizer(optimizer_name='RandomSearch')

current_obs = env.reset()
for t in range(200):
    action_to_execute, expected_obs, expected_reward = mpc_policy.act(
        current_obs, t)
    current_obs, reward, _, info = env.step(action_to_execute)
    env.render()
