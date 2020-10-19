"""
- instantiate an env for a pendulum
- instantiate an MPC controller using the true known analytical model
- record a rollout in a video.
"""
from blackbox_mpc.policies.mpc_policy import \
    MPCPolicy
from blackbox_mpc.utils.pendulum import PendulumTrueModel, \
    pendulum_reward_function
from blackbox_mpc.utils.recording import record_rollout
import gym

env = gym.make("Pendulum-v0")
mpc_policy = MPCPolicy(reward_function=pendulum_reward_function,
                       env_action_space=env.action_space,
                       env_observation_space=env.observation_space,
                       true_model=True,
                       dynamics_function=PendulumTrueModel(),
                       optimizer_name='CMA-ES',
                       num_agents=1)
record_rollout(env, horizon=200, policy=mpc_policy,
               record_file_path='./cma')
