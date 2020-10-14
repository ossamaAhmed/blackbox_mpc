"""
tf_neuralmpc/environment_utils/environment_wrapper.py
=====================================================
"""
import gym
from tf_neuralmpc.environment_utils.subprocess_env import SubprocVecEnv
import multiprocessing as mp


class EnvironmentWrapper:
    @staticmethod
    def make_standard_gym_env(env_name, random_seed=0, num_of_agents=1):
        """
        This is the make env function for standard gym envs which is responsible for creating the parallel environment,
        and a single enviornment as well. This takes care of traditional gym and mujoco envs.

        Parameters
        ---------
        env_name: String
            This specifies the standard env gym name.
        random_seed: Int
            This specifies the seed to use.
        num_of_agents: Int
            This specifies the number of agents to use.

        Returns
        -------
        recording_env: mujoco_env.MujocoEnv
            A single environment for one agent only.
        env: SubprocVecEnv
            A parellel environment for n agents running in parellel.
        """
        def make_envs(env_name_sub, rank):
            def _make_envs():
                env = gym.make(env_name_sub)
                # env = gym.wrappers.TimeLimit(env, max_time_steps)
                env.seed(random_seed + rank)
                return env

            return _make_envs

        queue = mp.Queue()
        env = SubprocVecEnv([make_envs(env_name, i) for i in range(num_of_agents)],
                             queue=queue)

        recording_env = gym.make(env_name)
        recording_env.seed(random_seed)
        return recording_env, env

    @staticmethod
    def make_custom_gym_env(env_class, random_seed=0, num_of_agents=1):
        """
           This is the make env function for custom gym envs which is responsible for creating the parallel environment,
           and a single enviornment as well. This takes care of custom gym and mujoco envs.

           Parameters
           ---------
           env_class: mujoco_env.MujocoEnv
               This specifies the class to be used in instantiating the envs.
           random_seed: Int
               This specifies the seed to use.
           num_of_agents: Int
               This specifies the number of agents to use.

           Returns
           -------
           recording_env: mujoco_env.MujocoEnv
                A single environment for one agent only.
           env: SubprocVecEnv
                A parellel environment for n agents running in parellel.
           """

        def make_envs(rank):
            def _make_envs():
                env = env_class()
                # env = gym.wrappers.TimeLimit(env, max_time_steps)
                env.seed(random_seed + rank)
                return env

            return _make_envs

        queue = mp.Queue()
        env = SubprocVecEnv([make_envs(i) for i in range(num_of_agents)],
                             queue=queue)

        recording_env = env_class()
        recording_env.seed(random_seed)
        return recording_env, env



