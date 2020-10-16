import mujoco_py
import gym
import gym_anymal
import numpy as np

env = gym.make("Anymal-v0")

env.reset()
for i in range(20000):
    env.step(np.zeros(12,))
    env.render()
