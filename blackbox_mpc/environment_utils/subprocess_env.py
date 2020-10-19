import numpy as np
import multiprocessing as mp
import gym


class EnvWorker(mp.Process):
    """This is the Env Worker class which is responsible for running one agent in one process"""
    def __init__(self, remote, env_fn, queue, lock):
        """
        This is the initializer function for the Env Worker class.

        Parameters
        ---------
        remote: multiprocessing.Pipe()
            The remote process responsible for sending and receiving to/from the parent process.
        env_fn: py_func
            A python function that would return a new environment when called
        queue: multiprocessing.Queue()
            A multiprocessing queue to be used in the cosumption of shared resources.
        lock: multiprocessing.Lock()
             A lock to be used in locking shared resources between agents.
        """
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.queue = queue
        self.lock = lock
        self.done = False

    def empty_step(self):
        """
        This is the empty step function for the Env Worker class, a step without an action basically.

        Returns
        -------
        observation: np.float32
            The observations received with a zeros in the action as a step in the environment.
        reward: np.float32
            The rewards received with a zeros in the action as a step in the environment.
        done: bool
            A boolean if true then the worker is terminated, if not then the worker is not terminated.
        info: empty dict
            An empty dict to keep the consistency with the other step.

        """
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        """
        This is the try reset function for the Env Worker class, which resets the environment of the worker.

        Returns
        -------
        observation: np.float32
            The observations received after a reset in the environment.
        """
        observation = self.env.reset()
        return observation

    def run(self):
        """
        This is the run function of the worker which sends and receives data.

        - If command is step then

        Returns
        -------
        observation: np.float32
            The observations after executing the actions received.
        reward: np.float32
            The rewards after executing the actions received.
        done: bool
            A boolean if true then the worker is terminated, if not then the worker is not terminated.
        info: dict
            A dict that returns extra information of the step that was just executed in the environment.



        - If command is reset then

        Returns
        -------
        observation: np.float32
            The observations after a reset in the environment.


        - If command is get_spaces then

        Returns
        -------
        observation_space: ObservationSpaceObject
            The observation space object which has a low and a high as well as shape.
        action_space: ActionSpaceObject
            The action space object which has a low and a high as well as shape.



        - If command is render then

        Returns
        -------
        result: np.float32
            returns the RGB array of rendering the environment at this current timestep.

        """
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = self.env.step(data)
                self.remote.send((observation, reward, done, info))
            elif command == 'reset':
                observation = self.try_reset()#get new id for the worker here
                self.remote.send((observation))
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                 self.env.action_space))
            elif command == 'render':
                self.remote.send(self.env.render(mode='rgb_array'))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    """This is the SubprocVecEnv class which is responsible for running n agents in n processes"""
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, env_factory, queue):
        """
        This is the initializer function for the SubprocVecEnv class.

        Parameters
        ---------
        env_factory: [py_func]
            A list of py_funcs where each py_func returns a new environment for a new process.
        queue: multiprocessing.Queue()
            A multiprocessing queue to be used in the cosumption of shared resources.
        """
        self.lock = mp.Lock()
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])
        self.workers = [EnvWorker(remote, env_fn, queue, self.lock)
            for (remote, env_fn) in zip(self.work_remotes, env_factory)]
        for worker in self.workers:
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        """
        This is the step function for the SubprocVecEnv class.

        Parameters
        ---------
        actions: np.float32
            Actions to be executed to each of the environment_utils/agents.


        Returns
        -------
        observation: np.float32
            The observations after executing the actions received.
        reward: np.float32
            The rewards after executing the actions received.
        done: [bool]
            A list of boolean if true then the agent is terminated, if not then the agent is not terminated.
        info: dict
            A dict that returns extra information of the step that was just executed in the parallel environment_utils.
        """
        self._step_async(actions)
        return self._step_wait()

    def _step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def _step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        """
        This is the reset function for the SubprocVecEnv class.

        Returns
        -------
        observation: np.float32
            The observations after resetting each of the environment_utils.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        return np.stack(results)

    def close(self):
        """
        This is the close function for the SubprocVecEnv class.
        """
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True

    def render(self, mode='human'):
        """
        This is the render function for the SubprocVecEnv class.

        Returns
        -------
        image: np.float32
            The RGB array of the current view of the environment_utils.
        """
        for remote in self.remotes:              
            remote.send(('render', None))
        results = [remote.recv() for remote in self.remotes]
        return results
