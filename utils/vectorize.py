from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from utils.normalize import RunningMeanStd

import numpy as np
import pickle
import sys
import os

class Callback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, args, verbose=0):
        super(Callback, self).__init__(verbose)
        '''
        Those variables will be accessible in the callback
        (they are defined in the base class)

        ==== The RL model ====
        self.model = None  # type: BaseAlgorithm

        ==== An alias for self.model.get_env(), the environment used for training ====
        self.training_env = None  # type: Union[gym.Env, VecEnv, None]

        ==== Number of time the callback was called ====
        self.n_calls = 0  # type: int
        self.num_timesteps = 0  # type: int

        ==== local and global variables ====
        self.locals = None  # type: Dict[str, Any]
        self.globals = None  # type: Dict[str, Any]

        ==== The logger object, used to report things in the terminal ====
        self.logger = None  # stable_baselines3.common.logger

        ==== parent class ====
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]
        '''
        self.name = args.name
        self.save_freq = args.save_freq
        self.save_path = args.save_path
        self.save_dir = args.save_dir


    def _print(self, contents) -> None:
        print(f"[{self.name}] {contents}")

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # self.rewards
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        if self.num_timesteps % self.save_freq == 0:
            self.model.save(self.save_path)
            self._print('save success!')
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        self.model.save(self.save_path)
        self._print('save success!')
        pass

class DobroSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, args, start_method=None):
        super().__init__(env_fns, start_method)
        self.obs_rms = RunningMeanStd(args.save_dir, self.observation_space.shape[0])

    def reset(self):
        observations = super().reset()
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        return norm_observations

    def step(self, actions):
        observations, rewards, dones, infos = super().step(actions)
        self.obs_rms.update(observations)
        norm_observations = self.obs_rms.normalize(observations)
        for info in infos:
            if 'terminal_observation' in info.keys():
                info['terminal_observation'] = self.obs_rms.normalize(info['terminal_observation'])
        if self.obs_rms.count % int(1e5) == 0:
            self.obs_rms.save()
        return norm_observations, rewards, dones, infos

class SingleEnvWrapper:
    def __init__(self, env) -> None:
        self._env = env
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        state = self._env.reset()
        return np.expand_dims(state, axis=0)
    
    def step(self, actions):
        state, reward, done, info = self._env.step(actions[0])
        if done:
            info['terminal_observation'] = state[:]
            state = self._env.reset()
        states = np.expand_dims(state, axis=0)
        return states, [reward], [done], [info]
