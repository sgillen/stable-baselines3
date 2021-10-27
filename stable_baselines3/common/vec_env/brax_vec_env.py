import brax
# ^^^ import first so user gets the correct error if sb3 is not found=
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type, Union
import numpy as np
import gym

act_dtype = np.ndarray

class BraxVecEnvWrapper(VecEnv):
    def __init__(self, brax_vec_gym_env: brax.envs.wrappers.VectorGymWrapper):
        self.brax_vec_gym_env = brax_vec_gym_env
        self.actions = None

        super().__init__(self.brax_vec_gym_env.num_envs,
                         self.brax_vec_gym_env.single_observation_space,
                         self.brax_vec_gym_env.single_action_space)

    def step_async(self, actions: act_dtype) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        obs, rews, dones, infos = self.brax_vec_gym_env.step(self.actions)
        return np.array(obs, copy=False), np.array(rews, copy=False), np.array(dones, copy=False), infos

    def step(self, actions: act_dtype):
        self.step_async(actions)
        return self.step_wait()

    def reset(self) -> VecEnvObs:
        return np.array(self.brax_vec_gym_env.reset(), copy=False)

    def render(self, mode: str = 'rgb_array'):
        return self.brax_vec_gym_env(mode)

    def close(self):
        raise NotImplementedError()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        raise NotImplementedError()

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        return NotImplementedError()

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        return NotImplementedError()
