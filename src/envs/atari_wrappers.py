from collections import deque
from dataclasses import dataclass
from typing import Deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NoopResetEnv(gym.Wrapper):
    """
    Apply a random number of NOOP actions on environment reset.
    """
    def __init__(self, env: gym.Env, noop_max: int = 30, noop_action: int = 0):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = noop_action

        assert isinstance(env.action_space, spaces.Discrete)
        assert 0 <= noop_action < env.action_space.n

    def reset(self, **kwargs):
        """
        Reset the environment and perform random NOOP steps.
        """
        obs, info = self.env.reset(**kwargs)
        n_noops = self.env.unwrapped.np_random.integers(1, self.noop_max + 1)

        for _ in range(n_noops):
            obs, reward, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Repeat the same action for a fixed number of frames (frame skipping)
    and return the max over the last two frames.
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer: Deque[np.ndarray] = deque(maxlen=2)

    def step(self, action):
        """
        Execute the given action for `skip` frames.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += float(reward)
            terminated = terminated or term
            truncated = truncated or trunc
            if terminated or truncated:
                break

        if len(self._obs_buffer) == 2:
            max_frame = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        else:
            max_frame = self._obs_buffer[0]
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and clear the internal frame buffer.
        """
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat loss of life as end of episode, but only reset the environment
    when the game is truly over.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        """
        Step the environment and convert life loss into episode termination.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        self.was_real_done = done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            terminated = True
        self.lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, reward, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Press the FIRE action on reset.
    """
    def __init__(self, env: gym.Env, fire_action: int = 1):
        super().__init__(env)
        self.fire_action = fire_action
        assert isinstance(env.action_space, spaces.Discrete)
        assert 0 <= fire_action < env.action_space.n

    def reset(self, **kwargs):
        """
        Reset the environment and perform a FIRE action once.
        """
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to their sign: -1, 0, or +1.
    """
    def reward(self, reward):
        """
        Map raw reward value to {-1, 0, +1}.
        """
        return float(np.sign(reward))


class WarpFrame(gym.ObservationWrapper):
    """
    Convert RGB frames to grayscale and resize to a fixed resolution (84x84).
    """
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84, grayscale: bool = True):
        super().__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to a single observation.
        """
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return obs.astype(np.uint8)


class FrameStack(gym.Wrapper):
    """
    Stack the last k observations along a new first axis.
    """
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames: Deque[np.ndarray] = deque(maxlen=k)

        assert isinstance(env.observation_space, spaces.Box)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, h, w), dtype=np.uint8
        )

    def reset(self, **kwargs):
        """
        Reset the environment and fill the frame buffer with the initial frame.
        """
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        """
        Step the environment and append the new observation to the frame buffer.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Build a stacked observation array from the internal frame buffer.
        """
        assert len(self.frames) == self.k
        return np.stack(list(self.frames), axis=0)


def is_fire_game(env_id: str) -> bool:
    """
    Check whether the environment typically requires a FIRE action on reset.
    """
    env_id_lower = env_id.lower()
    return "breakout" in env_id_lower


@dataclass
class AtariWrapperConfig:
    """
    Configuration for standard Atari preprocessing wrappers.

    Attributes:
        noop_max: Maximum number of NOOPs on reset.
        frame_skip: Number of frames to skip per action (MaxAndSkip).
        episodic_life: Whether to use episodic life (life loss ends episode).
        reward_clipping: Whether to clip rewards to {-1, 0, +1}.
        frame_stack: Number of stacked frames.
        fire_reset: Whether to apply FIRE on reset for applicable games.
        width: Output observation width.
        height: Output observation height.
    """
    noop_max: int = 30
    frame_skip: int = 4
    episodic_life: bool = True
    reward_clipping: bool = True
    frame_stack: int = 4
    fire_reset: bool = True
    width: int = 84
    height: int = 84


def apply_atari_wrappers(
    env: gym.Env,
    env_id: str,
    cfg: AtariWrapperConfig,
) -> gym.Env:
    """
    Apply a standard sequence of Atari preprocessing wrappers used in DQN.
    """
    env = NoopResetEnv(env, noop_max=cfg.noop_max)
    env = MaxAndSkipEnv(env, skip=cfg.frame_skip)

    if cfg.episodic_life:
        env = EpisodicLifeEnv(env)

    if cfg.fire_reset and is_fire_game(env_id):
        env = FireResetEnv(env, fire_action=1)

    env = WarpFrame(env, width=cfg.width, height=cfg.height, grayscale=True)

    if cfg.reward_clipping:
        env = ClipRewardEnv(env)

    env = FrameStack(env, k=cfg.frame_stack)
    return env
