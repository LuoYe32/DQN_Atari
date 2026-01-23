from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym

from .atari_wrappers import AtariWrapperConfig, apply_atari_wrappers


@dataclass
class AtariEnvConfig:
    """
    Configuration for creating an Atari environment.

    Attributes:
        env_id: Gymnasium Atari environment id (e.g. "ALE/Pong-v5").
        frameskip: Native ALE frameskip (usually kept as 1 because skipping is
            implemented explicitly by a wrapper).
        repeat_action_probability: Sticky actions probability (0.0 matches classic DQN setup).
        full_action_space: Whether to enable full action set.
        render_mode: Rendering mode (None, "human", or "rgb_array").
        wrappers: Wrapper configuration for standard DQN preprocessing.
    """
    env_id: str = "ALE/Pong-v5"

    frameskip: int = 1
    repeat_action_probability: float = 0.0
    full_action_space: bool = False
    render_mode: Optional[str] = None

    wrappers: AtariWrapperConfig = field(default_factory=AtariWrapperConfig)


def make_atari_env(cfg: AtariEnvConfig) -> gym.Env:
    """
    Create a Gymnasium Atari environment and apply standard DQN wrappers.
    """
    env = gym.make(
        cfg.env_id,
        frameskip=cfg.frameskip,
        repeat_action_probability=cfg.repeat_action_probability,
        full_action_space=cfg.full_action_space,
        render_mode=cfg.render_mode,
    )
    env = apply_atari_wrappers(env, env_id=cfg.env_id, cfg=cfg.wrappers)
    return env
