import os
from typing import List
import imageio.v2 as imageio
import numpy as np


def save_mp4(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(frames) == 0:
        print("No frames to save!")
        return

    frames = [f.astype(np.uint8) for f in frames]

    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        format="FFMPEG"
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def save_gif(frames: List[np.ndarray], path: str, fps: int = 30) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, frames, fps=fps)
