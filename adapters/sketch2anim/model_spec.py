"""What the released Sketch2Anim model works at: plain data shared by nodes.py (lengths it plans in model frames) and
worker.py (the resampling to the shot's frames), so the two cannot drift apart."""

from __future__ import annotations

MODEL_FPS = 20.0  # HumanML3D, and what the released weights generate at
MAX_MODEL_FRAMES = 196  # cfg.DATASET.SAMPLER.MAX_LEN: the longest latent the VAE was trained to decode
MIN_MODEL_FRAMES = 40  # cfg.DATASET.SAMPLER.MIN_LEN: a shorter motion is outside what it was trained on


def model_frames(frames: int, fps: float) -> int:
    """How many model frames a shot of `frames` frames at `fps` is generated as (at least one)."""
    return max(int(round(frames / fps * MODEL_FPS)), 1)


def shot_frames(fps: float) -> tuple[int, int]:
    """The fewest and the most shot frames at `fps` that model_frames puts inside what the model was trained on
    (MIN_MODEL_FRAMES … MAX_MODEL_FRAMES): the numbers the node's messages give, found through model_frames itself
    (its rounding included), searched over every frame count that can round onto the bound — one model frame spans
    fps / MODEL_FPS shot frames, so the search reaches that far on both sides of the plain product."""
    reach = int(fps / MODEL_FPS) + 2
    guess = int(MAX_MODEL_FRAMES / MODEL_FPS * fps)
    most = next(n for n in range(guess + reach, 0, -1) if model_frames(n, fps) <= MAX_MODEL_FRAMES)
    guess = int(MIN_MODEL_FRAMES / MODEL_FPS * fps)
    least = next(n for n in range(max(guess - reach, 1), guess + reach + 1) if model_frames(n, fps) >= MIN_MODEL_FRAMES)
    return least, most
