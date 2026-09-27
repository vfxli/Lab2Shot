"""Pi3 checkpoints: plain data shared by extension.py (installer) and worker.py."""

from __future__ import annotations

# weights param -> (Hugging Face repo, pinned revision, file, size, LFS sha256).
# Both CC BY-NC 4.0 per the repo README's licence table (the Pi3 model card's
# metadata says bsd-2-clause, but its text says commercial use needs the authors'
# permission; the Pi3X card says CC BY-NC 4.0).
MODELS = {
    "pi3x": ("yyfz233/Pi3X", "bb1deea4d7423de5b30691739cb451a3f57dc1d5", "model.safetensors",
             5440325620, "69972d6e1c4492cb4d737a84fe940e357087d81c52f5c9b7c160b49c1f41669a"),
    "pi3": ("yyfz233/Pi3", "ae722e7039287d0c8fde9f11f197f804f44b510c", "model.safetensors",
            3834909248, "33580e4702ac671558aedeab1148fd08118f7ce45bdbeb99f3e3cf340062875d"),
}
