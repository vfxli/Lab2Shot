"""VGGT checkpoints: plain data shared by extension.py (installer) and worker.py."""

from __future__ import annotations

# weights param -> (Hugging Face repo, pinned revision, file, size, LFS sha256).
# Same network (VGGT-1B, 1.26B params); only the training data / licence differ.
MODELS = {
    # Model card: license cc-by-nc-4.0 (non-commercial). Not gated.
    "original": ("facebook/VGGT-1B", "860abec7937da0a4c03c41d3c269c366e82abdf9", "model.safetensors",
                 5026367224, "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"),
    # Released 2025-07-29. Model card: license other / vggt-aup-license, "licensed for
    # commercial use, with the exception of military applications"; its LICENSE file is
    # the repo's LICENSE.txt word for word (VGGT License v1, 2025-07-29). Gated: request
    # form (name, date of birth, country, affiliation, job title) at
    # https://huggingface.co/facebook/VGGT-1B-Commercial
    "commercial": ("facebook/VGGT-1B-Commercial", "ebb29a532abe92960eeb6903a5530f16990ef4ab", "model.safetensors",
                   5026367224, "2b766b284359bc47ce26be107254621f685b758a0282082ff109f3ff02788b53"),
}
