"""AnyCalib: single-image lens calibration (focal, principal point, distortion)
for many camera models, combined over a few frames of a shot -> lens.json + ST-maps.
"""

from __future__ import annotations


from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

ANYCALIB_URL = "https://github.com/javrtg/AnyCalib.git"
ANYCALIB_COMMIT = "027a8497d893f4b2596f23d6324c05e4b81064ed"  # matches the v1.0.0 weights

# Same files as the GitHub release v1.0.0, from Hugging Face at a pinned revision
# (checked against the LFS sha256 after download).
HF_REVISION = "660ab9b7924ccca12d1e353585594e34f1251341"
CHECKPOINTS = {
    "anycalib_gen": "65d375de971456d0bcc2c8dcdeb711e4f2ac892366b8ae7dd4351868a9d4d940",
    "anycalib_pinhole": "e73b174563bcb90dc9a0e348ee64fbb898901d2a1786623e62fc0e3b176a1a38",
    "anycalib_dist": "89090e4b43e78b3ba16fefdb305f500047b870fb93da4d498ffa935ee27e8ee5",
}


from .lens import GROUP


class AnyCalib(Extension):
    name = "anycalib"
    lens_groups = (GROUP,)  # AnyCalib's own model names -> core formula table (lens.py)
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "AnyCalib"
    homepage = "https://github.com/javrtg/AnyCalib"
    source = GitSource(url=ANYCALIB_URL, commit=ANYCALIB_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/javrtg/AnyCalib/blob/main/LICENSE",
        )
    generative = False
    env = EnvSpec(
        python="3.12",
        # Upstream only needs "torch"; 2.8.0 cu128 runs on both the 4090 (sm_89) and the 5090 (sm_120).
        torch=("torch==2.8.0",),
        torch_backend="cu128",
    )
    weights = tuple(
        hf_file(
            "javrtg/AnyCalib", HF_REVISION, f"{name}.pt",
            key=name.replace("_", "-"),
            dest=f"{name}.pt",
            sha256=sha256,
        )
        for name, sha256 in CHECKPOINTS.items()
    )

    def worker_env(self) -> dict[str, str]:
        return {
            "XFORMERS_DISABLED": "1",  # plain PyTorch attention, identical on every machine
        }


EXTENSION = AnyCalib()
