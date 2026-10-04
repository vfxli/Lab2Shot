"""Meta Sapiens2: human-centric dense prediction (body-part segmentation, surface
normals, human alpha matte) from single frames, the whole frame at 1024x768."""

from __future__ import annotations

from pathlib import Path

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file

SAPIENS2_URL = "https://github.com/facebookresearch/sapiens2.git"
SAPIENS2_COMMIT = "7e5bae88456ac418ff0e58e74106c9fe192055d4"  # after the 1B matting release

# (task, size) -> (pinned Hugging Face revision of facebook/sapiens2-<task>-<size>,
# sha256). Each repo holds the same fp32 checkpoint twice
# (model.safetensors for the transformers port, sapiens2_<size>_<task>.safetensors
# for the original code): only the latter is downloaded, to weights/<task>/, the
# layout of upstream's $SAPIENS_CHECKPOINT_ROOT. Not gated.
CHECKPOINTS = {
    ("seg", "0.4b"): ("449b3c5335e6722bb94990abdd1aa6e612432f22",
                      "b85fdb50b7d6123a967d5ee4a505e222baff8d2f7ad6bbf353578c1a61dfbac9"),
    ("normal", "0.4b"): ("52886591372f049acd4d59ae6d0a792ae1a61ac5",
                         "d2f20e8df7186772b7710546416db0752be224c68003c7f18ccbbf8b980ef1fd"),
    ("seg", "1b"): ("60cf0b3584d68dccca9adb3b59677b72a5611378",
                    "4b73c44963b377e93fcb4c4053f72a189836a22d05e12c30383046b9cd3c5bd4"),
    ("normal", "1b"): ("3480505c6690732dfdc91e7078213dba14844ab3",
                       "74283c838275d30022149e14b1512797c35c95b971f74a43fc010f1dd15ea012"),
    # the only matting size released
    ("matting", "1b"): ("88b79074a58a0e2bb5befc00a237488c43cd94da",
                        "2a0ea4ea2234e8dbd51e952950d8cedfe3ee2eca9099003784fd2d9cec6bc6e1"),
}


def checkpoint_path(weights: Path, task: str, size: str) -> Path:
    return weights / task / f"sapiens2_{size}_{task}.safetensors"


# 0.8b is released too (between the two); 5b is 20 GB per task in fp32, too big
# and slow for a 24 GB card next to other jobs.
SIZES = ("0.4b", "1b")


class Sapiens2(Extension):
    name = "sapiens2"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Meta Sapiens2"
    homepage = "https://github.com/facebookresearch/sapiens2"
    source = GitSource(url=SAPIENS2_URL, commit=SAPIENS2_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md",
    )
    generative = False
    env = EnvSpec(
        python="3.12",  # upstream requires >= 3.12 and torch >= 2.7
        torch=("torch==2.13.0", "torchvision==0.28.0"),
        torch_backend="cu130",
    )
    # each key's note: extension.sapiens2.weight.<task>-<size>.note (i18n/<lang>.toml; one per CHECKPOINTS entry)
    weights = tuple(
        hf_file(
            f"facebook/sapiens2-{task}-{size}", rev, f"sapiens2_{size}_{task}.safetensors",
            key=f"{task}-{size}",
            dest=f"{task}/sapiens2_{size}_{task}.safetensors",
            sha256=sha256,
        )
        for (task, size), (rev, sha256) in CHECKPOINTS.items()
    )

    def worker_env(self) -> dict[str, str]:
        return {
            # Upstream's configs resolve checkpoints under this root; the worker
            # loads exactly weights/<task>/sapiens2_<size>_<task>.safetensors.
            "SAPIENS_CHECKPOINT_ROOT": str(self.paths.weights),
        }


EXTENSION = Sapiens2()
