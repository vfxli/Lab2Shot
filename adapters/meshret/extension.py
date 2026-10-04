"""Pinned official MeshRet extension; inference is isolated from the core."""
from lab2shot.sdk import Extension, EnvSpec, GitSource, LicenseInfo, Weight, COMMERCIAL, ManualItem, ExtensionWeights, manual_weight

CHECKPOINT = ManualItem(key="meshret-model", page="https://github.com/abcyzj/MeshRet", filename="meshret_model.tar.bz2",
    markers=("epoch=36-step=182743.ckpt",), looks_like=("*meshret_model*",),
    install=ExtensionWeights("meshret", ("epoch=36-step=182743.ckpt",)), alone=("epoch=36-step=182743.ckpt",))

class Meshret(Extension):
    name = "meshret"
    sdk = 2
    title = "MeshRet"
    homepage = "https://github.com/abcyzj/MeshRet"
    source = GitSource("https://github.com/abcyzj/MeshRet.git", "96ccd8c8b5ee102a1b067780c0300c30da7d925d")
    license = LicenseInfo(tag=COMMERCIAL, url="https://github.com/abcyzj/MeshRet/blob/main/LICENSE")
    generative = False
    submodules = ()
    import_repo = None  # worker_env lists every official package root together
    extra_sources = {"pytorch3d": GitSource("https://github.com/facebookresearch/pytorch3d.git", "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba")}
    env = EnvSpec(python="3.11", torch=("torch==2.8.0", "torchvision==0.23.0"), torch_backend="cu128",
                  pickled_checkpoints=True, build="build.py", compiled=("pysdf==0.1.9",), compiled_cuda=False,
                  imports=("model.retnet", "utils.body_armatures"))
    manual_items = (CHECKPOINT,)
    weights = (manual_weight(CHECKPOINT),)

    def worker_env(self):
        import os
        paths = [str(self.paths.repo / "")]
        paths.append(str(self.paths.repo / "submodules"))
        return {"PYTHONPATH": os.pathsep.join(paths)}

EXTENSION = Meshret()
