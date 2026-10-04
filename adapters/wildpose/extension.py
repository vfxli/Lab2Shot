"""WildPose camera solve with the released MASt3R and recurrent checkpoints."""
from lab2shot.sdk import CUDA_13_2_TOOLKIT, NONCOMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo, hf_file


class WildPose(Extension):
    name = "wildpose"
    sdk = 2
    title = "WildPose"
    homepage = "https://github.com/GradientSpaces/WildPose"
    source = GitSource(homepage + ".git", "7dd74e72b9ed8e572176e6ce86212fda23d259c2")
    submodules = ("thirdparty/lietorch",)
    extra_sources = {"moge": GitSource("https://github.com/microsoft/MoGe.git", "74fbce054ebed49800de42d0ad0e83495065719a")}
    import_repo = None  # worker_env lists repo + thirdparty/mast3r + moge; a non-None value would overwrite that PYTHONPATH
    license = LicenseInfo(url=homepage + "/blob/main/LICENSE", tag=NONCOMMERCIAL)
    generative = False
    env = EnvSpec(python="3.12", torch=("torch==2.9.0", "torchvision==0.24.0"), torch_backend="cu130",
                  cuda_toolkit=CUDA_13_2_TOOLKIT, compiled=("torch-scatter==2.1.2",),
                  build="build.py", pickled_checkpoints=True, imports=("lietorch", "droid_backends"))
    weights = (
        hf_file("gradient-spaces/WildPose", "95a538aa4ddbdbf79211fffc14dbccf8e7e2376f", "wildpose_v0.pth",
                key="wildpose", dest="wildpose_v0.pth", sha256="6c789c1f8506963ddb55b1490491fb22e797d7e0c615e0ede427bacffed8be5d"),
        hf_file("gradient-spaces/WildPose", "95a538aa4ddbdbf79211fffc14dbccf8e7e2376f", "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                key="mast3r", dest="mast3r.pth", sha256="e28f91b488554653e2b46ddae9c78c1143e0bcb2e27d3e26cdb0b717f1568eb2"),
        hf_file("Ruicheng/moge-2-vitl", "39c4d5e957afe587e04eec59dc2bcc3be5ecd968", "model.pt",
                key="moge2", dest="moge2.pt", sha256="3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5"),
    )

    def worker_env(self):
        import os
        return {"PYTHONPATH": os.pathsep.join([str(self.paths.repo), str(self.paths.repo / "thirdparty/mast3r"),
                                                str(self.paths.root / "moge")])}


EXTENSION = WildPose()
