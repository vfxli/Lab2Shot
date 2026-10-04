"""Pinned official Kinematic Refinement extension; inference is isolated from the core."""
from lab2shot.sdk import Extension, EnvSpec, GitSource, LicenseInfo, Weight, NONCOMMERCIAL

class KinematicRefinement(Extension):
    name = "kinematic_refinement"
    sdk = 2
    title = "Kinematic Refinement"
    homepage = "https://github.com/seokhyeonhong/kinematic-refinement"
    source = GitSource("https://github.com/seokhyeonhong/kinematic-refinement.git", "cdadbfd188beb4030726b670cd21a2bdb82bff61")
    license = LicenseInfo(tag=NONCOMMERCIAL, url="https://github.com/seokhyeonhong/kinematic-refinement/blob/main/LICENSE")
    generative = False
    submodules = ('src/fairmotion',)
    import_repo = None  # worker_env lists every official package root together
    env = EnvSpec(python="3.11", torch=("torch==2.8.0",), torch_backend="cu128",
                  pickled_checkpoints=True, imports=("kinref.kin_test", "kinref.geo_test"))
    weights = ()

    def worker_env(self):
        import os
        paths = [str(self.paths.repo / "src")]
        paths.append(str(self.paths.repo / "src/fairmotion"))
        return {"PYTHONPATH": os.pathsep.join(paths)}

EXTENSION = KinematicRefinement()
