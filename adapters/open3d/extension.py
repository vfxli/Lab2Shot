"""Open3D: classic surface reconstruction (Poisson / Ball Pivoting / Alpha Shape) from a point cloud.

Open3D ships as a pip wheel (the whole C++ library is inside it), so the adapter installs it into the isolated
environment and the worker imports `open3d` straight from pip — no build step, no repo import. The pinned
repository (isl-org/Open3D) is checked out only so the node's official citation can point at the real upstream
source. MIT.
"""

from __future__ import annotations

from lab2shot.sdk import COMMERCIAL, EnvSpec, Extension, GitSource, LicenseInfo

OPEN3D_URL = "https://github.com/isl-org/Open3D.git"
OPEN3D_COMMIT = "8fa74386dfe083d1a8eb8ef30e30ec955b72e66e"  # v0.19.0


class Open3D(Extension):
    name = "open3d"
    sdk = 2  # lab2shot.sdk.SDK_API this adapter is written for
    title = "Open3D"
    homepage = "https://www.open3d.org"
    source = GitSource(url=OPEN3D_URL, commit=OPEN3D_COMMIT)
    license = LicenseInfo(
        tag=COMMERCIAL,
        url="https://github.com/isl-org/Open3D/blob/main/LICENSE",
    )
    generative = False
    import_repo = None  # the worker imports open3d from pip, not from the repository
    env = EnvSpec(
        python="3.11",
        imports=("open3d",),
    )


EXTENSION = Open3D()
