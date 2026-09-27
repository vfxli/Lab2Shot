"""Scene arrays: how 3D data crosses between the core and a format module's worker (Alembic, FBX, the next file
format), in both directions, as one .npz of plain numpy arrays. The format family's contract: a reader writes what a
file holds in it, the core makes our packets of it (lab2shot/nodes/formats.py); a writer gets our scene in it.

Every item is one thing of one kind, as a DCC's outliner shows it:

    model       a mesh: static, moved by its transform, or deforming (its points change per sample)
    points      a point cloud
    curves      a set of 3D curves (hair, guide curves, a motion trail)
    camera      a camera
    character   a skeleton with its animation, the meshes skinned to it and their blend shapes

Top level
    fps            frames per second of the file
    unit_cm        centimetres per unit of every length below (1.0: already cm)
    axes           [3,3] turning the file's axes into Y up, right-handed (identity: already so); may be missing: the
                   node says (Alembic records no axes)
    options_<p>    [K] str, optional: the values the file allows a parameter <p> of the import node (an FBX's takes),
    auto_<p>       [str] and what <p> left empty reads (SceneArrays.option; the core lists them: nodes/formats.py)
    n_models, n_points, n_curves, n_cameras, n_characters

Every item <kind><i>_ (model0_, camera2_ ...)
    name           [str] its short name (what DCCs call it: the transform's name)
    path           [str] where it is in the file's hierarchy, as DCCs show it (/rig/cam_main)
    frames         [T] int, the frames its samples are at (T = 1: it never changes)
    world          [T,4,4] local-to-world, column vectors (translation in [:3, 3]), every parent composed; not for a
                   character, whose joints are in world space already

model          counts [P] corners per face, indices [N] corner points (faces counter-clockwise seen from the front),
               points [S,V,3] local, S = 1 (its shape never changes) or T (deforming: one per frame),
               optional uv [U,2] + uv_indices [N] (face-varying), normals [N,3] (face-varying, local);
               optional 分区 (a part of the mesh: USD's GeomSubset, Alembic's FaceSet, Houdini's primitive group,
               Maya's face set) as subset_names [K] str, subset_counts [K] how many faces each holds and
               subset_faces [sum] the face indices themselves, the parts one after another in the same order
points         counts [S] points per sample (S = 1 or T), points [sum,3] local, optional colors [sum,3] (0-1),
               widths [sum] (lengths); tracked points (3D tracks) also ids [sum] int (a point keeps its id from
               sample to sample: Houdini's id), velocities [sum,3] (lengths per second, local: Houdini's v) and
               visible [sum] int (1 seen in the picture at that frame, 0 not)
curves         counts [S] points per sample (S = 1 or T) and points [sum,3] local, exactly as a point cloud's, so
               every per-point array is cut the same way; on top of that curve_counts [S] curves per sample and
               curve_vertex_counts [sum_curves] points per curve (the samples' one after another, cut by the running
               sum of curve_counts). Optional widths [sum] (lengths, one per point: hair is thick at the root and
               thin at the tip), colors [sum,3] (0-1) and normals [sum,3] (local, the curve's own orientation)
camera         focal_mm [T], h_aperture_mm [T], v_aperture_mm [T], optional resolution [2] (px, when the file records
               one). Its world may carry a parent's scale: the core keeps only position and orientation
character      joints [J] names, parents [J] (-1: a root; parents come first), bind [J,4,4] joint-to-world in the
               pose the meshes are bound in, anim [T,J,4,4] joint-to-world per frame, n_meshes; per mesh
               <kind><i>_mesh<k>_: name, counts, indices, points [V,3] (bind pose, world), optional uv, uv_indices,
               joint_indices [V,K] + joint_weights [V,K] (skin), shapes [B] names, shape_offsets [B,V,3] (world,
               added to the bind points before skinning), shape_weights [T,B] (0-1)

Units and axes are the file's in what a reader writes (the core converts: centimetres, Y up); what the core gives a
writer is in the unit its output settings chose, Y up. Only numpy is needed on either side.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

KINDS = ("model", "points", "curves", "camera", "character")
MESH_KEY = re.compile(r"mesh\d+_")


class SceneArrays:
    """Builds the arrays item by item; save() writes them."""

    def __init__(self, fps: float, unit_cm: float = 1.0, axes=None):
        self.top: dict = {"fps": np.float64(fps), "unit_cm": np.float64(unit_cm)}
        if axes is not None:
            self.top["axes"] = np.asarray(axes, np.float64).reshape(3, 3)
        self.items: dict[str, list[dict]] = {k: [] for k in KINDS}

    def option(self, param: str, values, auto: str) -> None:
        """The values the file allows a parameter of the import node (an FBX's takes), and what it reads left empty."""
        self.top[f"options_{param}"] = np.array([str(v) for v in values])
        self.top[f"auto_{param}"] = np.array(str(auto))

    def add(self, kind: str, name: str, path: str, frames, world=None, **arrays) -> dict:
        """One item: its name, path, the frames of its samples and its local-to-world per sample (not for a
        character), then the arrays of its kind (see the module's docstring). Returns it (a character's meshes are
        added to it with add_mesh)."""
        item = {"name": np.array(str(name)), "path": np.array(str(path)), "frames": np.asarray(frames, np.int64).reshape(-1)}
        if world is not None:
            item["world"] = np.asarray(world, np.float64).reshape(-1, 4, 4)
        item.update({k: np.asarray(v) for k, v in arrays.items() if v is not None})
        if kind == "character":
            item["meshes"] = []
        self.items[kind].append(item)
        return item

    @staticmethod
    def add_mesh(character: dict, name: str, **arrays) -> None:
        character["meshes"].append({"name": np.array(str(name)), **{k: np.asarray(v) for k, v in arrays.items() if v is not None}})

    def arrays(self) -> dict[str, np.ndarray]:
        out = dict(self.top)
        for kind, items in self.items.items():
            out[f"n_{plural(kind)}"] = np.int64(len(items))
            for i, item in enumerate(items):
                for key, value in item.items():
                    if key != "meshes":
                        out[f"{kind}{i}_{key}"] = value
                if kind == "character":
                    out[f"{kind}{i}_n_meshes"] = np.int64(len(item["meshes"]))
                    for k, mesh in enumerate(item["meshes"]):
                        out.update({f"{kind}{i}_mesh{k}_{key}": v for key, v in mesh.items()})
        return out

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp.npz")
        np.savez(tmp, **self.arrays())
        tmp.replace(path)
        return path


def plural(kind: str) -> str:
    """How many of a kind the arrays count (n_models, n_points, n_curves ...): a kind whose name is already plural
    keeps it."""
    return kind if kind in ("points", "curves") else kind + "s"


def load(path: Path | dict) -> tuple[dict, dict[str, list[dict]]]:
    """(top-level values, kind -> its items as dicts of arrays; a character's meshes as its "meshes" list)."""
    data = dict(np.load(path, allow_pickle=False)) if not isinstance(path, dict) else path
    top = {k: data[k] for k in data if k in ("fps", "unit_cm", "axes", "frames") or k.startswith(("options_", "auto_"))}
    items: dict[str, list[dict]] = {}
    for kind in KINDS:
        items[kind] = []
        for i in range(int(data.get(f"n_{plural(kind)}", 0))):
            head = f"{kind}{i}_"
            item = {k[len(head):]: v for k, v in data.items() if k.startswith(head) and not MESH_KEY.match(k[len(head):])}
            if kind == "character":
                item["meshes"] = [{k[len(f"{head}mesh{m}_"):]: v for k, v in data.items() if k.startswith(f"{head}mesh{m}_")}
                                  for m in range(int(item.pop("n_meshes", 0)))]
            items[kind].append(item)
    return top, items


def text(value) -> str:
    """A string item value (a 0-d array of str or bytes)."""
    value = np.asarray(value)
    v = value.item() if value.ndim == 0 else value.reshape(-1)[0]
    return v.decode() if isinstance(v, bytes) else str(v)


def texts(value) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in np.asarray(value).reshape(-1).tolist()]
