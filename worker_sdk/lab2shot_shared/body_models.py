"""Body, hand and face models that must be downloaded manually (registration required, research licences).

Defines where Lab2Shot unpacks each model and which file the methods load, shared by the core (installation status)
and the workers (loading the file). Standard library only.

Downloading, recognition and unpacking are handled by the core (lab2shot.extensions.manual): the user places the
archive in the Lab2Shot inbox (downloads/) and the core unpacks it into ROOT/<model>/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .protocol import PROJECT_DIR

ROOT = PROJECT_DIR / "third_party" / "_body_models"  # <model>/: downloaded archives unpacked by Lab2Shot


@dataclass(frozen=True)
class BodyModel:
    title: str
    files: tuple[str, ...]  # file loaded by the methods: the official download name first, then common aliases
    # Manual-download metadata. The core's manual-download entries (lab2shot/extensions/manual.py) are generated
    # from this table; other manual downloads are declared by the extensions that require them.
    what: str = ""  # what it is, in English (shown through the catalogue: manual.<key>.what)
    page: str = ""  # download page URL
    download: str = ""  # name of the download item as labelled on that page
    filename: str = ""  # expected file name (recognition is by content, so renamed files are accepted)
    markers: tuple[str, ...] = ()  # glob patterns of file names that identify it inside an archive or folder
    looks_like: tuple[str, ...] = ()  # glob patterns of file names that resemble it but are not it; used for a hint


MODELS = {
    "smplx": BodyModel("SMPL-X", ("SMPLX_NEUTRAL.npz",),  # models_smplx_v1_1.zip: models/smplx/SMPLX_NEUTRAL.npz
                       "Body, hands and face model", "https://smpl-x.is.tue.mpg.de", "SMPL-X v1.1 (NPZ+PKL)", "models_smplx_v1_1.zip",
                       ("SMPLX_NEUTRAL.*", "SMPLX_MALE.*", "SMPLX_FEMALE.*"), ("*smplx*", "*smpl-x*")),
    # SMPL for Python v1.1.0 names the neutral body basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl; the SMPLify archive
    # (fetched by the WHAM and TRAM download scripts) names it basicModel_neutral_lbs_10_207_0_v1.0.0.pkl; it is also
    # commonly renamed to SMPL_NEUTRAL.pkl.
    "smpl": BodyModel("SMPL", ("basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl", "SMPL_NEUTRAL.pkl",
                               "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"),
                      "Body model", "https://smpl.is.tue.mpg.de", "SMPL for Python users 1.1.0", "SMPL_python_v.1.1.0.zip",
                      ("basicmodel_*.pkl", "SMPL_NEUTRAL.pkl", "SMPL_MALE.pkl", "SMPL_FEMALE.pkl", "smpl_uv*"), ("*smpl*",)),
    "mano": BodyModel("MANO", ("MANO_RIGHT.pkl",),  # mano_v1_2.zip: mano_v1_2/models/MANO_RIGHT.pkl; left hands use the mirrored model
                      "Hand model", "https://mano.is.tue.mpg.de", "Models & Code", "mano_v1_2.zip",
                      ("MANO_RIGHT.*", "MANO_LEFT.*"), ("*mano*",)),
    # FLAME2020.zip; SMIRK coefficients are defined for FLAME 2020, not FLAME 2023.
    # The FLAME masks and the MediaPipe landmark embedding are separate archives and are unpacked into the same folder.
    "flame": BodyModel("FLAME", ("generic_model.pkl",),
                       "Face model", "https://flame.is.tue.mpg.de", "FLAME 2020", "FLAME2020.zip",
                       ("generic_model.pkl", "FLAME2020", "FLAME2023*", "FLAME_masks.*", "mediapipe_landmark_embedding.*"), ("*flame*",)),
}


def find(model: str) -> Path | None:
    """Return the model file: the first accepted name found anywhere below ROOT/<model>/ (case-insensitive), or None."""
    folder = ROOT / model
    if not folder.is_dir():
        return None
    files: dict[str, Path] = {}
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            files.setdefault(p.name.lower(), p)
    return next((files[n.lower()] for n in MODELS[model].files if n.lower() in files), None)


# FLAME region masks (`FLAME_masks.pkl`, defined by the FLAME authors and distributed with generic_model.pkl):
# the vertex indices belonging to the scalp, face, neck, lips, etc. FLAME is a parametric model with a fixed vertex
# topology, so these indices apply to every subject; once face solving fits a subject's head to FLAME, the indices
# map onto that subject's own scalp.
#
# Hair solving uses the official scalp region. NeuralHaircut additionally trims the template region per hairstyle
# (cut_scalp.py: keeps scalp vertices within 7 cm of the hair SDF), which requires running its own network. HairGS
# does not need this step: it uses scalp vertices as root candidates, and no strands grow where hair is absent.
MASKS = {"flame": "FLAME_masks.pkl"}


def masks_file(model: str) -> Path | None:
    """Return the model's region-mask file, or None if it is not downloaded or the model has no masks."""
    name = MASKS.get(model)
    if not name:
        return None
    folder = ROOT / model
    if not folder.is_dir():
        return None
    return next((p for p in sorted(folder.rglob("*")) if p.is_file() and p.name.lower() == name.lower()), None)
