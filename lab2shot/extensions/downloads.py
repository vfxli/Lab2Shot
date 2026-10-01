"""Files more than one extension downloads, each pinned once here: where it comes from and its sha256.

Several methods run the same third-party network (ViTDet finds the people for HaMeR, SAM 3D Body and TRAM; GVHMR and
WHAM read the same HMR2.0a and ViTPose weights; ViPE runs UniDepth, Video Depth Anything, Depth Anything 3, GeoCalib
and DROID-SLAM). The installer already keeps one copy of identical files; this table keeps one statement of what the
file is, so a new pin (a moved URL, a new revision) is made in one place and every extension follows. Each extension
still says where the file goes in its own weights folder and what it uses it for (Download.weight, or the Download's
fields in the extension's own table), since their workers look for it in different places.

Like body models (extensions/manual.py body_model_weight), which are shared the same way but downloaded by hand."""

from __future__ import annotations

from dataclasses import dataclass

from .spec import GitSource, Weight


@dataclass(frozen=True)
class Download:
    source: str  # the URL (a Hugging Face file at a pinned revision: hf())
    sha256: str  # of the file (kind "url") or of the archive (kind "zip")
    kind: str = "url"
    repo: str = ""  # a Hugging Face file: its repository, revision and path in it (tables that list them apart)
    revision: str = ""
    filename: str = ""

    def weight(self, *, key: str, dest: str, note: str, **kw) -> Weight:
        """This file as one extension's weight: its own key, where its worker reads it, what it is for there."""
        return Weight(key=key, kind=self.kind, source=self.source, dest=dest, sha256=self.sha256, note=note, **kw)


def hf(repo: str, revision: str, filename: str, sha256: str) -> Download:
    """A Hugging Face file at a pinned commit (the file's LFS sha256)."""
    return Download(f"https://huggingface.co/{repo}/resolve/{revision}/{filename}", sha256, repo=repo, revision=revision,
                    filename=filename)


# people detection: hamer, sam_3d_body, tram
VITDET_H = Download("https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl",
                    "8601bc52000c8a87960f3db6a9672596c5e06ce33bc30a3b8f96a96efe42ae60")
# GVHMR's checkpoints as its authors mirror them: gvhmr, wham (WHAM's own script fetches the same bytes)
GVHMR_MIRROR, GVHMR_MIRROR_REVISION = "camenduru/GVHMR", "21b32d5389e2e59c0737d4c4095bbc0b8c23f66b"
YOLOV8X = hf(GVHMR_MIRROR, GVHMR_MIRROR_REVISION, "yolo/yolov8x.pt",
             "c4d5a3f000d771762f03fc8b57ebd0aae324aeaefdd6e68492a9c4470f2d1e8b")
HMR2A = hf(GVHMR_MIRROR, GVHMR_MIRROR_REVISION, "hmr2/epoch=10-step=25000.ckpt",
           "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e")
VITPOSE_H = hf(GVHMR_MIRROR, GVHMR_MIRROR_REVISION, "vitpose/vitpose-h-multi-coco.pth",
               "50e33f4077ef2a6bcfd7110c58742b24c5859b7798fb0eedd6d2215e0a8980bc")
# SLAM: tram, vipe
DROID_SLAM = Download("https://drive.usercontent.google.com/download?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&export=download&confirm=t",
                      "46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526")
# lens: geocalib, vipe
GEOCALIB_PINHOLE = Download("https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar",
                            "86d6aeacd8bbd974c59ce39f61854e00d36911c732ad89be471476fd708722ac")
# depth: unidepth, videodepthanything, depthanything3, track4world, vipe
UNIDEPTH_V2_L = hf("lpiccinelli/unidepth-v2-vitl14", "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9", "model.safetensors",
                   "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6")
UNIDEPTH_V2_L_CONFIG = hf(UNIDEPTH_V2_L.repo, UNIDEPTH_V2_L.revision, "config.json",
                          "09eb0ea8de53a6c9a1d428ac98c79847fe2602ea417701261f3f628099a30816")
VDA_SMALL = hf("depth-anything/Video-Depth-Anything-Small", "256875362cff76724b920335dfb4b29dd611f66e",
               "video_depth_anything_vits.pth", "13379300b739e659f076a59d52e9801bd8d38c541a7e71f73bbca4dcfb013609")
DA3NESTED_CONFIG = hf("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "b2359bdf726fb44ef62acca04d629dcf158053e7", "config.json",
                      "09adf89474017e717bc05aa86fd3a378708ba8914b036d61874eced328069468")
DA3NESTED_MODEL = hf("depth-anything/DA3NESTED-GIANT-LARGE-1.1", "b2359bdf726fb44ef62acca04d629dcf158053e7",
                     "model.safetensors", "8ebe871a022ed58d2fc8fdfb2ebdb31d57b60fe39611c849095851a7b7c6020c")
DA3METRIC_MODEL = hf("depth-anything/DA3METRIC-LARGE", "4010e39f3634a45bc60553321fb49fb760bd594e", "model.safetensors",
                     "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776")
# faces: mediapipe_face, smirk
FACE_LANDMARKER = Download("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                           "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff")  # 3,758,596 bytes
# DINOv2's network code (architecture only), what torch.hub would fetch: mapanything, woftsam
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_CODE = Download(f"https://github.com/facebookresearch/dinov2/archive/{DINOV2_COMMIT}.zip",
                       "04276715cddb29d45d05bff3a6fc132224dc27749b279ac98ad2ce4620e20d48", kind="zip")


# ------------------------------------------------------------------ source pins several extensions build from

def pip_git(name: str, source: GitSource) -> str:
    """A pip requirement installing `name` from a pinned git source."""
    return f"{name} @ git+{source.url}@{source.commit}"


def archive_zip(source: GitSource) -> str:
    """A GitHub source's archive of its pinned commit (a zip download of the code)."""
    return f"{source.url.removesuffix('.git')}/archive/{source.commit}.zip"


PYTORCH3D = GitSource("https://github.com/facebookresearch/pytorch3d.git", "33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba")  # v0.7.9: gvhmr, pixel3dmm
CHUMPY = GitSource("https://github.com/mattloper/chumpy.git", "580566eafc9ac68b2614b64d6f7aaa84eebb70da")  # hamer, pixel3dmm, smirk, tram, wham
DETECTRON2 = GitSource("https://github.com/facebookresearch/detectron2.git", "a1ce2f956a1d2212ad672e3c47d53405c2fe4312")  # hamer, sam_3d_body
# TRAM's build was installed and checked at this older commit; it has not been tried at DETECTRON2's: kept apart,
# here with the other pin, until a reinstall of TRAM checks it there
DETECTRON2_TRAM = GitSource(DETECTRON2.url, "a59f05630a8f205756064244bf5beb8661f96180")
VITPOSE = GitSource("https://github.com/ViTAE-Transformer/ViTPose.git", "d5216452796c90c6bc29f5c5ec0bdba94366768a")  # hamer, wham (WHAM's submodule)
