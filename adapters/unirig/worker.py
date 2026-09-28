"""UniRig worker：以 UniRig 的两个网络实现绑定家族的 worker 契约（lab2shot_worker.rigging）。
运行于 third_party/unirig/.venv，不导入 Lab2Shot 核心。

    python worker.py <job.json>        （节点 "unirig.auto_rig"）

一次计算分三步，与 upstream 的三个 launch 脚本一一对应，但不使用 Blender：

1. 整理网格（对应 upstream launch/inference/extract.sh → src/data/extract.py save_raw_data 的后半段）：
   节点提供的三角网格（厘米、Y 向上、世界坐标）经 trimesh 清理，面数超过 50 000 时用 fast_simplification 降至
   50 000（upstream faces_target_count 的默认值），计算逐顶点和逐面法线，写为 upstream 格式的 raw_data.npz。
   upstream 在这一步用 Blender 打开模型文件读取网格；此处网格已由节点提供，无需再读文件。
2. 骨架（generate_skeleton.sh 的第二步）和权重（generate_skin.sh 的第二步）：各启动一个进程运行
   runner.py（upstream run.py 的 predict 路径），写出 predict_skeleton.npz 和 predict_skin.npz。
   两个阶段分开运行与 upstream 一致，同时使两个网络的显存占用相互独立。
3. 将权重映射回完整网格（对应 merge.sh，精度更高）：upstream 的 merge 按最近顶点把权重搬到原网格；此处直接
   调用其 `src.system.skin.reskin`，参数与其 SkinWriter 完全相同（median、alpha 2.0、threshold 0.03），
   仅将 `vertices` 替换为完整网格；该函数本身的用途即是将采样点上的权重映射到任意网格的顶点。

坐标：UniRig 内部将模型归一化到 [-1, 1] 立方体（src/data/augment.py AugmentAffine），骨架阶段和权重
阶段各归一化一次（第二次的包围盒包含关节）。两次均为「整体缩放 + 平移」，因此不复现其公式，
而是从其输出数组中直接解出该相似变换（`fit_similarity`）：
  * 骨架阶段：raw_data.npz 的顶点（本项目单位）↔ predict_skeleton.npz 的顶点（归一化后的同一组点，顺序相同）；
  * 权重阶段：predict_skeleton.npz 的关节 ↔ predict_skin.npz 的关节（同一组关节，顺序相同）。
解出的变换用于将完整网格变换到权重阶段的空间，并将关节变换回本项目空间；误差在报告中核对（residual）。

参数：seed（骨架为自回归生成，不同种子产生不同骨架）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from lab2shot_worker import fail, require_cuda, say, serve
from lab2shot_worker import rigging as rg
from lab2shot_worker.run import Run

import codebase as cb

NODE = "unirig.auto_rig"
EXT = "unirig"
FACES_TARGET = 50000  # upstream launch/inference/generate_skeleton.sh 中 faces_target_count 的默认值
RESKIN = {"sample_method": "median", "alpha": 2.0, "threshold": 0.03}  # upstream SkinWriter 使用的参数
MAX_RESIDUAL = 1e-3  # 解出的相似变换在归一化立方体中允许的最大残差（立方体边长为 2）


def fit_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, float]:
    """求解将 `src` 变换为 `dst` 的「整体缩放 + 平移」，返回 (缩放, 平移 [3], 残差)。
    UniRig 的归一化正是此类变换（AugmentAffine：先平移到中心，再按最长边缩放），因此解是精确的。
    残差用于核对该前提；残差过大说明前提不成立，调用方应立即停止计算。"""
    src = np.asarray(src, np.float64).reshape(-1, 3)
    dst = np.asarray(dst, np.float64).reshape(-1, 3)
    if len(src) < 2 or len(src) != len(dst) or not float(((src - src.mean(0)) ** 2).sum()) > 0:
        fail("E-UNIRIG-FRAMEPOINTS", before=len(src), after=len(dst))
    a, b = src - src.mean(0), dst - dst.mean(0)
    denom = float((a * a).sum())
    scale = float((a * b).sum() / denom)
    shift = dst.mean(0) - scale * src.mean(0)
    residual = float(np.abs(src * scale + shift - dst).max())
    return scale, shift, residual


def prepare_asset(points: np.ndarray, faces: np.ndarray, asset_dir: Path) -> dict:
    """第 1 步：网格 → upstream 的 raw_data.npz。处理与 src/data/extract.py save_raw_data 的后半段相同
    （trimesh 清理 → 超过 FACES_TARGET 时降面 → 逐顶点、逐面法线），写出时使用 upstream 的
    RawData（沿用其字段与检查，不另定义格式）。前半段（用 Blender 打开模型文件读取网格）不需要：
    网格来自 Lab2Shot 的数据包。"""
    import fast_simplification
    import trimesh

    from src.data.raw_data import RawData

    mesh = trimesh.Trimesh(vertices=points, faces=faces)
    v = np.asarray(mesh.vertices, np.float32)
    f = np.asarray(mesh.faces, np.int64)
    simplified = len(f) > FACES_TARGET
    if simplified:
        v, f = fast_simplification.simplify(v, f, target_count=FACES_TARGET)
        mesh = trimesh.Trimesh(vertices=v, faces=f)
    raw = RawData(
        vertices=np.asarray(mesh.vertices, np.float32), vertex_normals=np.asarray(mesh.vertex_normals, np.float32),
        faces=np.asarray(mesh.faces, np.int64), face_normals=np.asarray(mesh.face_normals, np.float32),
        joints=None, skin=None, parents=None, names=None, matrix_local=None)
    raw.check()
    asset_dir.mkdir(parents=True, exist_ok=True)
    raw.save(path=str(asset_dir / "raw_data.npz"))
    return {"faces_in": int(len(faces)), "faces_used": int(len(mesh.faces)), "simplified": bool(simplified)}


def run_stage(run_dir: Path, task: str, seed: int, what: str) -> float:
    """运行一个阶段：runner.py 在独立进程中运行，当前目录为组合好的目录树。返回其报告的显存峰值（MB）。"""
    runner = Path(__file__).resolve().parent / "runner.py"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(run_dir), os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    done = subprocess.run([sys.executable, str(runner), f"configs/task/{task}.yaml", str(seed)],
                          cwd=run_dir, env=env, text=True, capture_output=True)
    for line in (done.stdout or "").splitlines() + (done.stderr or "").splitlines():
        print(line, flush=True)
    if done.returncode != 0:
        fail("E-UNIRIG-STAGE", stage=what, code=done.returncode)
    peaks = [float(l.split()[-1]) for l in (done.stdout or "").splitlines() if l.startswith("LAB2SHOT_PEAK_MB")]
    return max(peaks) if peaks else 0.0


def main(job_path: str) -> None:
    # 两个网络在各自的进程中运行（run_stage）并各自报告显存峰值；本进程不使用 GPU，因此按 CPU 方式记录，
    # 但仍在开始工作前进行 CUDA 检查
    run = Run.start(job_path, NODE, "UniRig", gpu=False)
    job = run.job
    require_cuda("UniRig")
    weights = job.weights_dir
    run.weights(weights / cb.SKELETON_CKPT, weights / cb.SKIN_CKPT, weights / "opt-350m" / "config.json")
    seed = int(job.params["seed"])

    points, faces = rg.read_mesh_job(job.inputs["mesh"])
    run.stage("整理网格")
    run_dir = cb.compose(job.repo_dir, weights, job.dir)
    sys.path.insert(0, str(run_dir))  # 组合好的目录树：upstream 的 src 从此处导入
    asset = run_dir / "clean" / cb.ASSET
    mesh_info = prepare_asset(points, faces, asset)

    run.stage("UniRig 生成骨架")
    peak_ar = run_stage(run_dir, "lab2shot_skeleton", seed, "骨架")
    skeleton = np.load(asset / "predict_skeleton.npz", allow_pickle=True)

    run.stage("UniRig 预测蒙皮权重")
    peak_skin = run_stage(run_dir, "lab2shot_skin", seed, "蒙皮权重")
    skin = np.load(asset / "predict_skin.npz", allow_pickle=True)

    run.stage("权重放回整张网格")
    from src.system.skin import reskin  # upstream 的函数，参数与其 SkinWriter 一致

    raw = np.load(asset / "raw_data.npz", allow_pickle=True)
    # 本项目空间 -> 骨架阶段的归一化空间 -> 权重阶段的归一化空间
    s1, b1, r1 = fit_similarity(raw["vertices"], skeleton["vertices"])
    joints_ar = np.asarray(skeleton["joints"], np.float64)
    s2, b2, r2 = fit_similarity(joints_ar, np.asarray(skin["joints"], np.float64))
    if max(r1, r2) > MAX_RESIDUAL:
        fail("E-UNIRIG-FRAME", residual=round(max(r1, r2), 6))
    in_skin_space = (points * s1 + b1) * s2 + b2
    parents_raw = list(skeleton["parents"])
    parents = [None if p is None or int(p) < 0 else int(p) for p in parents_raw]
    weights = reskin(sampled_vertices=np.asarray(skin["vertices"], np.float64), vertices=in_skin_space,
                     parents=parents, faces=faces, sampled_skin=np.asarray(skin["skin"], np.float64), **RESKIN)
    joints = (joints_ar - b1) / s1
    names = [str(n) for n in skeleton["names"]] if skeleton["names"] is not None else [f"bone_{i}" for i in range(len(joints))]
    if mesh_info["simplified"]:
        say("N-UNIRIG-SIMPLIFIED", faces=mesh_info["faces_in"], used=mesh_info["faces_used"])
    rg.write_rig(run, joints, [-1 if p is None else p for p in parents], names, weights,
                 method="UniRig", model="articulation-xl", seed=seed, joints_count=int(len(joints)),
                 vertices=int(len(points)), **mesh_info,
                 gpu_peak_mb=round(max(peak_ar, peak_skin)))  # 两个阶段进程的峰值，而非本进程


if __name__ == "__main__":
    serve(main)
