"""构建 UniRig 运行时所需的目录树，以符号链接组合而成，原始仓库保持不变。

upstream 的 `run.py` 按相对当前目录的路径读取配置（`configs/data/...`、`configs/task/...`，转换配置中还引用
`./configs/skeleton/mixamo.yaml`），中间结果和日志也写入当前目录（`logs/`、`experiments/`）。因此每次计算
在任务自己的文件夹中构建如下目录树：

    <job work>/run/
        src      -> third_party/unirig/repo/src        （整个源码目录，单个链接）
        configs/                                        upstream configs 的副本，另加本项目的几份配置：
            data/lab2shot.yaml                          待处理模型的位置
            task/lab2shot_skeleton.yaml                 骨架阶段：检查点使用绝对路径，仅导出 npz
            task/lab2shot_skin.yaml                     权重阶段：同上
            model/unirig_ar_350m_1024_81920_float32.yaml  将 facebook/opt-350m 指向本地的 config.json
            transform/inference_skin_transform.yaml     voxel_skin 的后端改为 open3d（默认后端需要 EGL 上下文）
        clean/<asset>/raw_data.npz                      本项目写出的网格（不经过 Blender）
        clean/datalist.txt

所有输出均写入此处，仓库仅被读取。仅使用标准库。
"""

from __future__ import annotations

from pathlib import Path

ASSET = "asset"  # clean/ 下唯一资产的文件夹名
DATALIST = "datalist.txt"
SKELETON_CKPT = "skeleton/articulation-xl_quantization_256/model.ckpt"
SKIN_CKPT = "skin/articulation-xl/model.ckpt"
AR_MODEL = "unirig_ar_350m_1024_81920_float32"  # upstream 的模型配置名，在组合目录树中被覆盖
SKIN_TRANSFORM = "inference_skin_transform"


def compose_tree(repo: Path, run: Path, overrides: dict[str, str]) -> None:
    """在 `run` 下构建 run.py 所需的目录树：`src` 为指回仓库的符号链接（源码只读，体积较大，不复制），`configs`
    整体复制（均为几 KB 的 yaml），再写入 `overrides` 中的配置。

    configs 采用复制而非链接：upstream 按相对当前目录的路径读取，本项目需要在其中添加配置，
    而原始仓库不得修改，因此在副本上修改。"""
    import shutil

    run.mkdir(parents=True, exist_ok=True)
    src_link = run / "src"
    if src_link.is_symlink():
        src_link.unlink()
    src_link.symlink_to(repo / "src")
    configs = run / "configs"
    if configs.exists():
        shutil.rmtree(configs)
    shutil.copytree(repo / "configs", configs)
    for rel, text in overrides.items():
        path = configs / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _yaml(text: str) -> str:
    return text.lstrip("\n")


def data_config(clean: Path, cls: str = "inference") -> str:
    """`cls`：数据集里这一组的类别名，就是 upstream run.py 的 --cls（Datapath(cls=...)）。类别名是模型生成时的起始
    token（src/model/unirig_ar.py generate：cls 不为空就把它的 token 放在最前）：「vroid」让模型按 VRoid 人形模板
    生成，骨骼带模板里的名字（configs/skeleton/vroid.yaml）；「inference」不是类别，模型自己判断，骨骼叫 bone_N。"""
    return _yaml(f"""
# Written by Lab2Shot: where the one model to compute this time is (the same structure as upstream configs/data/quick_inference.yaml)
input_dataset_dir: &input_dataset_dir {clean}

predict_dataset_config:
  shuffle: False
  batch_size: 1
  num_workers: 0
  pin_memory: False
  persistent_workers: False
  datapath_config:
    input_dataset_dir: *input_dataset_dir
    use_prob: False
    data_path:
      {cls}: [
        [{clean / DATALIST}, 1.0],
      ]
""")


def skeleton_task(weights: Path, seed_note: str = "") -> str:
    """基于 upstream configs/task/quick_inference_skeleton_articulationxl_ar_256.yaml，修改三处：
    检查点使用绝对路径（否则 run.py 会从 Hugging Face 下载）、数据配置替换为本项目的配置、不导出 fbx（需要 Blender）。"""
    return _yaml(f"""
mode: predict
debug: False
experiment_name: lab2shot_skeleton
resume_from_checkpoint: {weights / SKELETON_CKPT}

components:
  data: lab2shot
  tokenizer: tokenizer_parts_articulationxl_256
  transform: inference_ar_transform
  model: {AR_MODEL}
  system: ar_inference_articulationxl
  data_name: raw_data.npz

writer:
  __target__: ar
  output_dir: ~
  add_num: False
  repeat: 1
  export_npz: predict_skeleton

trainer:
  max_epochs: 1
  num_nodes: 1
  devices: 1
  precision: bf16-mixed
  accelerator: gpu
  strategy: auto
""")


def skin_task(weights: Path) -> str:
    """基于 upstream configs/task/quick_inference_unirig_skin.yaml，修改内容同上。"""
    return _yaml(f"""
mode: predict
debug: False
experiment_name: lab2shot_skin
resume_from_checkpoint: {weights / SKIN_CKPT}

components:
  data: lab2shot
  transform: {SKIN_TRANSFORM}
  model: unirig_skin
  system: skin
  data_name: predict_skeleton.npz

writer:
  __target__: skin
  output_dir: ~
  add_num: False
  repeat: 1
  save_name: predict
  export_npz: predict_skin
  export_fbx: ~   # upstream defaults this key to False (not None), which it reads as "export" and imports bpy,
                  # then logs the ModuleNotFoundError inside its own try. Written as ~ it stays clean

trainer:
  num_nodes: 1
  devices: 1
  precision: bf16-mixed
  accelerator: gpu
  strategy: auto
  inference_mode: True
""")


def ar_model_config(repo: Path, opt_dir: Path) -> str:
    """基于 upstream configs/model/<AR_MODEL>.yaml，仅将 facebook/opt-350m 替换为本地 config.json 所在文件夹：
    worker 离线运行，AutoConfig.from_pretrained 不得访问网络。网络结构保持不变。"""
    text = (repo / "configs" / "model" / f"{AR_MODEL}.yaml").read_text(encoding="utf-8")
    return text.replace("pretrained_model_name_or_path: facebook/opt-350m",
                        f"pretrained_model_name_or_path: {opt_dir}")


def skin_transform_config(repo: Path) -> str:
    """基于 upstream configs/transform/<SKIN_TRANSFORM>.yaml，仅将 voxel_skin 的后端从 pyrender 改为 open3d：
    pyrender 需要 EGL 上下文，而 worker 运行在无显示的机器上。upstream 在该行的注释中给出了此替代方案。"""
    text = (repo / "configs" / "transform" / f"{SKIN_TRANSFORM}.yaml").read_text(encoding="utf-8")
    return text.replace("backend: pyrender", "backend: open3d")


def compose(repo: Path, weights: Path, work: Path, cls: str = "inference") -> Path:
    """在 `work` 下构建目录树，返回运行 runner.py 时使用的当前目录。"""
    run = Path(work) / "run"
    clean = run / "clean"
    compose_tree(repo, run, {
        "data/lab2shot.yaml": data_config(clean, cls),
        "task/lab2shot_skeleton.yaml": skeleton_task(weights),
        "task/lab2shot_skin.yaml": skin_task(weights),
        f"model/{AR_MODEL}.yaml": ar_model_config(repo, weights / "opt-350m"),
        f"transform/{SKIN_TRANSFORM}.yaml": skin_transform_config(repo),
    })
    (clean / ASSET).mkdir(parents=True, exist_ok=True)
    (clean / DATALIST).write_text(f"{ASSET}\n", encoding="utf-8")
    return run
