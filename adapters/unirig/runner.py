"""跑一次 UniRig 的 predict：upstream `run.py` 的 predict 那条路，一行不多一行不少。

为什么不直接用 `run.py`：它开头有 `from src.data.extract import get_files`，而 `src/data/extract.py` 第一行
是 `import bpy`——那是它命令行模式用 Blender 读模型文件用的。Lab2Shot 的网格是从我们自己的 USD 数据包里
拿的（worker.py），不经过 Blender，所以这一条 import 只会让扩展环境里多装一个 240 MB、GPL-3.0 的 Blender。
除此之外，下面每一步都和 `run.py` 的 predict 分支一一对应（对照 run.py 第 60–265 行），配置也是从同一批
yaml 里读的；模型、采样、写出全是 upstream 自己的代码。

    python runner.py configs/task/<任务>.yaml <随机种子>

当前目录要是 codebase.compose() 拼出来的那棵树（配置按相对路径读）。
"""

from __future__ import annotations

import os
import sys

import yaml
from box import Box


def load(kind: str, path: str) -> Box:
    """run.py 的 load()。"""
    if path.endswith(".yaml"):
        path = path.removesuffix(".yaml")
    return Box(yaml.safe_load(open(path + ".yaml", encoding="utf-8")))


def main(task_path: str, seed: int) -> None:
    import lightning as L
    import torch

    from src.data.dataset import DatasetConfig, UniRigDatasetModule
    from src.data.transform import TransformConfig
    from src.model.parse import get_model
    from src.system.parse import get_system, get_writer
    from src.tokenizer.parse import get_tokenizer
    from src.tokenizer.spec import TokenizerConfig

    torch.set_float32_matmul_precision("high")
    L.seed_everything(seed, workers=True)

    task = load("task", task_path)
    assert task.mode == "predict", f"runner.py runs predict only, this task is {task.mode}"

    data_config = load("data", os.path.join("configs/data", task.components.data))
    transform_config = load("transform", os.path.join("configs/transform", task.components.transform))

    tokenizer_config = task.components.get("tokenizer", None)
    if tokenizer_config is not None:
        tokenizer_config = TokenizerConfig.parse(config=load("tokenizer", os.path.join("configs/tokenizer", tokenizer_config)))

    data_name = task.components.get("data_name", "raw_data.npz")
    predict_dataset_config = DatasetConfig.parse(config=data_config.predict_dataset_config).split_by_cls()
    predict_transform_config = TransformConfig.parse(config=transform_config.predict_transform_config)

    model_config = task.components.get("model", None)
    if model_config is not None:
        model_config = load("model", os.path.join("configs/model", model_config))
        tokenizer = get_tokenizer(config=tokenizer_config) if tokenizer_config is not None else None
        model = get_model(tokenizer=tokenizer, **model_config)
    else:
        model = None

    data = UniRigDatasetModule(
        process_fn=None if model is None else model._process_fn,
        train_dataset_config=None,
        predict_dataset_config=predict_dataset_config,
        predict_transform_config=predict_transform_config,
        validate_dataset_config=None,
        train_transform_config=None,
        validate_transform_config=None,
        tokenizer_config=tokenizer_config,
        debug=False,
        data_name=data_name,
        datapath=None,
        cls=None,
    )

    writer = get_writer(**task.writer, order_config=predict_transform_config.order_config)
    system = get_system(
        **load("system", os.path.join("configs/system", task.components.system)),
        model=model,
        optimizer_config=None,
        loss_config=None,
        scheduler_config=None,
        steps_per_epoch=1,
    )
    trainer = L.Trainer(callbacks=[writer], logger=None, **task.get("trainer", {}))
    trainer.predict(system, datamodule=data, ckpt_path=task.resume_from_checkpoint, return_predictions=False)
    # 这一次用掉的显存峰值，worker 从这一行读回去写进交付物的来源信息
    print(f"LAB2SHOT_PEAK_MB {torch.cuda.max_memory_allocated() / 2**20:.0f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
