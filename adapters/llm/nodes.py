"""Nodes provided by the LLM extension（llama.cpp 引擎 MIT；Hy-MT2 / Qwen3.5 权重 Apache-2.0）。

两个文字组件节点，共用一个 worker（单次生成、写 result.json 的 "text"）：
「翻译」按 Hy-MT2 官方的默认翻译提示格式组装。
采样参数照各模型官方推荐照抄，见 extension.py 的模型表。
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from lab2shot.sdk import Cost, LlmText, NodeParams, Official, OptionTrait, P, Param

# The target languages: option ids -> Hy-MT2's full language names for its prompt, in hy_mt2.toml (data the model reads;
# the options' names for people are the catalogue's, node.llm.translate.param.target_lang.option.<id>)
HY_MT2 = tomllib.loads((Path(__file__).with_name("hy_mt2.toml")).read_text(encoding="utf-8"))
LANGUAGES = tuple(HY_MT2["languages"])
Lang = Literal[LANGUAGES]

# 模型表（与 extension.py 的四份 GGUF 对齐）：引擎与采样按官方推荐。worker 只收 prepare 解析出的
# extra（gguf / n_ctx / sampling / no_think），不认识这张表。
MODELS = {
    "hy_mt2_q4": {"gguf": "hy_mt2/Hy-MT2-1.8B-Q4_K_M.gguf", "n_ctx": 8192,
                  "no_think": False,
                  "sampling": {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05, "max_tokens": 4096}},
    "hy_mt2_q8": {"gguf": "hy_mt2/Hy-MT2-1.8B-Q8_0.gguf", "n_ctx": 8192,
                  "no_think": False,
                  "sampling": {"temperature": 0.7, "top_p": 0.6, "top_k": 20, "repeat_penalty": 1.05, "max_tokens": 4096}},
    "qwen35_q4": {"gguf": "qwen35_9b/Qwen_Qwen3.5-9B-Q4_K_M.gguf", "n_ctx": 36864,
                  "no_think": True,
                  "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                               "repeat_penalty": 1.0, "max_tokens": 32768}},
    "qwen35_q8": {"gguf": "qwen35_9b/Qwen_Qwen3.5-9B-Q8_0.gguf", "n_ctx": 36864,
                  "no_think": True,
                  "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5,
                               "repeat_penalty": 1.0, "max_tokens": 32768}},
}
Model = Literal["hy_mt2_q4", "hy_mt2_q8", "qwen35_q4", "qwen35_q8"]


def model_param(default: str) -> Model:
    return P(default, group="model")


def _engine(params: dict) -> dict:
    """解析后的引擎参数（extra 的一部分）：GGUF、上下文长度、采样、thinking 开关。"""
    m = MODELS[params["model"]]
    return {"gguf": m["gguf"], "n_ctx": m["n_ctx"], "sampling": m["sampling"], "no_think": m["no_think"]}


class Translate(LlmText):
    id = "llm.translate"
    version = 3  # 2：去掉术语表，只做翻译（官方「默认翻译」提示）；3: target_lang values are English ids (hy_mt2.toml)
    retired_params = frozenset({"glossary_on", "glossary"})  # 去掉的术语表：存了它的节点图照样能打开
    # 引的是 Hy-MT2 官方 README：默认翻译的提示格式和 transformers 例程——文字进去（{source_text}），
    # 翻译结果出来（例程里的 response）。语言全称表也是这份 README 的。
    official = Official(
        cite=("third_party/llm/hy_mt2/README.md:76-77",
              "third_party/llm/hy_mt2/README.md:139-151"),
        takes={"text": "source_text"},
        gives={"text": "response"},
    )
    on_node = ("target_lang", "model")
    category = "value"
    runtime = "llm"
    # 显存实测（RTX 4090，空闲基线 1278 MB）：Hy-MT2-1.8B Q4_K_M 2.3 GB / Q8_0 3.0 GB；Qwen3.5-9B Q4_K_M 7.8 GB / Q8_0 12.4 GB
    cost = Cost(gpu=True, vram_gb=2.3, seconds_per_frame=None, whole=True, note=True)
    traits = (
        OptionTrait(Param("model").one_of("hy_mt2_q8"), vram_gb=3.0),
        OptionTrait(Param("model").one_of("qwen35_q4"), vram_gb=7.8),
        OptionTrait(Param("model").one_of("qwen35_q8"), vram_gb=12.4),
    )

    class Params(NodeParams):
        target_lang: Lang = P("english", group="translation", worker=False)
        model: Model = model_param("hy_mt2_q4")

    @classmethod
    def assemble(cls, ctx, text: str) -> dict:
        # the official Default Translation prompt, the language by its full name (hy_mt2.toml)
        prompt = HY_MT2["prompt"].format(lang=HY_MT2["languages"][ctx.params["target_lang"]], text=text)
        return {"prompt": prompt, **_engine(ctx.params)}


NODES = (Translate,)
