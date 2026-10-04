"""LLM text family: 文字进文字出（value.text → value.text），一次推理没有帧。

语言模型（llama.cpp + GGUF，adapters/llm）的组件节点共用这个家族：不碰画面、没有帧范围，把接进来的
文字按节点自己的规则组成最终提示词（翻译按 Hy-MT2 官方格式，在接入层），经 extra 送给 worker；
worker 加载 GGUF、渲染模型自带的聊天模板、生成一次，把文字写进 result.json。

原始契约（worker 写、convert 读）：

    raw/result.json   标准字段（seconds / load_seconds / count …）+ "text"（生成的文字）

节点的计算指纹除了参数本身，还带管理员数据的解析结果（family 的 `resolved`：术语对的实际内容——
管理员改了词对，用它的节点重算；只记开关不够，词对内容变了也得重算）。"""

from __future__ import annotations

from typing import ClassVar

from ...data.values import TEXT, read as read_value, value_packet
from ...errors import NothingToCook
from ...messages import Msg
from ..base import Port
from .base import Job, WorkerNode

class LlmText(WorkerNode):
    """文字进文字出的 LLM 节点：prepare 组最终提示词，worker 生成一次，convert 读回文字。

    节点自己声明 `assemble(ctx, text) -> dict`（extra：最终提示词、GGUF、采样参数……）；用到的一切都在节点参数里
    （存在节点图里，按参数进计算指纹），没有后台数据。"""

    inputs: ClassVar[tuple[Port, ...]] = (Port("text", "value.text"),)
    outputs: ClassVar[tuple[Port, ...]] = (Port("text", "value.text", may_be_empty=True),)
    main: ClassVar[str] = "text"
    nothing_without: ClassVar[str] = "text"  # no text in, nothing out (prepare): known before the cook for a constant text
    # 没有画面：既不跟随谁的画面，也不是 frame source；结果就是文字本身
    picture: ClassVar[str] = ""

    @classmethod
    def assemble(cls, ctx, text: str) -> dict:
        """这一节点对一段文字的最终提示词与推理参数（extra）：子类实现。"""
        raise NotImplementedError(f"{cls.id}: assemble() does not say how this text becomes the final prompt")

    @classmethod
    def prepare(cls, ctx) -> Job:
        text = read_value(ctx.input("text")).value
        if not str(text).strip():  # nothing to translate (a card's 文字 left empty: it is optional): nothing given,
            # as expected — a parameter it drives keeps its own value, nothing is said downstream
            raise NothingToCook(Msg("I-LLMTEXT-NOTEXT", node=ctx.label), expected=True)
        return Job(None, extra=cls.assemble(ctx, str(text)))

    @classmethod
    def convert(cls, ctx, raw, job) -> dict:
        return {"text": value_packet(ctx.outputs["text"], TEXT, str(raw.result()["text"]).strip())}
