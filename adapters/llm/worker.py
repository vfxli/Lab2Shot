"""LLM worker: llama.cpp + GGUF，一次聊天模板生成。运行在 third_party/llm/.venv
（pinned llama-cpp-python，GGML_CUDA 从源码编译，见 build.py），从不 import Lab2Shot 核心。

    python worker.py <job.json>

节点 llm.translate（家族 lab2shot/nodes/families/llm_text.py）。job params 里是
prepare() 解析好的最终提示词与引擎参数（「模型」选项已在节点解析成具体 GGUF，worker 不认识选项表）：

    prompt     str   最终用户消息（按 Hy-MT2 官方翻译格式组装，在节点一侧完成）
    gguf       str   权重文件（相对 weights 目录）
    n_ctx      int   上下文长度（按各模型官方 max_tokens 加提示词余量）
    sampling   dict  官方推荐采样参数（Hy-MT2 / Qwen3.5 各一套，照抄模型卡）
    no_think   bool  Qwen3.5 关 thinking：聊天模板按 enable_thinking=False 渲染（预填空 <think></think>），
                     生成不含推理过程——这是验收项，不是可选优化

聊天模板用的是 GGUF 里内嵌的那份（tokenizer.chat_template），经 llama-cpp-python 的 Jinja2ChatFormatter
渲染——与 create_chat_completion 同一条路（同样的 add_bos / special / eos 停止处理），只是多传了
enable_thinking 这一个模板变量。

Output（家族契约）：raw/result.json = 标准字段 + "text"（生成的文字）、"model"、prompt_tokens /
generated_tokens。模型经 @resident 常驻（movable=False：显存由 llama.cpp 的 CUDA 后端持有，不经
torch，「卸载」即释放）；换 GGUF = 换加载参数 = 新的常驻条目。
"""

from __future__ import annotations

from pathlib import Path

from lab2shot_worker import fail, reason, resident, serve
from lab2shot_worker.run import Run


@resident(movable=False)  # 显存不在 torch 手里（llama.cpp 自己的 CUDA 后端）：离开 GPU 就是释放
def load_model(gguf: Path, n_ctx: int):
    from llama_cpp import Llama

    # flash_attn：llama.cpp 的 FlashAttention 核（本扩展按 8.9 / 12.0 编译，4090 与 5090 实测均走 FLASH_ATTN_EXT，
    # 输出与关闭时逐字相同，计算缓冲更小、短句生成更快）
    return Llama(model_path=str(gguf), n_ctx=int(n_ctx), n_gpu_layers=-1, flash_attn=True, verbose=False)


def generate(run: Run, llm, prompt: str, sampling: dict, no_think: bool) -> tuple[str, dict]:
    """渲染聊天模板并生成一次：返回 (文字, 用量)。"""
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    formatter = Jinja2ChatFormatter(
        template=str(llm.metadata["tokenizer.chat_template"]),
        # 特殊 token（eos / bos）要按 llama-cpp-python 官方的取法拿文本（Llama.__init__ 同款）：
        # detokenize 默认不带特殊 token，eos 会解成空串，stop=[''] 会让生成一步即停（实测踩坑）
        eos_token=llm._model.token_get_text(llm.token_eos()),
        bos_token=llm._model.token_get_text(llm.token_bos()),
        stop_token_ids=[llm.token_eos()],
    )
    # no_think：Qwen3.5 官方的 enable_thinking=False（GGUF 内嵌模板自带的开关；Hy-MT2 的模板没有这个变量，不传）
    formed = formatter(messages=[{"role": "user", "content": prompt}],
                       **({"enable_thinking": False} if no_think else {}))
    tokens = llm.tokenize(formed.prompt.encode("utf-8"), add_bos=not formed.added_special, special=True)
    out = llm.create_completion(tokens, stop=formed.stop, stopping_criteria=formed.stopping_criteria, **sampling)
    return str(out["choices"][0]["text"]), dict(out["usage"])


def main(job_path: str) -> None:
    run = Run.start(job_path, "llm.translate", "LLM", gpu=False)
    job, params = run.job, run.params
    gguf = Path(job.weights_dir) / str(params["gguf"])
    run.weights(gguf, what=reason("I-LLM-GGUF"))

    with run.loading("load_model", model="GGUF"):
        llm = load_model(gguf, int(params["n_ctx"]))

    run.stage("generate")
    text, usage = generate(run, llm, str(params["prompt"]), dict(params["sampling"]), bool(params.get("no_think")))
    if not text.strip():
        fail("E-LLMTEXT-NOTEXTOUT", tokens=int(usage.get("completion_tokens") or 0))

    # 没有帧：标准字段带 seconds / load_seconds / count，不带 frames 和每秒（USD customData 不收它们）
    run.finish([], text=text.strip(), model=str(params["gguf"]),
               prompt_tokens=int(usage.get("prompt_tokens") or 0),
               generated_tokens=int(usage.get("completion_tokens") or 0))


if __name__ == "__main__":
    serve(main)
