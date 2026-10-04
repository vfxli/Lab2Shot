+++
team = "ggml-org（llama.cpp）· 腾讯混元（Hy-MT2）· 阿里通义（Qwen3.5）"
people = "Georgi Gerganov 与 llama.cpp 社区；腾讯混元 LLM 团队；Qwen 团队"
paper = "Hy-MT2: A Family of Fast, Efficient and Powerful Multilingual Translation Models in the Wild（arXiv 2605.22064）"
paper_url = "https://arxiv.org/abs/2605.22064"
website = "https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF"
repo = "https://github.com/ggml-org/llama.cpp"
year = 2026
+++

## 这是什么

语言模型的组件节点，文字进文字出（`value.text` → `value.text`）：把中文翻译成下游要的英文提示词。一个共用扩展按 GGUF 权重表换模型——Hy-MT2-1.8B（专为翻译造的快思考多语模型，33 语）或 Qwen3.5-9B（thinking 模型，按非思考用法调用）；每个模型 Q4_K_M / Q8_0 两个量化档，供质量 / 显存取舍。

在 Lab2Shot 里，它给传统 CG 工作流补上缺失的一环：抠像类提示词、Kimodo 动作提示词都要写英文，而用户大多说中文。结果就是普通「文字」值，可直接预览、复制，也能接线驱动任何文字参数（下游参数本就支持「提升到节点」，无需改动任何下游节点）。

## 输入输出

**官方要什么、给什么**

- Hy-MT2：官方 README 的提示格式是纯文本指令（`将以下文本翻译为 {target_lang}…` + `{source_text}`），transformers 例程把用户消息过 `apply_chat_template` 后 `generate`，输出的 `response` 就是译文。GGUF 卡明确：本量化依赖 llama.cpp 的 STQ 内核（PR #22836，那是 1.25bit 极限量化的路径；Q4_K_M / Q8_0 是标准量化，随 llama.cpp 的 hunyuan-dense 架构支持加载），推荐参数 temp 0.7 / top_p 0.6 / top_k 20 / rep 1.05 / max_tokens 4096，**无默认 system prompt**。
- Qwen3.5-9B：官方模型卡——thinking 模型（助手轮以 `<think>` 开头），**不支持 Qwen3 的 `/no_think` 软开关**，关 thinking 走 `enable_thinking=False`（聊天模板 kwarg）；Instruct（非思考）模式通用任务推荐 temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence_penalty 1.5 / rep 1.0。

**我们怎么接的**

- 推理引擎是 llama.cpp（MIT），经 pinned 的 `llama-cpp-python==0.3.36` 在 worker 进程内调用；它内嵌的 llama.cpp 就是本扩展锁定的那份（2026-10-01 master，`hunyuan-dense` / `qwen35` 架构都在）。编译按 GGML_CUDA 从源码（build.py：CMAKE_ARGS + pip CUDA 13.2 工具链，头文件与 glibc ≥ 2.43 兼容的第一版），**不往核心装推理依赖**。
- 聊天模板用 GGUF 里内嵌的那份（`tokenizer.chat_template`）：Hy-MT2 是它自带的 ChatML 形模板（「无默认 system prompt」由模板天然满足，我们只给用户消息）；Qwen3.5 是它自带的模板，`enable_thinking=False` 时预填空 `<think></think>`，生成不含推理过程——这是验收项，不是可选优化。
- 采样参数**照官方照抄**：Hy-MT2 temp 0.7 / top_p 0.6 / top_k 20 / rep 1.05 / max_tokens 4096；Qwen3.5 temp 0.7 / top_p 0.8 / top_k 20 / min_p 0 / presence_penalty 1.5 / rep 1.0 / max_tokens 32768。
- 提示词组装在节点一侧：翻译套 Hy-MT2 官方「默认翻译」式，术语表开着且有词对时套官方「术语表」式（词对来自后台「提示词」页，实际内容进计算指纹）。worker 只收解析好的最终提示词与引擎参数（`extra`），不认识选项表。

**出处**：提示格式与推荐参数来自 `third_party/llm/hy_mt2/README.md`（六式表格、Inference and Deployment 一节）；Qwen3.5 的采样与 thinking 开关来自官方模型卡；聊天模板机制核对的是 `third_party/llm/repo/common/chat.h` 与内嵌 GGUF 的 `tokenizer.chat_template`（gguf-py 只读导出）。

## 在 Lab2Shot 里怎么用

- 典型接法：`文字`（value_string）→「翻译」→ 下游任何文字参数（Kimodo 的「文字描述」、抠像的提示词、文生图的「提示词」）。结果「文字」值可预览、复制，也可直接接线。
- 「翻译」：「翻译成」用官方语言全称表（33 语，中文提示用中文名——官方规定）；「术语表」开着时，管理员维护的词对按官方术语表式排在翻译指令前，CG 术语按表翻（抠像 → matting 这类）；没录入词对时自动按不带术语表翻译并提示。
- 「模型」：默认 Hy-MT2-1.8B（专为翻译，小、快），也可选 Qwen3.5-9B；各 Q4_K_M / Q8_0 两档，质量与显存取舍（见「效果和局限」）。

## 效果和局限

- **显存与加载（RTX 4090 实测，nvidia-smi 抓加载后增量，空闲基线 1278 MB）**：

  | 模型 | 量化 | 加载后显存增量 | 加载时间 |
  |---|---|---|---|
  | Hy-MT2-1.8B | Q4_K_M | 2.3 GB | 1.8 s |
  | Hy-MT2-1.8B | Q8_0 | 3.0 GB | 0.8 s |
  | Qwen3.5-9B | Q4_K_M | 7.8 GB | 2.3 s |
  | Qwen3.5-9B | Q8_0 | 12.4 GB | 10.4 s |

  四档都在 RTX 4090 的 24 GB 内；生成约 5.6 ms/词元（Hy-MT2-1.8B）上下。换模型 = 换加载参数 = 新的常驻条目，旧模型按驻留机制卸载。
- 翻译正确性（验收证据）：「抠像与遮罩是后期制作的关键步骤」→「Keying and masking are crucial steps in post-production.」；术语表式（抠像 → matting）→「This step involves matting.」。
- 关 thinking（验收证据）：Qwen3.5 按 `enable_thinking=False` 渲染内嵌聊天模板，输出不含 `<think>` 推理过程。
- 语言模型是生成式输出：同样的输入每次结果可能不同（采样），翻译的文字不保证逐字可复现；术语表约束是「尽量按表翻」，不是硬性替换。
- Hy-MT2 的官方提示格式适合整段翻译；结构化数据（代码、JSON 键名、占位符）的翻译要按官方「结构化」式，本节点暂只提供默认式与术语表式，其余格式留给后续按需加。
- 一次生成（无帧、无画面）：不参与帧范围，结果就是文字本身。

## 团队

推理引擎 llama.cpp 是 ggml 社区的项目（Georgi Gerganov 等，MIT）；Hy-MT2 是腾讯混元 LLM 团队的快思考多语翻译模型（Apache-2.0，报告 arXiv 2605.22064）；Qwen3.5 是阿里 Qwen 团队的模型（Apache-2.0，官方模型卡）。

## 模型下载和安装

- 自动安装：`lab2shot ext install llm`。下载 pinned llama.cpp 源码（本扩展用它编译 llama-cpp-python 的 CUDA 后端，也作 official 引文的核对对象）和 Hy-MT2 官方仓（提示词格式出处），建独立 Python 3.12 环境，从源码编译 llama-cpp-python（GGML_CUDA，pip CUDA 13.2 工具链），再取四份 GGUF 权重（Hy-MT2-1.8B、Qwen3.5-9B 各 Q4_K_M / Q8_0，约 1.1 + 1.9 + 6.2 + 9.8 GB）。
- 无手动下载项：两个模型都可商用，无需注册。

## 许可证说明

- 可商用。推理引擎 llama.cpp 是 MIT；Hy-MT2-1.8B 官方仓 LICENSE 是 Apache-2.0（权重目录里的 GGUF 卡 LICENSE.txt 同此）；Qwen3.5-9B 官方模型卡 license 栏是 Apache-2.0（GGUF 为 bartowski 的 imatrix 量化，随原模型同许可）。量化不改变模型本身的许可。
- Qwen3.5 是 thinking 模型，本扩展按官方 `enable_thinking=False` 的非思考用法调用，输出不含推理过程。

## 参考

- llama.cpp：https://github.com/ggml-org/llama.cpp
- Hy-MT2：https://github.com/Tencent-Hunyuan/Hy-MT2 ｜ GGUF：https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF ｜ 报告：https://arxiv.org/abs/2605.22064
- Qwen3.5-9B：https://huggingface.co/Qwen/Qwen3.5-9B ｜ GGUF：https://huggingface.co/bartowski/Qwen_Qwen3.5-9B-GGUF
- llama-cpp-python：https://github.com/abetlen/llama-cpp-python
