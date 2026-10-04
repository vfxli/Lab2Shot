+++
team = "Lightricks"
people = "Yoav HaCohen, Nisan Chiprut, Benny Brazowski, Daniel Shalem, Dudu Moshe, Eitan Richardson, Eran Levin, Guy Shiran, Nir Zabari, Ori Gordon, Poriya Panet, Sapir Weissbuch, Victor Kulikov, Yaki Bitterman, Zeev Melumian, Ofir Bibi"
paper = "LTX-Video: Realtime Video Latent Diffusion"
paper_url = "https://arxiv.org/abs/2501.00103"
website = "https://ltx.io/model/ltx-2-5"
repo = "https://github.com/Lightricks/LTX-2"
year = 2026
+++

## 这是什么

LTX-2.5 是 Lightricks 的开源视频模型（22B 参数，音视频联合）。官方在它上面训练了一系列 IC-LoRA（以参考视频为条件的视频到视频 LoRA），每个 LoRA 把底模变成一个专门的处理工具。

在 Lab2Shot 里，这个扩展包是这些功能的**共用底座，本身没有节点**。它负责三样东西：固定版本的官方代码（ltx-core、ltx-pipelines）、运行环境，以及底模文件。用到它的扩展包：

- **LTX-2.5 Alpha Gen**（`alphagen.matte`）：从原画面抠出 alpha；
- **LTX-2.5 Clean Plate**（`cleanplate.clean_plate`）：去掉画面里的人和车，得到干净背景板。

## 怎么共用

- 底模文件只存一份。每个功能扩展包都声明同一组底模文件（SHA-256 相同），安装器按哈希判断：已有的文件直接硬链接过来，不会重复下载，也不会多占磁盘。
- 运行层 `adapters/ltx/ltx_runtime.py`：加载底模、挂上一组 LoRA、跑一次 IC-LoRA 推理。它不知道任何具体功能。每个功能只在自己的 worker 里写死两样东西：用哪个 LoRA，以及一个固定的 `Recipe`（提示词、种子、窗口帧数、处理尺寸上限）。用户在节点上只能选处理分辨率。
- 推理方式是 LoRA 模型卡推荐的：蒸馏版 transformer（加载时转为 fp8）、LoRA 强度 1.0、只跑第一阶段（原生分辨率、8 步、不做 CFG）、只算视频（不生成音频），最后用视频 VAE 解码。
- 严格对应输入：每个输入帧对应一个输出帧，尺寸相同，不会多出帧。为满足模型要求补上的部分会在输出时全部去掉：
  - 边长补到 32 的倍数（反射补边，不拉伸）；
  - 帧数补到 8n+1（重复最后一帧）；
  - 超长镜头切成相互重叠的窗口，重叠处交叉淡化。

## 以后加一个新功能（新 LoRA）要做什么

1. 新建 `adapters/<上游名>/`：
   - `extension.py`：`requires = ("ltx",)`，`source` / `env` / 底模文件从本扩展的 `extension.py` 引入，再加自己的 LoRA 文件；
   - `requirements.txt`：复制本扩展的那份。
2. `worker.py`：写一个 `Recipe` 常量，用 `ltx_runtime.load(...)` 加 `run_shot(...)` 跑，输出前校验帧数和尺寸。
3. `nodes.py`：按功能选家族和端口。只开放处理分辨率，不开放提示词、种子、步数、LoRA、帧数。
4. `i18n/zh.toml`、`i18n/en.toml`，再加 `docs.md`。

运行层本身不用改。只有遇到新的推理方式（比如两阶段放大）时，才需要在 `ltx_runtime.py` 里加一条路径。

## 模型下载和安装

- `lab2shot ext install ltx` 只下载底模，用不着单独装。装任意一个功能扩展包时，底模会一起装好。
- 底模放在 Hugging Face 的 Lightricks/LTX-2.5，访问有门槛：要先登录 Hugging Face，在仓库页面点「Agree and Access」（同意共享联系方式）。
- 文件（都校验 SHA-256）：
  - 蒸馏版 transformer bf16，42.0 GB；
  - Gemma 4 12B 文本编码器，26.3 GB：只用来把固定提示词编码一次，结果缓存后就不再加载；
  - 视频 VAE，1.5 GB。
- 运行环境：PyTorch 2.10（cu128，4090 和 5090 都能用），transformers 5.14.1。ltx-core 和 ltx-pipelines 直接从固定版本的代码目录导入，不通过 pip 安装。

## 许可证说明

LTX-2.x Community License：

- 个人和年收入低于 1000 万美元的公司可以免费使用，包括商用；
- 年收入达到 1000 万美元的公司，除非商业的研究和评估外，任何使用都要向 Lightricks 购买授权；
- 再分发要附带许可证原文和其中的使用限制（附件 A）；
- 不能去掉模型里的安全、来源标记和水印功能。

## 参考

- 代码：https://github.com/Lightricks/LTX-2
- 许可证原文：https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
- 底模：https://huggingface.co/Lightricks/LTX-2.5
- IC-LoRA 说明：https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra
