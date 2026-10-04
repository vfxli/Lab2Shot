+++
team = "Qwen（杭州通义实验室，Hangzhou Tongyi Laboratory Technology Co., Ltd.）"
people = "Qwen 团队"
paper = "Qwen-Image 2.1（统一文生图与图像编辑模型，7B 单流 DiT + Qwen3-VL 文本编码器 + 64 通道 RGBA VAE）"
paper_url = "https://qwen.ai/blog?id=qwen-image-2.1"
website = "https://qwen.ai/blog?id=qwen-image-2.1"
repo = "https://github.com/QwenLM/Qwen-Image-2.1"
year = 2026
+++

## 这是什么

官方介绍：Qwen-Image 2.1 把文生图和图像编辑统一在一个模型里——视觉生成部分 7B 参数（32 层单流 DiT，
混合粒度注意力和前缀 KV 复用），文本编码器是 Qwen3-VL 8B（提示词和条件图一起编码），VAE 是 64 通道
RGBA 自编码器（16× 空间压缩，原生透明）。要点：原生透明（RGBA）生成与透明层编辑；至多 10 张参考图
的多主体合成；人像与产品的身份保持；文字排版与质感。

在 Lab2Shot 里它是一个小实验：验证两件事——我们的解算通道（遮罩、深度、法线、光照）作为
参考图能否改善生成；它能否生成我们要的通道（贴图、干净 alpha、素材补全）。图进图出，不拆模型组件，
一切走既有机制。

## 输入输出

**官方要什么、给什么**（`QwenImage21Pipeline.__call__`，third_party/diffusers 检出
`src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py:505-529`）

- 吃：`prompt`（str）、`image`（一张或一组 PIL 图，至多 10 张）、`negative_prompt` + `true_cfg_scale`
  （大于 1 且给了负向提示才做 CFG，官方设计为默认无引导）、`num_inference_steps`（默认 40）、
  `width`/`height`（缺省按最后一张条件图的长宽比、面积 = `output_resolution²` 折算）、`generator`
  （种子）、`use_kv_cache`（默认开，官方注明开关会改变低精度下的逐比特结果，我们固定默认）。
- 给：`QwenImagePipelineOutput.images`——PIL 图一张（生成透明时为 RGBA）。
- 官方低显存策略：`enable_model_cpu_offload()`（`model_cpu_offload_seq = "text_encoder->transformer->vae"`）。

**我们怎么接的**

- 一个节点一条管线：`diffusers.generate`，按「模式」三选一——文生图（不传 image，宽高按官方比例表折算
  后显式传入——README「Integration with the Pipeline」的用法）；图像编辑（image 一张，跟随输入比例）；
  多参考图（主体一张 + 「参考图」列表合成官方 image 列表，主体在前）。
- 「图像」口是家族根类型（通道跟随）：带 alpha 的输入整图转成直通 alpha 的 8 位 PNG 送模型（上游示例
  的输入保真），透明层编辑；无 alpha 走标准 display PNG（压黑底 RGB）。
- 「参考图」是 `image[]` 列表口：项目里任何节点的输出都能接。画面取其显示图；数值图（遮罩、深度、
  法线……）按其声明的显示范围归一化成显示图——与视图里看到的一致，视觉模型按它训练的方式看图。
  静帧用其一张，序列取与「图像」同一帧，覆盖不到的跳过并在节点上说明；超过 10 张拒绝（官方上限）。
- 生成结果是静帧（Info.still）：一张 8 位 sRGB PNG，显示色即工作色彩空间（逐比特），透明时带直通
  alpha（项目内读取即预乘，与 Nuke 一致）。
- 种子确定结果：同参数同卡同种子可复现（use_kv_cache 固定官方默认）。

出处：输入输出依据上述检出源码与官方 README（github.com/QwenLM/Qwen-Image-2.1）；接入见
`adapters/diffusers/nodes.py`、`worker.py` 与家族 `lab2shot/nodes/families/image_generation.py`。

## 效果和局限

- **许可**：代码 Apache-2.0；权重 Qwen Research License——仅限研究与评估用途，商用需另行向 Qwen 取得
  授权。需要管理员单独授予「生成式扩散」能力。
- **显存（RTX 5090 实测）**：全量 bf16 权重约 32.4 GB（文本编码器 8B 17.5 + 变换器 7B 14.2 + VAE 0.7，
  safetensors 头实测），装不进 32 GB 的 5090，也装不进 24 GB 的 4090。worker 按空闲显存和生成分辨率选加载方式
  （结果相同，只差速度；2048 档分块解码的拼接处会有细微差别）：「变换器常驻」（变换器和 VAE 留在卡上，文本编码器逐层流过）1024 及以下要 ≥22 GB，
  实测 1024 档 20.5 GB；2048 档最后 VAE 解码时已占 25.0 GB 还要再申请 4.5 GB，5090（可用 29.45 GB）显存不够，
  官方 `enable_model_cpu_offload()` 也不行（解码本身就是 2048 画面的 VAE 激活：已占 23.5 GB 再申请 2.25 GB）。所以
  2048 档在空闲 <34 GB 的卡上改用 VAE 自带的分块解码（`vae.enable_tiling()`，256 像素一块、64 像素重叠渐变拼接，
  去噪不变），解码变小后 ≥28 GB 仍让变换器常驻：5090 实测常驻 + 分块，10 步 26 秒，整卡峰值 19.1 GB；4090
  走 offload + 分块，10 步 59 秒，整卡峰值 17.4 GB（不含空闲占用）。result.json 的 `vae_tiling` 记下这次是否分块。offload
  峰值显存以变换器权重为主、与卡无关：文生图约 16.9 GB、图条件（编辑/多参考）约 18.6 GB（torch
  max_memory_reserved；nvidia-smi 整卡约再 +1 GB CUDA 上下文）。全量上卡留给更大的卡（1024 档 ≥40 GB、2048 档
  ≥52 GB，按常驻门槛加文本编码器推算）。
- **耗时（RTX 5090，cpu offload，机器空闲时）**：文生图 512×512 10 步约 3.8 秒/步；多参考图（3 图，
  576×448）10 步约 7.7 秒/步。变换器注意力按 token 数平方缩放，所以更大的生成分辨率明显更慢。默认档按
  官方 README「Default Parameters」取原生 2K（2048 基准、40 步）——质量优先，offload 下会到每张数分钟甚至
  更久量级；512 / 768 是可用档，2048 是质量优先的慢档（原生 2K 是官方推荐最优质量）。同参数同卡同种子
  输出逐字节一致（use_kv_cache 固定官方默认）。
- 已知边界：模型按条件图比例折算生成尺寸（32 像素网格）；CFG 默认关闭（官方设计）；输出恒为 RGBA
  （VAE 原生 4 通道，普通提示词 alpha 全不透明）；PE 提示改写专用模型（PE-T2I/PE-I2I）不在本扩展范围。
