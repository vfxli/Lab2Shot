+++
team = "NVIDIA 多伦多 AI 实验室（NVIDIA Toronto AI Lab）+ 多伦多大学、Vector Institute"
people = "Ruofan Liang, Zan Gojcic, Huan Ling, Jacob Munkberg, Jon Hasselgren, …, Sanja Fidler, Zian Wang"
paper = "DiffusionRenderer: Neural Inverse and Forward Rendering with Video Diffusion Models（CVPR 2025 口头报告）"
paper_url = "https://arxiv.org/abs/2501.18590"
website = "https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/"
repo = "https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer"
year = 2025
+++

## 这是什么

Cosmos-Transfer1-DiffusionRenderer 是一套基于 NVIDIA Cosmos World Foundation Models 的视频重打光
框架，为输入的画面或视频做高质量的去光照（de-lighting）与重打光（re-lighting）（上游 README 开头）。
README 还写明：它由 NVIDIA 的 Cosmos 框架驱动，在研究项目 DiffusionRenderer 的基础上改进了数据
流程、提高了视觉保真度。

在 Lab2Shot 里，它接成两个节点。「DiffusionRenderer PBR 通道」走去光照那条路：从画面拆出每一帧的
基础色（去掉光影的 base color）、法线、深度图、粗糙度、金属度，相当于把实拍素材拆成一套渲染器用的
AOV。「DiffusionRenderer 重新打光」走重打光那条路：拿这套 G-buffer 加一张新的 HDRI，重新渲出
这段镜头在新环境光下的样子。两个都用 Cosmos 7B 视频扩散模型，一次处理一小段连续的帧，所以前后帧
比逐帧算的方法稳定得多。

## 输入输出

**官方要什么、给什么**（两个模型，两条命令）

- **逆向渲染**（`inference_inverse_renderer.py`）：吃 `--dataset_path`——一个装着视频帧或图片的目录；
  给的是 `--inference_passes` 列出的几张通道图，默认就是
  `basecolor normal depth roughness metallic` 五张，每张各跑一遍 7B 视频模型。
- **正向渲染**（`inference_forward_renderer.py`）：吃 `--dataset_path`，官方写明它应指向逆向渲染的输出，
  也就是上一步的通道图；环境光走 `--use_custom_envmap` 那一路，`envlight_path` 经 `process_environment_map` 拆成
  `env_ldr` 和 `env_log` 两张送进模型；给的是重新打光后的画面 `output`。

**我们怎么接的**

- 「DiffusionRenderer PBR 通道」：「RGB」口就是 `--dataset_path` 那段画面，
  五个输出口「基础色」「法线图」「深度图」「粗糙度」「金属度」就是官方默认的那五个 pass，一样不多一样不少。
- 「DiffusionRenderer 重新打光」：「HDRI」口就是 `envlight_path`（「旋转」「曝光」两个参数对应
  `--rotate_light` 和曝光处理）。
- **不一样的一处**：正向那一步上游要的是**上一步的通道图**，节点收的是**画面**——
  worker 先跑一遍逆向模型拆出通道图，再送进正向模型。也就是一个节点里跑了官方的两步；
  只想要通道图就用另一张卡。

出处：简介来自论文摘要（arXiv 2501.18590）和上游 README；输入输出依据
`third_party/diffusionrenderer/repo/cosmos_predict1/diffusion/inference/inference_inverse_renderer.py`、
`inference_forward_renderer.py` 和 `adapters/diffusionrenderer/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 拆 G-buffer：读取序列 → **DiffusionRenderer PBR 通道** → 每帧一组 G-buffer（基础色、法线、深度图、粗糙度、金属度）。基础色写成场景线性 EXR（ACEScg），法线是相机空间（X 向右、Y 轴向上、Z 轴朝向镜头），深度图是相对值（0 近、1 远），粗糙度、金属度 0–1。可以写成 EXR 进 Nuke 做重打光 / 调色的辅助通道，法线和深度图可以当 relight、雾效、景深的参考。
- 重打光：读取序列 + 一张 HDRI（例如 **DiffusionLight** 节点算出来的，或者现场拍的 HDRI）→ **DiffusionRenderer 重新打光**（实验性）→ 新光照下的序列图。单张 HDRI 用「读取序列」节点选那张 EXR。HDRI 方向约定和 DiffusionLight 一样：**经纬图正中间那一列 = 镜头看的方向**，顶上 = 画面上方。「旋转」参数绕竖直轴转环境（+90° 把原来正前方的物体转到镜头左边），「曝光」整体加减 EV。
- 什么素材好：画面清楚、曝光正常、主体占画面比例大的镜头（人像、物体、室内外场景都行）。镜头越长，分段越多、越慢。
- 关键参数：
  - 每段帧数（max_frames）：模型一次看多少帧（1、9、17 … 57）。默认 41：24 GB 显卡上网络和编解码器能同时放进显存。57 是模型训练时的长度，时序最稳，但在 24 GB 卡上需要把网络临时挪到内存（多占约 15 GB 内存，每段多几秒）。
  - 处理分辨率（resolution）：模型内部画布的宽度，默认 1280（画布 1280×704，和模型训练时一样）。画布永远是 16:9 横幅。和官方一样，画面按比例放大到盖满画布、裁出 16:9 的一块来算；不是 16:9 的画面一块盖不全，就沿长边裁成几块（相邻块至少重叠四分之一），每块各算一遍，再在重叠处渐变拼回整幅（深度先按重叠处对齐）、缩放回原画面大小。接近 16:9（裁掉不超过一成）的画面只算一块：按比例整幅放进画布，四周窄边用边缘像素延伸。最多裁两块：要裁三块及以上的画面（竖幅、方形）每块看到的画面太少，改为按比例整幅放进画布、四周镜像填充，只算一遍。裁两块的画面要算两遍（时间翻倍）。实测（有真值，整幅）：Hypersim 4:3、17 帧，法线误差中位数 26.2°（整幅放进画布、四周镜像填充）→ 21.2°（裁两块），基础色误差 0.223 → 0.200；Sintel 2.35:1、5 帧，基础色误差 0.116 → 0.110（裁两块）；Hypersim 裁成 3:4 竖幅、5 帧，裁三块 30.3°，整幅放进画布、镜像填充 22.4°。改小（1024、960、768…）更快、更省显存，细节变少。
  - 段间重叠（overlap）：相邻两段共用的帧数，默认 8，接缝处线性过渡（深度图先按缩放+偏移对齐上一段）。

## 效果和局限

RTX 4090（24 GB）上，默认参数（画布 1280×704，每段 41 帧，15 步）：

| 素材 | 做什么 | 总时间 | 每帧 | 显存峰值 |
|---|---|---|---|---|
| 面部特写 772×855，24 帧 | 拆全部 5 个通道 | 232 秒 | 9.7 秒 | 17.5 GB |
| 舞者 864×480，24 帧 | 拆全部 5 个通道 | 224 秒 | 9.4 秒 | 17.5 GB |
| 舞者，70 帧（两段，重叠 12 帧） | 拆全部 5 个通道 | 736 秒 | 10.5 秒 | 19.1 GB |
| 舞者，57 帧一段（网络临时挪到内存） | 拆全部 5 个通道 | 527 秒 | 9.3 秒 | 17.8 GB（内存多约 15 GB） |
| 面部特写，24 帧 | 重打光（含拆 G-buffer） | 280 秒 | 11.7 秒 | 17.5 GB |

- 时间里含每次加载 7B 模型约 17 秒（重打光要加载两个）。只拆一个通道大约是五个通道的 1/5。
- 官方说视频要约 27 GB 显存；这里在 24 GB 卡上默认把显存控制在 20 GB 以内：7B 网络用 bf16（14.2 GB），每段 41 帧时网络和视频编解码器可以同时放在显存里；选 57 帧一段时，编解码那几秒把网络挪到内存（和官方 --offload 选项一样）。
- 效果：拆 G-buffer 效果好——基础色把光影去得很干净，法线细节清楚，深度图能分出人和背景；段与段之间的接缝是 12 帧里慢慢过渡，没有跳帧。粗糙度、金属度对"看不懂"的区域（墙面、AI 生成的素材）会出现块状噪声。
- **重打光在真实实拍镜头上还不能用**：方向是对的（合成球体测试：光从右边来就亮右边、从镜头背后来就正面亮），官方示例图的前景（石膏像、人像、骰子）也能打出合理的新光，但上面两段实拍镜头的结果满是斑块和彩色噪点，背景常被画成环境图里的纹理。原因多半是真实镜头拆出来的粗糙度、金属度噪声太大，正向模型会把它们当成花哨的材质去"渲染"。把它当实验功能。

局限：

- **慢。** 7B 模型，每一帧、每个通道都要跑 15 步扩散。拆全部 5 个通道比重打光慢得多；只接了哪些输出口就只算哪些通道，只接「法线图」能省下 4/5 的时间。
- **深度图是相对的**：0 = 最近、1 = 最远，每段各自归一化后再拼接，不是真实尺度的深度图，不能当点云用（要真实尺度的深度图用 MoGe、Video Depth Anything）。
- 基础色是 sRGB 编码的（像贴图一样看是对的），拿去渲染前要转线性。
- 这是模型"猜"出来的材质：高光、阴影很重的地方，基础色里可能残留一点光影；粗糙度、金属度只能当参考。
- 重打光结果是模型"画"的，不是物理渲染：大方向（光从哪边来、冷暖、明暗）可控，细节（阴影形状、反射）不保证物理正确；面部可能有五官变化；画面里的背景、黑边会被当成"看得见的环境"，画上 HDRI 的内容。
- 分段处理：段内时序稳定，段与段之间靠重叠过渡，长镜头在接缝处可能有轻微跳变。
- 模型只在 1280×704 横幅上训练过：竖幅、方形画面整幅放进 16:9 画布、四周镜像填充来算；按竖幅尺寸直接算时重打光会变成满脸彩色噪点，所以统一用 16:9 画布。

## 团队

NVIDIA 多伦多 AI 实验室（负责人 Sanja Fidler，同时是多伦多大学教授），和多伦多大学、Vector Institute 合作，第一作者 Ruofan Liang、通讯作者 Zian Wang。论文是 CVPR 2025 口头报告。这个实验室做过很多和三维内容相关的有名工作：GET3D（从图片学出带贴图的三维模型）、Video LDM（早期的高清视频扩散模型），Lab2Shot 里的相机解算 ViPE 也出自这个实验室；合作者 Jacob Munkberg、Jon Hasselgren 是 nvdiffrec（可微渲染反求材质）的作者。Cosmos 是 NVIDIA 面向物理世界 AI 的视频基础模型平台，DiffusionRenderer 是在它上面训练的。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install diffusionrenderer`。会下载：
  - 锁定版本的官方代码（约 110 MB）；
  - 独立 Python 环境（PyTorch 2.8 + CUDA 12.8，约 7 GB）；
  - Hugging Face 上 NVIDIA 官方的模型（锁定版本并校验 sha256）：逆渲染 7B（28.9 GB）、正向渲染 7B（28.9 GB）、Cosmos 视频编解码器（约 0.2 GB）。合计约 58 GB，时间主要看网速。
- 不需要申请权限，不需要手动下载，装好后离线运行。
- 显卡：需要 NVIDIA 显卡，24 GB 显存（RTX 4090 / 3090）按默认参数可以跑；显存更小就把工作分辨率和每段帧数调小。内存建议 32 GB 以上（选 57 帧一段时要多 15 GB）。

## 许可证说明

**可以商用。**

- 代码（cosmos-transfer1-diffusion-renderer）：Apache-2.0。
- 两个 7B 模型和 Cosmos 编解码器：NVIDIA Open Model License。免费、可商用、可以改、可以分发；生成的结果归你，NVIDIA 不主张所有权。条件：
  - 分发模型本身时要附上许可证，并写明 "Licensed by NVIDIA Corporation under the NVIDIA Open Model License"；
  - 对外提供用到 Cosmos 模型的产品或服务时，要在网站、界面或文档里写 "Built on NVIDIA Cosmos"；
  - 绕过或削弱模型自带的安全限制、或者因为这个模型对别人提起专利 / 版权诉讼，许可自动终止（DiffusionRenderer 官方推理本身就不带安全过滤模型，Lab2Shot 按官方方式运行）；
  - 要遵守 NVIDIA 的可信 AI 条款。
- 官方安装说明里的 nvdiffrast 是 NVIDIA 非商用许可，Lab2Shot **没有安装它**：它原本只用来把 HDRI 投影成模型要的格式，这一步用 PyTorch 重写了。TransformerEngine、Megatron-Core 也没有安装，用等价的 PyTorch 代码代替。

## 参考

- 论文：https://arxiv.org/abs/2501.18590
- 项目主页：https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/
- 代码（Cosmos 版，Lab2Shot 用的这个）：https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer ，许可证 https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer/blob/main/LICENSE
- 论文原版（基于 Stable Video Diffusion 的学术版）：https://github.com/nv-tlabs/diffusion-renderer
- 模型卡：逆渲染 https://huggingface.co/nvidia/Diffusion_Renderer_Inverse_Cosmos_7B ，正向渲染 https://huggingface.co/nvidia/Diffusion_Renderer_Forward_Cosmos_7B ，编解码器 https://huggingface.co/nvidia/Cosmos-Tokenize1-CV8x8x8-720p
- NVIDIA Open Model License：https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
- NVIDIA 博客：https://blogs.nvidia.com/blog/cvpr-2025-ai-research-diffusionrenderer/ ，演示视频 https://www.youtube.com/watch?v=Q3xhYNbXM9c
- NVIDIA Cosmos：https://www.nvidia.com/en-us/ai/cosmos/
- 实验室主页：https://research.nvidia.com/labs/toronto-ai/ ，GET3D https://research.nvidia.com/labs/toronto-ai/GET3D/ ，Video LDM https://research.nvidia.com/labs/toronto-ai/VideoLDM/ ，nvdiffrec https://nvlabs.github.io/nvdiffrec/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
