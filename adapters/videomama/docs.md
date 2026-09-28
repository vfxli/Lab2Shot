+++
team = "韩国 KAIST 计算机视觉实验室（KAIST AI）+ 高丽大学（Korea University）+ Adobe Research"
people = "Sangbeom Lim, Seoung Wug Oh, Jiahui Huang, Heeji Yoon, Seungryong Kim, Joon-Young Lee"
paper = "VideoMaMa: Mask-Guided Video Matting via Generative Prior（CVPR 2026）"
paper_url = "https://arxiv.org/abs/2601.14255"
website = "https://cvlab-kaist.github.io/VideoMaMa/"
repo = "https://github.com/cvlab-kaist/VideoMaMa"
year = 2026
+++

## 这是什么

VideoMaMa 是遮罩引导的视频抠像框架，借的是视频生成模型的先验。依靠这份先验，它在各类视频上都表现稳定，抠像质量做到细微处。

在 Lab2Shot 里，这个节点叫「VideoMaMa 精细抠像」：接每帧的画面和一张粗遮罩，交出带软边的 alpha。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`inference_onestep_folder.py --image_root_path <画面文件夹> --mask_root_path <遮罩文件夹>`，
  也就是每帧一张画面加一张粗遮罩，外加 SVD 底模和 VideoMaMa 权重两条路径；
  `--keep_aspect_ratio` 保长宽比。**每一帧都要有配对的遮罩**：`load_image_sequence` 配不上就报错
  （`inference_onestep_folder.py`）。一次算一窗，`--num_frames` 默认 16 帧，处理分辨率 1024 × 576。
- **给**：每帧的 alpha（`inference_onestep_folder.py` 的 `generated_frames`），一步出图：
  只做一次去噪，用的是 Stable Video Diffusion 时空 UNet 微调出来的权重。

**我们怎么接的**

- 「RGB」= 上游的画面文件夹，「粗遮罩」= 上游的遮罩文件夹（每一帧都要有，缺的那帧按空遮罩算）；
  「处理分辨率」= 上游 1024 × 576 那个长边（按 64 的倍数取整）；「遮罩扩缩」是送进去之前对粗遮罩做的扩张 / 收缩。
- 「Alpha」= 上游的 alpha，保持浮点、双线性缩回画面尺寸（上游存成 8 位图）。
- **不一样的三点，都不改数值**（`adapters/videomama/worker.py`）：① 不加载 CLIP 画面编码器——上游本来就把它的嵌入换成 0；
  ② 画面分批过 VAE 编码器（上游一次全进）；③ 上游一次算一窗 16 帧，节点把镜头切成有重叠的窗、重叠处线性交叉淡化，
  显存只跟窗口长度走，不跟镜头长度走。

**出处**：简介来自论文摘要（arXiv 2601.14255，`VideoMaMa: Mask-Guided Video Matting via Generative Prior`：
`converts coarse segmentation masks into pixel accurate alpha mattes, by leveraging pretrained video diffusion models`）；
输入输出依据 `third_party/videomama/repo/README.md`、`repo/inference_onestep_folder.py`
和 `adapters/videomama/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → SAM 3 视频分割（跟满整段，每帧都有遮罩）→ 遮罩接到本节点的「粗遮罩」，画面接「RGB」→ 输出 Alpha → 序列图输出设置（EXR），进 Nuke 合成。
- 粗遮罩要每帧都有（没有遮罩的帧当作"这帧没人"）。遮罩可以很粗（多边形、方块都行），模型训练时就是这么练的。
- 适合：任何主体（人、动物、物体都行），有遮挡、进出画的镜头。不适合：遮罩本身严重跑偏的镜头；想要超过 1024 分辨率的极细发丝（官方默认按 1024×576 计算）。
- 关键参数：
  - 处理分辨率（resolution，默认 1024）：官方分辨率。1080p 素材按 1024 长边算再放大回原尺寸。上限是 **1024**（显存 12–15 GB）：官方没有验证过更大尺寸的效果，所以不能填更大的数字，服务器也按同样的上限拒绝。
  - 遮罩扩缩（erode_dilate，默认 0）：正数把遮罩扩大（遮罩把头发切掉了时用），负数缩小（遮罩把背景也包进来时用）。
  - 没有「首帧预热次数」参数（那是 MatAnyone 2 的）。

## 效果和局限

RTX 4090 上（半精度，默认参数，引导遮罩用 BiRefNet 抠像结果）：

- 舞者（864×480，124 帧，固定机位，按 896×512 计算）：约 0.34 秒/帧（整段 46 秒，含 4 秒加载），显存峰值 12.2 GB。
- 背影行走（1080×1920，60 帧，手持跟拍，按 576×1024 计算后放大回原尺寸）：约 0.51 秒/帧，显存峰值 14.8 GB。
- 长镜头：每次算 16 帧，相邻两段重叠 4 帧并交叉淡化，显存只和这 16 帧有关，和镜头长度无关；接缝处没有可见跳动（接缝附近的跳动 0.0295，其余 0.0291）。
- 稳定性（按光流对齐前后帧后，人物边缘 ±8 像素带里 alpha 的平均跳动，越小越稳）：舞者镜头上 BiRefNet 逐帧抠像 0.043 → VideoMaMa 0.030（少约 30%，三者最稳），MatAnyone 2 为 0.033；背影行走镜头上 0.043 → 0.032（MatAnyone 2 0.035）。静止背景里几乎是 0。
- 细节：边缘过渡最柔和，头部附近的半透明像素比 BiRefNet 多约 15%；1080p 素材是按 1024 长边算再放大的，发丝不如 MatAnyone 2 原尺寸计算那么利落。
- 局限：慢（比 MatAnyone 2 慢 3–11 倍）、显存要 12–15 GB；遮罩哪一帧错了，那一帧的 alpha 也跟着错；只输出 alpha，不输出前景颜色。「半精度」关掉时用 bfloat16（和官方一样）：float32 和 float16 结果几乎一样（最大差 0.03），但要 22 GB 显存，所以不提供。

## 团队

KAIST 计算机视觉实验室（Seungryong Kim 教授）、高丽大学和 Adobe Research（Joon-Young Lee、Seoung Wug Oh 是视频物体分割经典工作 STM 的作者）。同一篇论文还用本模型自动标注了 5 万多段真实视频，做成视频抠像数据集 MA-V，并训练出 SAM2-Matte（会抠 alpha 的 SAM 2）。模型只在合成数据上训练，靠视频生成模型的先验泛化到真实素材。

## 模型下载和安装

- 自动安装：`lab2shot ext install videomama`。下载固定版本的官方代码、独立 Python 环境（PyTorch 2.10 + diffusers，约 5 GB，多数和其他扩展包共用缓存）和两份权重（都校验 SHA-256）：
  - VideoMaMa 的视频 UNet（Hugging Face SammyLim/VideoMaMa，6.1 GB）；
  - Stable Video Diffusion XT 的 VAE（195 MB）和 Stability AI 许可证原文。
  6 GB 的 UNet 下载最慢：Hugging Face 单线程下载有时只有几百 KB/s，可能要一两个小时；中断了再运行一次同样的命令会接着下。
- 不需要申请权限（这两个 Hugging Face 仓库都不是 gated）。

## 许可证说明

**非商用。** 代码是 CC BY-NC 4.0：只能非商业使用，分享时要署名、注明许可证。权重（VideoMaMa 的 UNet 和 SVD 的 VAE）是 Stability AI Community License：研究和非商业使用免费；商用要先在 Stability AI 官网登记，并且公司年收入低于 100 万美元（超过要找 Stability AI 买企业授权）；再分发要附带许可证原文和 “This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved” 声明，并标注 “Powered by Stability AI”；不能用模型或它的输出去训练别的基础生成模型；要遵守 Stability AI 的可接受使用政策。因为代码是 CC BY-NC，整体只能非商用。

## 参考

- 论文：https://arxiv.org/abs/2601.14255
- 项目主页（效果视频）：https://cvlab-kaist.github.io/VideoMaMa/
- 代码：https://github.com/cvlab-kaist/VideoMaMa
- 代码许可证原文（CC BY-NC 4.0）：https://github.com/cvlab-kaist/VideoMaMa/blob/main/License.md
- 模型卡：https://huggingface.co/SammyLim/VideoMaMa
- 在线演示：https://huggingface.co/spaces/SammyLim/VideoMaMa
- 底座模型 Stable Video Diffusion XT：https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- Stability AI Community License 原文：https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE.md ，商用登记：https://stability.ai/community-license
- 实验室主页：https://cvlab.kaist.ac.kr/
