+++
team = "微软研究院（Microsoft Research）"
people = "Lingyu Kong, Ruicheng Li, Ruicheng Wang, Sicheng Xu, Chengtang Yao, Jianfeng Xiang, Jiaolong Yang"
paper = "MoGe-3: Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement（2026）"
paper_url = "https://arxiv.org/abs/2607.17967"
website = "https://qft-333.github.io/moge3page/"
repo = "https://github.com/microsoft/MoGe"
year = 2026
+++

## 这是什么

上游自己的话：MoGe 是一个用来从单目开放域画面恢复三维几何的强力模型，恢复出来的内容包括真实尺度的点图、真实尺度的深度图、法线图和相机视场角。官方列出的特点还有：所有结果由一个模型、一次前馈同时给出；可以把已知的真实视场角作为可选输入，进一步提高精度；支持从 2:1 到 1:2 的各种分辨率和画幅比。

在 Lab2Shot 里我们用的是最新的 MoGe-3，细小结构和物体边缘比前两代清楚。它交出的深度图是真实尺度、单位厘米，点云是模型自己的点图（不是我们拿深度图反投的），另外还有法线图和视场角，相当于给每一帧做了一个深度图 pass、法线 pass 和 P pass。

## 输入输出

**官方要什么、给什么**

- 吃：**一张**画面。命令行是 `moge infer -i 画面或文件夹 --version v3 --o 输出文件夹 --maps --glb --ply`
  （`third_party/moge/repo/README.md:233-256`）。模型这一层是
  `infer(image, num_tokens, resolution_level, force_projection, apply_mask, fov_x, refine_steps, use_fp16)`
  （`moge/model/v3.py:220-245`）：`image` 是 `(B,3,H,W)` 或 `(3,H,W)`；
  `fov_x` 是**可选**的已知水平视场角（度），不给就自己从点图推（`v3.py:241`）。
- 给：一个字典（`v3.py:246-255`）——`points`（相机空间点图 `(H,W,3)`）、`intrinsics`（内参）、
  `depth`（深度图）、`mask`（有效像素）、`normal`（法线图）。

**我们怎么接的**

- 「RGB」口就是 `image`，「精度等级」= `resolution_level`，「模型」= `--version`。
- 「深度图」= `depth`、「法线图」= `normal`、「点云」= 模型自己的 `points`（不是我们拿深度图反投的）、
  「相机」= `intrinsics`（**只有内参**，MoGe 不出外参）。
- **上游可选的那一项在我们这儿是参数**：`fov_x` 对应节点上的「已知 Focal Length」「Filmback」
  （worker 把 Focal Length（mm）换成视场角送进去），所以节点上没有「相机」输入口——上游只吃一个视场角，
  接的就是那个数，不接整台相机。
- 上游的 `mask`（有效像素）没有单独一个口：它在家族里当深度图的有效位用（天空、无定义的地方就是无效）。

出处：简介抽自 `third_party/moge/repo/README.md:3`（MoGe is a powerful model… 那一句）；
输入输出依据 `third_party/moge/repo/README.md:233-256` 和 `moge/model/v3.py:220-255`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → **MoGe 深度图与法线** → 深度图 / 法线图接「序列图输出设置」写 EXR 进 Nuke（做景深、雾、重打光的辅助 pass）；点云接「USD 输出设置」进 Houdini / Maya 当场景参考、摆资产、对位。模板「深度图与法线 · MoGe」就是这套接法。
- 知道镜头就填「已知 Focal Length」「Filmback」（也可以把别的节点给的 Focal Length 接到这个参数上）：按这个镜头算深度图；不填时 Focal Length 由模型自己估。输出的相机都是"不动、在原点"。
- 什么素材好：普通焦段、有明确前后景的镜头（室内、街景、人物中近景）。不好：长焦（模型容易把视角估得太宽）、大片天空和纯色墙面（没有远近变化，遮罩里会被排除）、镜面和玻璃。
- 关键参数：
  - Focal Length / Filmback：知道镜头就一定填（worker 换成水平视场角送进模型）。填对了深度图和尺度都会稳很多。
  - 模型：默认「标准 ViT-L」够用。「大模型 ViT-G」慢约 2 倍、显存约 7 GB，我们实测深度图并不更稳，只在细节不够时试试。
  - 点云间隔：默认每 4 个像素取一个点（1/16 的像素）；设成 1 每帧就是整张图的点，1080p 约 200 万点，文件会很大。

## 效果和局限

RTX 4090 实测（标准 ViT-L，精度等级 9）：

- sh020 手机长焦跟拍（1080×1920，12 帧）：约 0.15 秒/帧，显存峰值约 2.6 GB；加载模型约 4–5 秒，第一帧多花约 1 秒（编译加速程序，之后有缓存）。
- 视场角估计：同一个 sh020（iPhone 长焦，按拍摄记录约 5350–5600 Focal Length（px），ViPE 解算 5484 像素），MoGe 自己估的 Focal Length 只有约 2700 像素，差不多把视角估宽了一倍。**长焦镜头必须填 Focal Length。**
- 局限：
  - 每帧单独计算，没有前后帧的约束，深度图会轻微闪动。要时序稳定的深度图序列用「Video Depth Anything 深度图」。
  - "真实尺度"是模型从画面猜出来的尺度，不是测量值，和实拍尺寸会有误差；关键镜头请用实测距离（或已知物体尺寸）校一次。
  - 深度图、点云的单位都是厘米；法线在相机空间（+Z 轴朝向镜头，和 USD / Maya / Houdini 的相机习惯一致）；天空等无效区域在深度图 / 法线图的有效位（valid 通道）里是 0，值也写成 0。

## 团队

微软研究院（Microsoft Research）Jiaolong Yang、Xin Tong 等研究员带的团队，和中国科学技术大学、清华大学的合作者一起做的。MoGe 系列已经出了三代：MoGe-1 是 CVPR 2025 口头报告论文，MoGe-2 发表在 NeurIPS 2025（加入了真实尺度和法线），MoGe-3 在 2026 年发布。同一批作者还做了 TRELLIS（CVPR 2025，从图片或文字生成三维资产）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install moge`。会下载：
  - 锁定版本的官方代码（约 25 MB）；
  - 独立 Python 环境（PyTorch 2.13 + CUDA 13，约 5 GB）；
  - 两个 MoGe-3 权重（Hugging Face，锁定版本）：`moge-3-vitl`（3.7 亿参数，约 1.5 GB，默认用它）和 `moge-3-vitg`（12.5 亿参数，约 5 GB）。
  - 合计约 11–12 GB，时间主要看网速。
- 不需要申请权限，不需要额外手动下载。装好后完全离线运行。

## 许可证说明

**可以商用。** 代码是 MIT，两个 MoGe-3 权重的模型卡也标 MIT；里面用到的 DINOv2 主干网络是 Meta 的 Apache-2.0。依赖的 utils3d-moge、FlexGEMM 也是 MIT。utils3d-moge 里有一个可选模块会用到 NVIDIA 的 nvdiffrast（非商用），MoGe 推理用不到，本扩展包也不安装它。MIT 只要求再分发代码时保留版权声明。

## 参考

- MoGe-3 论文：https://arxiv.org/abs/2607.17967
- MoGe-3 项目主页（效果视频）：https://qft-333.github.io/moge3page/
- 代码：https://github.com/microsoft/MoGe
- 许可证原文：https://github.com/microsoft/MoGe/blob/main/LICENSE
- 模型卡：https://huggingface.co/Ruicheng/moge-3-vitl 、https://huggingface.co/Ruicheng/moge-3-vitg
- 在线演示：https://huggingface.co/spaces/Ruicheng/MoGe-3
- MoGe-2（NeurIPS 2025）：论文 https://arxiv.org/abs/2507.02546 ，主页 https://wangrc.site/MoGe2Page/
- MoGe-1（CVPR 2025 口头报告）：论文 https://arxiv.org/abs/2410.19115 ，主页 https://wangrc.site/MoGePage/
- 同团队的 TRELLIS：https://github.com/microsoft/TRELLIS

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
