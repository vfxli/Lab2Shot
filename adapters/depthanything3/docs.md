+++
team = "字节跳动 Seed 团队（ByteDance Seed）"
people = "Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng, Bingyi Kang"
paper = "Depth Anything 3: Recovering the Visual Space from Any Views（2025）"
paper_url = "https://arxiv.org/abs/2511.10647"
website = "https://depth-anything-3.github.io"
repo = "https://github.com/ByteDance-Seed/Depth-Anything-3"
year = 2025
+++

## 这是什么

上游 README 这样介绍它：Depth Anything 3（DA3）能从任意的视觉输入预测出空间上一致的几何，
相机位姿已知或未知都可以。作者说它追求最小建模，给出两条结论：一个普通的 plain transformer（例如
原版 DINO 编码器）作骨干就够，不需要为它做架构上的特化；单一的 depth-ray 表示省掉了复杂的多任务
学习。README 还写明：单目深度估计上明显超过 DA2，多视图深度估计和位姿估计上明显超过 VGGT，
全部模型只用公开学术数据训练；上游另外发布了做真实尺度单目深度估计的 DA3Metric-Large。

在 Lab2Shot 里，它接成两个节点：「Depth Anything 3 深度与相机」把整段画面一起看，同时给出各帧
一致的几何和每一帧的相机（位置、朝向、Focal Length）；「Depth Anything 3 深度图」按单张画面估深度图。
选 DA3Metric-Large 时结果是真实尺度。

## 输入输出

**官方要什么、给什么**

- 吃：一组画面，`x: (B, N, 3, H, W)`（`src/depth_anything_3/model/da3.py`）。
  另外两个**可选**的条件是已知相机：`extrinsics (B, N, 4, 4)` 和 `intrinsics (B, N, 3, 3)`。
  一张也行、一组也行：有没有已知相机位姿都能预测空间一致的几何。
- 给：一个 `Prediction`（`src/depth_anything_3/specs.py`）——
  `depth (N,H,W)`、`sky`、`conf`、`extrinsics (N,4,4)`、`intrinsics (N,3,3)`，
  外加 `is_metric`、`scale_factor`（是不是真实尺度、尺度系数）和可选的 `gaussians`。**没有三维点图这一项**。

**我们怎么接的**

- 「图像」口就是 `x`。逐帧那张卡（「Depth Anything 3 深度图」）只用它给的 `intrinsics`，不用它预测的外参；
  整段那张卡（「Depth Anything 3 深度与相机」）内外参都用。
- 「深度图」= `depth`、「天空遮罩」= `sky`、「置信度」= `conf`、「相机」= `intrinsics` 加 `extrinsics`。
- **上游的可选条件在节点上是参数，不是口**：`intrinsics` 这一路对应节点上的「已知 Focal Length」「Filmback」
  （worker 换算成上游要的视场角），所以节点上没有「相机」输入口——只用内参就接内参，不接整台相机。
  上游的 `extrinsics` 不给（没有这个用法）。
- **上游没有、节点上也没有**：遮罩和人物框输入口。
- 节点上没有「点云」口，因为上游本来就不出点图。上游的 `gaussians`（三维高斯分支）没有开，
  `is_metric` / `scale_factor` 在 worker 里用来决定尺度怎么标，没有单独开口。

出处：简介来自 `third_party/depthanything3/repo/README.md`；输入输出依据
`third_party/depthanything3/repo/src/depth_anything_3/model/da3.py`、`src/depth_anything_3/specs.py`
和 `adapters/depthanything3/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 两种用法：
  - 「Depth Anything 3 深度图」（逐帧几何）：每帧单独估计真实尺度的深度图、点云和 Focal Length，和 MoGe / UniDepth 同一类。
  - 「Depth Anything 3 深度与相机」：整段镜头一起算，输出每帧相机 + 深度图（和 VGGT、Pi3 同一套输出），长镜头自动分段、用重叠帧拼接。DA3 不出三维点图，所以没有「点云」口，要点云接「深度转点云」。
- 节点上没有遮罩 / 人物框输入口：DA3 的 forward 只吃画面和可选的相机参数。想只算画面的一部分，在送进去之前把其余部分涂黑：人物检测 → 人物框转遮罩 → 图像相乘 → 这个节点的「图像」口。
- 模型选择：Giant 1.1（默认，最好，非商用）、Large 1.1（更快，非商用）、Base（可商用，效果差一些）；逐帧几何另有 Metric-Large 单独使用（可商用，必须填视场角）。所有组合都用 Metric-Large 把深度图换算成米。
- 知道镜头就填"水平视场角"（逐帧几何）：决定真实尺度（深度值 = Focal Length × 网络输出 / 300），效果提升很大，见下面数字。它不能把 Focal Length 当作网络条件输入（相机编码器需要连同位姿一起给）。
- 逐帧几何不填视场角时，每帧估的 Focal Length 跳动很大，深度图会明显闪动；要稳定的整段深度图请用「Depth Anything 3 深度与相机」。
- "处理分辨率"：长边像素，默认 504（官方训练尺寸）；输出始终是原图大小。
- 「Depth Anything 3 深度与相机」手填「每段最多帧数」的上限是 **110 帧**（按最吃显存的 Giant 模型、24 GB 显卡定的：124 帧要 19.4 GB；Large / Base 能放更多但没有单独验证，统一用更保守的数字），服务器按同样的上限拒绝更大的数字。

## 效果和局限

RTX 4090 上（bf16）：

- 逐帧几何速度：Giant 约 0.09–0.13 秒/帧，显存峰值 12.8 GB（加载瞬间，推理时约 9 GB）；Large 约 0.08–0.11 秒/帧、4.6 GB；Metric-Large 单独约 0.02–0.04 秒/帧、2.6 GB。
- 逐帧几何，固定机位跳舞（864×480，参考 Focal Length 约 560–590px）：Giant Focal Length 中位数 590px（很准），但逐帧在 394–715px 之间跳，后墙深度闪动标准差 55 厘米（10%）；Large Focal Length 636px，闪动 28 厘米（4.6%）。Metric-Large + 已知视场角（Focal Length 586px）：舞者 2.6 米、后墙 5.7 米，闪动只有 8 厘米（1.5%）——知道镜头时这是最稳的逐帧方案，而且可商用。
- 逐帧几何，iPhone 长焦跟拍（1080×1920，参考 Focal Length 约 5484px）：Giant 自己估 Focal Length 3845px（3018–4423），人物 7.8 米、逐帧跳动 13%；填入真实视场角后人物约 11 米（针孔几何核算约 10–11 米）。
- 「Depth Anything 3 深度与相机」，固定机位（124 帧）：Giant Focal Length 501px（偏短约 13%），后墙 4.8 米，墙面闪动只有 0.8 厘米（0.17%），静止相机几乎不动（全段漂移 1 厘米、0.1°）；显存：124 帧一次算要 19.4 GB，所以默认一次最多约 110 帧，这段会分两段拼接（拼接后后墙起伏约 20 厘米、相机前后漂移 0.44 米，固定机位没有视差时分段拼接的尺度和前后位置分不清）。Large 124 帧一次算：Focal Length 639px，后墙 6.1 米，闪动 0.4 厘米，7 秒，11.6 GB。Base：Focal Length 721px，后墙 6.9 米，4.8 秒，8.4 GB。
- 「Depth Anything 3 深度与相机」，长焦跟拍（30 帧）：Giant Focal Length 4950px（偏短约 10%，逐帧 4925–4982 很稳），人物 11.1 米、逐帧跳动 0.4%，相机向前走约 0.7 米（跟拍，合理）；帧间深度一致性误差约 1%。30 帧 2.4 秒、12.8 GB。Large Focal Length 4545px；Base Focal Length 3029px、人物 6.9 米（明显偏近）。
- 「回环闭合」（默认关，和 VGGT、Pi3 共用一套）：分段拼接时一段接一段误差会积累；打开后找出隔得远却拍到同一处的两段，把这两处的画面放在一起再算一次，用它把各段拉回一致（和 VGGT-Long 的做法相同，但不需要另外的检索模型：按缩略图像不像、按目前的拼接结果互相看不看得见来找；放在一起算对不上的、优化后仍和别的对不上的会被丢掉）。临时文件放在缓存里、算完就删（792 帧约 1.3–2.2 GB）。固定机位用 Giant 切成 4 段：找到 3 处都用上，但前后漂移没有变好（最大移动 80 → 99 cm，转动 0.20° → 0.18°）：固定机位没有视差时每段内部的前后位置本来就定不准，回环只能调整段与段之间；真正绕一圈回到原处的镜头上没有验证过，所以默认关。
- 已知问题：逐帧 Focal Length 不稳；Giant 在 24 GB 显卡上一次最多约 110 帧（504 长边），更长的镜头靠分段拼接，固定机位的分段拼接会有尺度 / 前后漂移。

## 团队

字节跳动 Seed 团队（项目负责人 Bingyi Kang），Depth Anything 系列的作者：Depth Anything V1/V2（CVPR 2024 / NeurIPS 2024）是目前最常用的单图深度模型之一，Video Depth Anything（本项目已接入）是它的视频版。Depth Anything 3 用一个普通的视觉 Transformer 统一了单图深度、多视角深度和相机估计，官方报告在相机和几何上超过 VGGT。

## 模型下载和安装

- 运行 `lab2shot ext install depthanything3`：下载代码（锁定版本）、建独立 Python 环境（PyTorch 2.9，和 UniDepth、UniK3D 共用下载缓存），再下载 4 个模型（Hugging Face 固定版本，安装程序校验 sha256）：Giant 1.1（含 Metric-Large）6.8 GB、Large 1.1 1.6 GB、Base 0.54 GB、Metric-Large 1.3 GB，共约 10 GB。网速正常时 5–10 分钟。
- 没有下载 DA3-GIANT-1.1：它的权重文件和已弃用的旧版 DA3-GIANT 一模一样（官方重新训练的 Giant 1.1 只在 DA3NESTED-GIANT-LARGE-1.1 里）。DA3-SMALL、DA3MONO-LARGE（都是 Apache-2.0）也没有下载。
- 不需要申请权限，不需要自行下载任何文件。

## 许可证说明

- 代码：Apache-2.0，可商用。
- **Giant 1.1（DA3NESTED-GIANT-LARGE-1.1）：CC BY-NC 4.0，非商用。**
- **Large 1.1（DA3-LARGE-1.1）：按非商用对待。** 官方 README 的模型表写的是 CC BY-NC 4.0，但它的 Hugging Face 模型卡写着 Apache-2.0；同系列 DA3-LARGE 的模型卡在 2025-11-19 已经从 Apache-2.0 改成 CC BY-NC 4.0，所以按更严格的非商用对待。
- Base（DA3-BASE）和 Metric-Large（DA3METRIC-LARGE）：Apache-2.0，可商用；二者组合（Base 多视角重建 / 逐帧几何，或 Metric-Large + 已知视场角）的结果可以用于商业项目。
- 官方说明所有模型只用公开学术数据集训练。

## 参考

- 论文：https://arxiv.org/abs/2511.10647
- 项目主页：https://depth-anything-3.github.io
- 代码：https://github.com/ByteDance-Seed/Depth-Anything-3
- 许可证原文（代码）：https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/LICENSE
- 模型卡：https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1 、https://huggingface.co/depth-anything/DA3-LARGE-1.1 、https://huggingface.co/depth-anything/DA3-BASE 、https://huggingface.co/depth-anything/DA3METRIC-LARGE
- 在线演示：https://huggingface.co/spaces/depth-anything/Depth-Anything-3
