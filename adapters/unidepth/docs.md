+++
team = "苏黎世联邦理工学院（ETH Zürich）计算机视觉实验室 + INSAIT + 丰田欧洲（Toyota Motor Europe）"
people = "Luigi Piccinelli, Christos Sakaridis, Yung-Hsu Yang, Mattia Segu, Siyuan Li, Wim Abbeloos, Luc Van Gool"
paper = "UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler（2025）"
paper_url = "https://arxiv.org/abs/2502.20110"
website = "https://lpiccinelli-eth.github.io/pub/unidepth/"
repo = "https://github.com/lpiccinelli-eth/UniDepth"
year = 2025
+++

## 这是什么

UniDepthV2 只凭跨领域的单张画面，就还原出真实尺度的三维场景。与以往单目真实尺度深度估计的做法不同，它在推断时不需要任何额外条件，便预测出真实尺度的三维点。模型里有一个可自我提示的相机模块，预测出一份稠密的相机表示，用来给深度图分支的特征做条件；输出采用伪球面表示，把相机和深度图两部分的表示解开。V2 比 V1 多一路不确定度输出，供需要把握程度的下游使用。

在 Lab2Shot 里，这个节点叫「UniDepth 深度图」：逐帧各算各的，交出深度图、点云、置信度和它预测出的内参。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`model.infer(rgb_torch, camera)`——一张 RGB 画面张量，加一台**可选**的相机：
  `Pinhole(K=…)` 的内参矩阵（`scripts/demo.py:12-21`）。不给相机时网络自己预测
  （论文摘要：`directly predicts metric 3D points from the input image at inference time without any additional information`）。
  **它只收内参，不收任何一帧的外参。**
- **给**：`predictions` 字典（`unidepth/models/unidepthv2/unidepthv2.py:241-338`）——`depth`（真实尺度，米）、
  `points`（模型自己的三维点，:336）、`intrinsics`（预测出的内参，:330）和一路不确定度。

**我们怎么接的**

- 「RGB」= 上游那张画面，逐帧各算各的（它没有时序模型）。
- 「已知 Focal Length」「Filmback」= 上游那台**可选**相机：填了就按针孔造一台（主点在画面中心）喂进去、也照它返回；不填就让网络自己预测。
  **这两个是参数，不是一台相机**——上游只用内参，所以节点上没有「相机」输入口。
- 「深度图」= `depth`，「点云」= `points`，「置信度」= 那一路不确定度，「相机」输出 = `intrinsics`（**只有内参，没有外参**）。
- **不一样的一点**：上游没有天空 / 无效区域这一路输出，所以我们的有效区域就是「深度图为有限正值」的那些像素（`adapters/unidepth/worker.py:14-16`）。

**出处**：简介抽自论文摘要（arXiv 2502.20110：`We propose a new model, UniDepthV2, capable of reconstructing metric 3D scenes from solely single images across domains`；
`UniDepthV2 implements a self-promptable camera module predicting a dense camera representation to condition depth features`；
`an additional uncertainty-level output`）；
输入输出依据 `third_party/unidepth/repo/scripts/demo.py:12-21`、`repo/unidepth/models/unidepthv2/unidepthv2.py:241-338`
和 `adapters/unidepth/nodes.py` 的 `official`、`worker.py:6-18`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → UniDepth 深度图 → 深度图 / 位置图 / 点云，导进 Houdini / Nuke 做深度图合成、雾效、摆放参考白模；接上相机（ViPE 解算或读取的相机）时用相机的 Focal Length。
- 知道镜头就一定填「已知 Focal Length」「Filmback」：UniDepth 会把这个 Focal Length 当作条件输入网络（不是事后缩放），尺度明显更可信；长焦镜头它自己估的 Focal Length 偏短。
- "精度等级" 0–9（默认 9）：网络内部计算的分辨率（0.2–0.6 百万像素之间），降低更快、细节更少；输出始终是原图大小。
- 模型：ViT-L（默认，最好）、ViT-B、ViT-S（小而快，效果差一些）。
- 每帧单独计算，没有时序平滑，深度图会轻微闪动；要稳定的整段深度图用 Video Depth Anything 或 Depth Anything 3 深度与相机。
- 没有天空 / 无效区域检测：遮罩是所有像素，天空会被给一个"很远"的距离。也不输出法线。

## 效果和局限

RTX 4090 实测（ViT-L，精度等级 9，fp16）：

- 速度和显存：864×480 约 0.05 秒/帧，1080×1920 约 0.08 秒/帧；显存峰值约 3.2 GB，内存约 3.6 GB；加载模型约 5 秒。
- sh010（864×480 固定机位跳舞，参考 Focal Length 约 560–590px）：估计 Focal Length 624px（逐帧 614–632），偏长约 8%；舞者中位距离 2.8 米（2.4–3.3），合理；后墙 6.1 米，比预期的约 5 米远约 20%。静止后墙的逐帧闪动：墙面小块中位深度随时间的标准差约 6.5–7 厘米（1.1%），逐帧跳动 0.4%，整段最大起伏约 28 厘米。
- sh020（1080×1920 iPhone 长焦跟拍，参考 Focal Length 约 5484px）：自己估的 Focal Length 约 3975px（逐帧 3750–4250），偏短 27%；人物约 7.8 米。填入真实视场角（11.25°）后人物约 8.9 米。按针孔几何核算（人高约 1.6 米、画面里约 830 像素高、Focal Length 5484px），人物应在 10–11 米处（Depth Anything 3 「Depth Anything 3 深度与相机」也给出约 11 米），所以填 Focal Length 后更接近，但仍偏近。
- 输出的相机内参是从输出点云本身拟合出来的，和点云严格一致（重投影误差约 0.03 像素）；填了视场角时内参就是填的镜头。
- 已知问题：长焦镜头 Focal Length 被低估；远处墙面尺度偏大；逐帧有 1% 左右的尺度闪动。

## 团队

苏黎世联邦理工学院（ETH Zürich）Luc Van Gool 教授的计算机视觉实验室，一作 Luigi Piccinelli，与保加利亚 INSAIT 研究所和丰田欧洲（Toyota Motor Europe）合作。第一版 UniDepth 发表于 CVPR 2024，曾在 KITTI 深度榜单排名第一；同一作者还做了 UniK3D（任意镜头的单图三维，CVPR 2025）。Van Gool 是欧洲最资深的计算机视觉学者之一，经典的 SURF 特征点算法就是他参与提出的。

## 模型下载和安装

- 运行 `lab2shot ext install unidepth`：下载 UniDepth 代码（锁定版本）、建独立 Python 环境（PyTorch 2.9，约 7 GB，和 UniK3D、Depth Anything 3、Video Depth Anything 共用下载缓存，基本不额外占空间），再下载三个模型：ViT-L 1.4 GB、ViT-B 0.46 GB、ViT-S 0.14 GB（从 Hugging Face 固定版本下载并校验 sha256）。网速正常时十分钟左右。
- 不需要申请权限，不需要自行下载任何文件。

## 许可证说明

- **非商用**。代码是 CC BY-NC 4.0（仓库 LICENSE 和源码文件头都这么写）：可以研究、修改、分享，要署名，不能用于商业目的。
- 权重（Hugging Face 上的 unidepth-v2-vitl14 / vitb14 / vits14）的模型卡没有单独写许可证，按仓库的 CC BY-NC 4.0 对待，结果不能用于商业项目。
- 主干网络 DINOv2 的结构来自 Meta（Apache-2.0），权重已经包含在 UniDepth 的检查点里。

## 参考

- 论文 UniDepthV2：https://arxiv.org/abs/2502.20110
- 论文 UniDepth（CVPR 2024）：https://arxiv.org/abs/2403.18913
- 项目主页：https://lpiccinelli-eth.github.io/pub/unidepth/
- 代码：https://github.com/lpiccinelli-eth/UniDepth
- 许可证原文：https://github.com/lpiccinelli-eth/UniDepth/blob/main/LICENSE
- 模型卡（ViT-L）：https://huggingface.co/lpiccinelli/unidepth-v2-vitl14
