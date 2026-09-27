+++
team = "字节跳动（ByteDance）Depth Anything 团队"
people = "Sili Chen, Hengkai Guo, Shengnan Zhu, Feihu Zhang, Zilong Huang, Jiashi Feng, Bingyi Kang"
paper = "Video Depth Anything: Consistent Depth Estimation for Super-Long Videos（CVPR 2025 Highlight）"
paper_url = "https://arxiv.org/abs/2501.12375"
website = "https://videodepthanything.github.io"
repo = "https://github.com/DepthAnything/Video-Depth-Anything"
year = 2025
+++

## 这是什么

Video Depth Anything 建在 Depth Anything V2 之上，任意长的视频都能算，质量、一致性和泛化能力都不打折扣。与其他基于扩散模型的做法相比，它推断更快、参数更少，深度一致性上的精度更高。

在 Lab2Shot 里，这个节点叫「Video Depth Anything 深度图」：接一段序列，每帧交出一张深度图。真实尺度的那一版走「深度图」口，相对的那一版走「视差图」口。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`run.py --input_video`，一段视频（`run.py:24`）。`--resolution` 网络输入边长（默认 518）、`--max_res` 上限 1280、
  `--encoder` 选 `vits` / `vitb` / `vitl`、`--metric` 换成真实尺度那一版、`--max_len` / `--target_fps` 限长和抽帧、
  `--fp32` 用单精度（默认是半精度）（`run.py:26-32`）。
- **给**：每帧一张深度图（`run.py:49-57` 的 `depths`），可存成 npz 或 EXR（`--save_npz` / `--save_exr`，`run.py:34-35`）。
  离线那条路按 32 帧一窗、窗之间重叠 10 帧、其中 2 帧作为关键帧对齐尺度和偏移。
  **上游只有 `depths` 这一样输出**：选真实尺度的权重时它是米，选相对的权重时它是相对视差。

**我们怎么接的**

- 「图像」= 上游那段素材；「模型」参数 = 上游的 `--encoder` 加 `--metric`（Small 可商用，Base / Large 和真实尺度版非商用）；
  「处理分辨率」= `--resolution`；「半精度」= 上游默认的 float16（它的 `--fp32` 是反过来那个开关）。
- 「深度图」和「视差图」**是同一样官方数据的两种情形**，不是我们多加的第二种结果：真实尺度那一版走「深度图」口（米），
  相对那一版走「视差图」口，另一个口空着（`adapters/videodepthanything/nodes.py` 的 `official`）。
- **不一样的一点**：上游的 `infer_video_depth` 把每一帧和每张全分辨率深度图都留在内存里，
  我们把同样的窗口流程改成流式，任意长的镜头只留大约 40 帧——算法一步不改，用的是它自己的常数和函数
  （`adapters/videodepthanything/worker.py:6-14`）。

**出处**：简介抽自 `third_party/videodepthanything/repo/README.md:19`（「This work presents Video Depth Anything based on Depth Anything V2, which can be applied to arbitrarily long videos without compromising quality, consistency, or generalization ability…」两句）；
输入输出依据同一份 README、`repo/run.py:24-57` 和 `adapters/videodepthanything/nodes.py` 的 `official`、`worker.py:1-16`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 →「Video Depth Anything 深度图」→ 序列图输出设置写 EXR（Z 通道），拿到 Nuke 里做景深、雾、深度图抠像，或者给 Houdini 当投影 / 置换参考。真实尺度的深度图已经换算成厘米（项目单位）。
- 选模型（「模型」参数）：
  - **真实尺度 Small / 相对 Small**：可商用，最快，默认用真实尺度 Small。
  - **真实尺度 Base / Large、相对 Base / Large**：**非商用**（仅限研究），模型更大，细节和远近层次一般更好，但更慢、更吃显存，稳定性不一定更好（见下面实测）。
  - 要放进三维场景、和相机对齐 → 选真实尺度；只要合成里的深度图效果 → 相对就够。
- 素材：普通焦段、画面清楚的实拍效果最好；画幅比 16:9 更宽的素材会被自动缩小处理（官方的显存限制）。它不解算相机，真实尺度是网络"估"的，不同模型之间可以差 20–40%，要精确尺度请用 ViPE 的相机 + 深度图或实测距离校正。
- 关键参数：「处理分辨率」默认 518（训练尺寸，最稳）。调大细节更多，但显存涨得很快：**Large 在 24 GB 显卡上最多约 756**，Base 可以到 1036，Small 到 1036 约 11 GB。

## 效果和局限

RTX 4090 实测，素材 sh010（864×480 固定机位跳舞镜头，124 帧），处理分辨率 518、半精度；时间含加载模型。稳定性 = 5 块没被人挡住的墙面 / 地面区域，"区域中值随时间的标准差"和"相邻帧平均变化"，越小越稳：

| 模型 | 许可 | 总时间（每帧） | 显存峰值 | 静止背景 标准差 / 相邻帧变化 |
|---|---|---|---|---|
| 相对 Small | Apache-2.0 | 2.8 s（0.023 s） | 2.8 GB | 1.77% / 0.19% |
| 真实尺度 Small | Apache-2.0 | 3.4 s（0.028 s） | 2.8 GB | 1.15% / 0.25% |
| 相对 Base | 非商用 | 5.3 s（0.043 s） | 5.2 GB | 1.36% / 0.15% |
| 真实尺度 Base | 非商用 | 4.7 s（0.038 s） | 5.2 GB | 2.86% / 0.23% |
| 相对 Large | 非商用 | 11.2 s（0.091 s） | 10.6 GB（占用 12.0 GB） | 1.54% / 0.14% |
| 真实尺度 Large | 非商用 | 11.6 s（0.094 s） | 10.6 GB（占用 12.0 GB） | 1.79% / 0.20% |

- 大模型的逐帧抖动（相邻帧变化）略小，但整段的慢漂移不比 Small 小；这条镜头上真实尺度 Small 反而最稳，真实尺度 Base 最差。
- 真实尺度各模型不一致：同一面后墙，真实尺度 Small / Base / Large 分别估 5.7 / 6.7 / 7.9 米，舞者 2.9 / 3.4 / 3.5 米。
- 处理分辨率和显存（同一素材）：Base 1036 → 19.2 GB（占用 23.8 GB），0.57 s/帧，能跑；Large 756 → 20.2 GB（占用 22.9 GB），0.23 s/帧，能跑；Large 840 → 24.5 GB 超出显存，溢出到内存后变成 1.25 s/帧；Large 1036 → 36 GB，8.6 s/帧（慢 90 倍），不要用。
- 已知问题：相对深度图每条镜头的尺度和偏移都不同，不能跨镜头比较；真实尺度只是估计；天空、玻璃、镜面一般不准；不输出相机。

## 团队

字节跳动的 Depth Anything 团队（通讯作者 Bingyi Kang、Hengkai Guo）。他们做的 Depth Anything（CVPR 2024）和 Depth Anything V2 是目前最常用的单图深度模型，很多 DCC 插件和 ComfyUI 节点都在用；Video Depth Anything 是它的视频版，后来的 Depth Anything 3 也出自同一系列。

## 模型下载和安装

- 运行 `lab2shot ext install videodepthanything`：下载代码（锁定版本）、建独立 Python 环境（PyTorch 2.9 + CUDA 12.8，装好约 7 GB，和其他扩展共用下载缓存），再下载 6 个模型，都锁定了 Hugging Face 版本并校验 sha256：Small 和米制 Small 各 116 MB，Base 和米制 Base 各 458 MB，Large 和米制 Large 各 1.54 GB，合计约 4.2 GB。网速正常（十几 MB/s）时 10 分钟左右。
- 不需要申请权限；已经下好的文件再次安装会跳过。运行时完全离线。

## 许可证说明

- 代码：Apache-2.0，可以商用。
- Video-Depth-Anything-Small、Metric-Video-Depth-Anything-Small：Apache-2.0，**可以商用**。
- Video-Depth-Anything-Base / Large、Metric-Video-Depth-Anything-Base / Large：CC-BY-NC-4.0，**非商用**，只能用于研究、学习，用它们算出的结果不能用于商业项目；使用时要注明出处。商业合作官方让发邮件联系。
- 米制模型的训练数据含 Virtual KITTI 2（CC BY-NC-SA 3.0）和 IRS，米制 Small 模型本身按 Apache-2.0 发布，但商用前请自己评估训练数据的风险。

## 参考

- 论文：https://arxiv.org/abs/2501.12375
- 项目主页：https://videodepthanything.github.io
- 代码：https://github.com/DepthAnything/Video-Depth-Anything
- 许可证原文（代码）：https://github.com/DepthAnything/Video-Depth-Anything/blob/main/LICENSE
- 模型卡：https://huggingface.co/depth-anything/Video-Depth-Anything-Small 、https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small 、https://huggingface.co/depth-anything/Video-Depth-Anything-Base 、https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base 、https://huggingface.co/depth-anything/Video-Depth-Anything-Large 、https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large
- CC-BY-NC-4.0 原文：https://creativecommons.org/licenses/by-nc/4.0/
- 在线演示：https://huggingface.co/spaces/depth-anything/Video-Depth-Anything
- 前作 Depth Anything V2：https://depth-anything-v2.github.io
