+++
team = "南洋理工大学 S-Lab（S-Lab, Nanyang Technological University）+ 商汤新加坡（SenseTime Research, Singapore）"
people = "Peiqing Yang, Shangchen Zhou, Kai Hao, Qingyi Tao"
paper = "MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator（CVPR 2026 Highlight）"
paper_url = "https://arxiv.org/abs/2512.11782"
website = "https://pq-yang.github.io/projects/MatAnyone2/"
repo = "https://github.com/pq-yang/MatAnyone2"
year = 2026
+++

## 这是什么

上游自己的话：MatAnyone 2 是一个实用的人物视频抠像框架，避开分割那样生硬的边界，因而保住细节，在复杂的真实拍摄条件下也更稳。论文《Scaling Video Matting via a Learned Quality Evaluator》。

在 Lab2Shot 里，给它一段画面和**一帧**粗遮罩（SAM 3、BiRefNet 出的遮罩即可），它记住这个人，整段镜头跟着往下抠，交出 0–1 的 alpha：头发丝、衣服边、运动模糊处是半透明的，前后帧稳定，不会像逐帧抠像那样边缘闪烁。相当于把一个粗 roto 自动变成精修过的 key。

## 输入输出

**官方要什么、给什么**

- 吃：一段画面加**第一帧的一张遮罩**。官方 `inference_matanyone2.py` 收
  `-i/--input_path`（「Path of the input video or frame folder」）、
  `-m/--mask_path`（`Path of the first-frame segmentation mask`）、
  `-w/--warmup`（第一帧预热次数，默认 10）、`-e/--erode_kernel`、`-d/--dilate_kernel`（对输入遮罩的腐蚀膨胀）、
  `--max_size`，见 `third_party/matanyone/repo/inference_matanyone2.py`。
  第一帧把遮罩编码进去，其余帧只送画面。
- 给：`pha`——每帧一张 alpha。脚本里还写出一个 `com_np`，那是
  `image * pha + 绿幕 * (1 - pha)` 合成出来的对照画面，不是模型的输出。

**我们怎么接的**

- 「RGB」口 = `--input_path`，「首帧粗遮罩」口 = `--mask_path`（官方的输入），「首帧预热次数」= `--warmup`，
  「遮罩修补」对应 `--erode_kernel` / `--dilate_kernel`，「处理分辨率」= `--max_size`。
- 「Alpha」口就是 `pha`。合成出来的那一张不做成口——视图里叠加就能看，没必要多一份数据。
- 上游只用第一帧的遮罩；「首帧粗遮罩」口收的是整段序列，worker 取第一张有内容的遮罩送进去（其余帧不参与），
  这样上游接一张图、下游接一段序列都不用改线。

出处：简介来自论文摘要（arXiv 2512.11782，摘要原文：「we introduce a learned Matting Quality Evaluator (MQE) that assesses semantic and
boundary quality of alpha mattes without ground truth」）；输入输出依据 `third_party/matanyone/repo/inference_matanyone2.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → SAM 3 视频分割（或 BiRefNet 抠像）→ 遮罩接到本节点的「首帧粗遮罩」，画面接「RGB」（要指定首帧遮罩取哪一帧，中间接「FrameHold」：留空取首帧，填帧号就取那一帧，帧号保持不变，本节点从那一帧开始往后抠、往前的帧倒着算）→ 输出 Alpha → 序列图输出设置（EXR），进 Nuke 做合成，或进 Houdini / Maya 当 holdout。
- 只用一帧遮罩：自动取第一张"有内容"的遮罩所在帧作为起点，往后一路跟下去；起点之前的帧从起点倒着算。所以粗遮罩只要第一帧准就够了，后面帧的遮罩不看。
- 所以「首帧粗遮罩」接一帧的遮罩就行（只查尺寸，不要求每帧都有）：粗遮罩不必整段算。省时的接法是把「FrameHold」放在粗遮罩节点**前面**：读取序列 → FrameHold（留空 = 首帧，或填起点帧号）→ BiRefNet 抠像 / SAM 3 → 本节点「首帧粗遮罩」，「RGB」仍接整段读取序列——粗遮罩只算那一帧（离线核对：200 帧的镜头，粗遮罩节点的处理量是 1 帧，本节点照常 200 帧，提交前没有提示）。
- 适合：人（专门为人训练），头发、宽松衣服、运动模糊。不适合：非人物体（车、动物效果一般）、人完全出画后又进来（跟丢后不会自己找回来）、多人时只能当一个整体抠。
- 关键参数：
  - 处理分辨率（resolution，默认 1920）：画面长边超过它才缩小。1080p 素材保持原尺寸；4K 素材默认按 1080p 算。上限是 **1920**（显存约 7.5 GB，不随镜头长度增长）；更大的分辨率没有验证过，服务器会拒绝。
  - 遮罩修补（mask_close，默认 10 像素）：官方做法，先扩 10 像素再缩 10 像素，把粗遮罩里的小洞补上。遮罩本身很干净时可以设 0。
  - 首帧预热（warmup，默认 10）：在起点帧上多迭代几次先把第一帧的 alpha 修好再往后传，一般不用改。

## 效果和局限

RTX 4090 上（半精度，默认参数，引导遮罩用 BiRefNet 抠像结果）：

- 固定机位舞者（864×480，124 帧）：约 0.03 秒/帧（整段 4.9 秒，含加载），显存峰值 0.8 GB。
- 手持跟拍背影行走（1080×1920 原尺寸）：60 帧约 0.15 秒/帧，300 帧约 0.10 秒/帧，显存峰值都是 7.5 GB 左右——显存不随镜头长度增长（只记最近 5 个记忆帧）。
- 稳定性（按光流对齐前后帧后，人物边缘 ±8 像素带里 alpha 的平均跳动，越小越稳）：舞者镜头上 BiRefNet 逐帧抠像 0.043 → MatAnyone 2 0.033（少约 23%），VideoMaMa 0.030；行走镜头上 0.043 → 0.035，VideoMaMa 0.032。静止背景里都几乎是 0（没有杂点闪烁）。
- 细节：原尺寸计算，边缘最利落；发丝的半透明过渡保留得住，头部附近的半透明像素比 BiRefNet 少约 25%（边更"实"）。素材本身运动模糊重时，发丝细节有限。速度是三者里最快的（VideoMaMa 慢 3–11 倍）。
- 局限：只看第一帧遮罩，中途有人挡住又出现、或者起点帧遮罩有错，错误会一直带下去（这时换 VideoMaMa，它每帧都看遮罩）；运动模糊很重的脚偶尔被抠成一整块；只输出 alpha，不输出去溢色的前景颜色。

## 团队

南洋理工大学 S-Lab（Chen Change Loy 吕健勤教授的 MMLab@NTU）和商汤新加坡。同一团队做过 MatAnyone（CVPR 2025，本模型的第一版）、ProPainter（视频去物体补背景）、CodeFormer（老照片/面部修复）、Upscale-A-Video（视频超分）；模型结构来自 Cutie（视频物体跟踪）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install matanyone`。下载固定版本的官方代码、独立 Python 环境（PyTorch 2.10，约 5 GB，多数和其他扩展包共用缓存）和权重 `matanyone2.pth`（135 MB，作者 GitHub 发布页 v1.0.0，下载后校验 SHA-256）。网速正常时几分钟装完。
- 不需要申请权限，不需要额外手动下载。

## 许可证说明

**非商用。** 代码和权重都按 NTU S-Lab License 1.0 发布：只允许非商业用途的使用、修改和再分发，再分发要保留版权声明和免责声明，不能用作者或学校的名义做宣传。商业使用（包括给客户交付的项目）必须先联系作者团队另行授权（联系方式在 LICENSE 文件末尾）。

## 参考

- 论文：https://arxiv.org/abs/2512.11782
- 项目主页（效果视频）：https://pq-yang.github.io/projects/MatAnyone2/
- 代码：https://github.com/pq-yang/MatAnyone2
- 权重发布页：https://github.com/pq-yang/MatAnyone2/releases/tag/v1.0.0
- 许可证原文：https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE.txt
- 在线演示：https://huggingface.co/spaces/PeiqingYang/MatAnyone
- 第一版 MatAnyone（CVPR 2025）：https://github.com/pq-yang/MatAnyone ，论文 https://arxiv.org/abs/2501.14677
- 模型结构来源 Cutie：https://github.com/hkchengrex/Cutie
- 实验室主页：https://www.mmlab-ntu.com/
