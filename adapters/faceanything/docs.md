+++
team = "慕尼黑工业大学 视觉计算与人工智能实验室（TUM Visual Computing & AI Lab）+ 华为诺亚方舟实验室伦敦（Huawei Noah's Ark Lab）"
people = "Umut Kocasari, Simon Giebenhain, Richard Shaw, Matthias Nießner"
paper = "Face Anything: 4D Face Reconstruction from Any Image Sequence（ECCV 2026）"
paper_url = "https://arxiv.org/abs/2604.19702"
website = "https://kocasariumut.github.io/FaceAnything/"
repo = "https://github.com/kocasariumut/FaceAnything"
year = 2026
+++

## 这是什么

上游 README 的「Overview」一节这样介绍它：一个统一的前馈模型，从任意的画面序列做高保真的 4D
面部还原与稠密的面部跟踪。核心想法叫规范面部点预测——给每个像素一个共享规范空间里的归一化面部
坐标；作者说这个表述把稠密的面部跟踪和动态还原化成了同一个规范还原问题，由此得到时间上连贯的
几何和可靠的对应关系。上游一次推理写出的原始结果是 depth、intrinsics、extrinsics、canonical、
conf、valid 这几样。

在 Lab2Shot 里，它是「FaceAnything 面部通道」，把上面那几样逐帧落成：深度图（模型自己的相对尺度，
不是真实厘米）；规范坐标（每个像素对应「标准脸」上的
哪一点，同一处在每一帧里都是同一个颜色，作用和 Houdini 的 rest 位置、Nuke 的 Pref 通道一样，
可以把贴图、纹身、伤疤钉在脸上跟着走）；面部遮罩（用 Robust Video Matting 抠出的整个人物前景，
深度图和规范坐标只在遮罩里面有值）；相机，以及每帧一片点云（颜色取自原画面）。
要每个像素在相机空间里的三维位置图，把深度图和相机接「深度转世界位置」。
代码和权重都是 CC BY-NC 4.0，**非商用**：只能用于研究和评估，不能用于商业制作。

## 输入输出

**官方要什么、给什么**

- 吃：一段面部画面。官方 `run_inference.py` 收 `--input`（图片目录或一个视频文件）、`--output`，
  可选 `--mask-dir`（前景遮罩目录）、`--remove-background`（自己抠一张前景遮罩出来）、
  `--process-res`（默认 504）、`--stride`、`--max-frames`。
  模型这一层是 `run_inference(model, frame_paths, mask_paths, process_res, use_ray_pose, monocular, ...)`
  （`src/faceanything/predict.py`）。
- 给：一个 `FacePrediction`——`depth (N,H,W)`、`intrinsics (N,3,3)`、
  `extrinsics (N,4,4)`、`images`、`canonical (N,H,W,3)`（规范面部坐标）、`conf`、`valid`（前景遮罩）。
  默认 `monocular=True`，也就是把预测的外参换成单位阵，每帧的点各自待在自己的相机里。

**我们怎么接的**

- 「图像」口就是 `frame_paths`，「处理分辨率」就是 `--process-res`，「处理方式」对应 `per_frame`。
- 「深度图」= `depth`、「规范坐标」= `canonical`、「面部遮罩」= `valid`、「置信度」= `conf`、
  「相机」= `intrinsics`（按官方默认 `monocular=True`，外参是单位阵，所以相机口只有内参有意义）。
- **不一样的一处**：上游的 `--mask-dir` 是可选输入，节点上没有这个口；
  官方 `--remove-background` 那一路用 Robust Video Matting 自己生成遮罩
  （`src/faceanything/background.py`），worker 照着做同一步。

出处：简介来自 `third_party/faceanything/repo/README.md`（Overview）；
输入输出依据 `third_party/faceanything/repo/run_inference.py`、`src/faceanything/predict.py`
和 `adapters/faceanything/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- **典型接法**：读取序列 → FaceAnything → 序列图输出设置（规范坐标、深度图、遮罩各接一个）。相机和点云可以接「USD 输出设置」拿到 Houdini 里看。内置模板「面部 UV 与深度图 · FaceAnything」就是这样接的。
- **什么素材效果好**：脸占画面大的特写。它只看得懂脸：手、麦克风、眼镜等挡在脸前的物体，算出来的坐标不可靠（论文里也这么写）；侧脸太厉害、遮挡太多时质量会下降。
- **关键参数**（每段帧数、处理分辨率只提供在 24 GB 显卡上验证过的几个选项，不能手填任意数字；服务器按同样的选项拒绝，防止绕过网页提交把显存挤爆）：
  - **处理方式**：「分段」几帧一起算，前后帧一致，视频用这个；「逐帧」每帧单独算，细节略多但会闪，适合单张照片，显存最省。
  - **每段帧数**：8 / 16（默认）/ 24 三档。24 GB 显卡上的显存：8 用 13.2 GB，16 约 15 GB，24 用 18.8 GB。
  - **处理分辨率**：504（模型训练尺寸），只有这一档——更高的没在 24 GB 显卡上验证过。
- **点云输出口**：深度图 + 相机经核心的「深度转点云」逻辑（`lab2shot/nodes/core/geometry.py` 的 `points_from_depth`，和「深度转点云」节点共用同一份代码）一次给出彩色点云，不用再手动接一个「深度转点云」节点。

## 效果和局限

RTX 4090（24 GB）上：

- **面部特写**（772×855、113 帧，默认参数：分段 16 帧、504）：一共 32 秒，其中加载模型 7 秒、背景抠像 4 秒、解算 21 秒（约 0.19 秒/帧）。
- **显存**（处理分辨率 504，不同的每段帧数）：逐帧 8.6 GB；分段 8 帧 13.2 GB；16 帧约 15 GB；24 帧 18.8 GB；40 帧 20.4 GB。PyTorch 占用的峰值到过 21 GB，24 GB 显卡基本占满，计算时别和其他大模型同时跑。显存不够时停下来，说清这张卡的显存，列出更小的「每段帧数」（逐帧时是「处理分辨率」）让你选，不会自己调低。

已知问题：

- **深度图没有真实尺度**，是模型自己的单位，不是米或厘米。点云在视图里乘了 100 显示，只能看形状，不能当尺寸量。
- 分段处理时，每段的深度图比例不完全一样。节点会用相邻两段重叠的 2 帧把比例对齐，面部特写素材上各段之间的比例差在 ±6% 以内。
- 模型每帧估出的 Focal Length 会飘（长焦尤其明显），节点统一用整段的中位数当一个镜头，避免看起来像在变焦。
- 遮罩是 Robust Video Matting 抠的人物前景，头发边缘只是大致准确；脖子、肩膀、头发上也会有深度和位置值，但这个模型是专门针对脸的，这些地方的值要谨慎使用。

## 团队

慕尼黑工业大学（TUM）Matthias Nießner 教授的视觉计算与人工智能实验室，加上华为诺亚方舟实验室伦敦的 Richard Shaw。Nießner 实验室做面部和三维重建很有名，代表作有实时面部跟踪 Face2Face 和室内三维扫描数据集 ScanNet。第二作者 Simon Giebenhain 做过神经参数化头部模型 NPHM、多视角动态头部重建 NeRSemble、MonoNPHM；高精度面部跟踪 Pix2NPHM 也是他的工作。

## 模型下载和安装

- **自动安装**：运行 `uv run lab2shot ext install faceanything`，会下载：
  - FaceAnything 模型（Hugging Face 上的 UmutKocasari/FaceAnything，一个约 15 GB 的 checkpoint.pt，不需要申请权限）；
  - Robust Video Matting 背景抠像的代码（约 5 MB）和 ResNet-50 权重（约 100 MB），都从 GitHub 下载；
  - 一个独立的 Python 环境（含 PyTorch，约 7.8 GB），纯 PyTorch，不用编译，不用 CUDA 工具包。
  - 主要时间花在那 15 GB 上：按 100 Mbps 宽带算约 20 分钟。中途断了重新运行安装会接着下。
- 以上都在本机离线运行，解算时不会再联网下载任何东西。

## 许可证说明

- **非商用**：FaceAnything 的代码和权重都是 **CC BY-NC 4.0**，只能用于研究和评估，**不能用于商业制作（包括商业项目里的镜头）**。署名：使用和分享时要注明出处。
- 模型主干 Depth Anything 3 的 DA3-GIANT 本身也是 CC BY-NC 4.0。
- 训练数据由 NeRSemble 多视角面部数据集加 FLAME 面部模型跟踪做出来。想商用，要分别联系作者谈授权。
- 原仓库的 pyproject.toml 里写着 Apache-2.0，但以仓库 LICENSE 文件和 Hugging Face 权重页上的 CC BY-NC 4.0 为准。
- 背景抠像用的 Robust Video Matting 是 GPL-3.0：可以商用，但修改后再分发要同样用 GPL-3.0 开源。它在扩展包自己的独立进程里运行，不影响 Lab2Shot 本身。

## 参考

- 论文：https://arxiv.org/abs/2604.19702
- 项目主页：https://kocasariumut.github.io/FaceAnything/
- 代码仓库：https://github.com/kocasariumut/FaceAnything
- 许可证原文：https://github.com/kocasariumut/FaceAnything/blob/main/LICENSE
- 模型权重：https://huggingface.co/UmutKocasari/FaceAnything
- 在线演示：https://huggingface.co/spaces/UmutKocasari/FaceAnything
- 演示视频：https://www.youtube.com/watch?v=wSGHpAscp0Y
- Nießner 实验室：https://niessnerlab.org/
- Robust Video Matting（背景抠像）：https://github.com/PeterL1n/RobustVideoMatting
- Depth Anything 3（模型主干）：https://github.com/ByteDance-Seed/Depth-Anything-3
