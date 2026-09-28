+++
team = "卡内基梅隆大学、北京大学、斯坦福大学"
people = "Bowei Zhang, Lei Ke, Adam W. Harley, Katerina Fragkiadaki"
paper = "TAPIP3D: Tracking Any Point in Persistent 3D Geometry（NeurIPS 2025）"
paper_url = "https://arxiv.org/abs/2504.14717"
website = "https://tapip3d.github.io/"
repo = "https://github.com/zbw001/TAPIP3D"
year = 2025
+++

## 这是什么

TAPIP3D 做的是单目 RGB 和 RGB-D 视频里的长时前馈 3D 点跟踪。它引入一种三维特征云表示，把画面特征抬进一个持久的世界坐标空间里，抵掉相机自身的运动，整段里的轨迹因此估得准。

在 Lab2Shot 里，我们拿它把画面上的点跟成三维轨迹：接一段画面、每帧的深度图和相机，跟出来的点在相机的世界里、按深度图的尺度，和相机、「深度转点云」的点云严丝合缝对得上。在 Houdini / Maya 里可以拿这些点当 locator 钉 CG 道具，这是 2D 跟踪点给不了的。可以**商用**（代码 Apache-2.0，权重 MIT）。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`inference.py` 收一个 .mp4，或一个 .npz——里面是 `video`，加可选的 `depths`、`intrinsics`、`extrinsics`、`query_point`
  （`inference.py:40-49`）。没给深度图时它自己去执行 MegaSAM 补一份（`inference.py:52-58`）；
  没给外参时才退回单位阵（`inference.py:68-70` 打印「No extrinsics provided, using identity matrix」）。
  推理函数 `inference(model, video, depths, intrinsics, extrinsics, query_point)`
  （`utils/inference_utils.py:108-118`）**每帧外参和内参都吃**。
- **给**：`coords`（每个查询点每帧的三维位置）和 `visibs`（可见性），连同 video / depths / intrinsics / extrinsics 一起存进 `.result.npz`
  （`inference.py:145-157`）；可见性上游按 0.9 留一个是非值。

**我们怎么接的**

- 「RGB」= `video`，「深度图」= `depths`（米），「相机」= `intrinsics` + `extrinsics`。
  这个「相机」口成立，是因为 worker 内参和每帧外参都读了——上游整台相机都用上了。
- 「参考帧」「网格点数」「手动点」一起造出上游的 `query_point`；**「遮罩」口不进模型**，它只圈定网格点撒在哪。
- 「3D 跟踪点」= `coords`；「可见门槛」用的是上游 `visibs` 的概率本身（上游只留是非值，我们把概率留着，好让艺术家自己定门槛）。
- **不一样的一点**：上游没给深度图时会自己执行 MegaSAM 补一份，我们不走那条路——深度图在节点图上由别的节点给，
  哪个模型算的、什么尺度，线上一眼看得见。

**出处**：简介抽自 `third_party/tapip3d/repo/README.md:28`（「TAPIP3D is a method for long-term feed-forward 3D point tracking in monocular RGB and RGB-D video sequences…」两句）；
输入输出依据 `repo/inference.py:40-70、114-157`、`repo/utils/inference_utils.py:108-118` 和 `adapters/tapip3d/nodes.py:14-29`、`worker.py:7-19`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 →「ViPE 相机解算」→「ViPE 深度图」（或任何深度图节点 + 任何相机节点）→ TAPIP3D 3D 跟踪点 →「合成场景」
  （3D 跟踪点；相机直接从解算节点接过去）→「USD 输出设置」→「输出」。模板「3D 跟踪点 · TAPIP3D」就是这样。
- 输入「深度图」和「相机」都必须接，而且要**对得上**：同一个解算节点出的深度图和相机最好（ViPE、Depth Anything 3、MapAnything……）；
  MoGe 这类逐帧深度图配它自己的相机（相机空间，适合固定机位）。深度图只知道远近（affine）时先接「深度对齐」。
- 要跟的点：网格点（「网格点数」，接遮罩就只撒在遮罩里，比如只跟旗子）和在 2D 视图里点的手动点（任何帧都行，前后都会跟）。
- 输出只有「3D 跟踪点」（上游只交出 coords / visibs）：点云，每个点每帧都在，带编号 `id`、速度 `v`（厘米/秒）、
  `visible`（1 看得见，0 被挡住）、起始帧上的画面颜色。
  Houdini 读 USD 就是带 id / v 的点（发射粒子、Copy to Points、钉道具），Maya 用 Alembic 输出设置。
  没有「2D 跟踪点」和「相机」口：要 2D 点用「TAPNext++ 2D 跟踪点」，相机直接从解算节点接。
- 「处理分辨率」：标准 384×512（论文评测用的，快）；精细 543×724（上游默认，更准，显存和时间约翻倍）。
  三维位置不受这个缩放影响。
- 「可见门槛」：模型给出看得见的概率；留空用上游的 0.9。

## 效果和局限

RTX 4090 实测：

- 速度和显存：sh020 1080×1920 的 200 帧、256 个点（ViPE 的深度图和相机，精细 543×724）：跟点 15.6 秒（每帧 0.08 秒），显存峰值 10.0 GB；
  sh010 24 帧 100 个点 2.9 GB。显存大致和帧数成正比，300 帧以内放得下。

已知局限：
- 结果的好坏取决于接进来的深度图和相机：深度图逐帧闪烁（单帧深度图模型）时，静止物体上的点也会跟着前后抖；用视频深度图或多帧解算的深度图更稳。
- 所有帧的特征都放在显卡上：显存和帧数成正比，整段一次跟，很长的镜头放不下（300 帧以内放得下）。
- 模型在 Kubric 合成数据上训练；真实画面里细长的物体（头发、线）跟不准。

## 团队

卡内基梅隆大学 Katerina Fragkiadaki 组，Bowei Zhang（北京大学）和 Lei Ke 共同一作，合作者包括 AllTracker 的
Adam W. Harley（斯坦福）。NeurIPS 2025。

## 模型下载和安装

- 自动安装：`lab2shot ext install tapip3d`。锁定 GitHub 仓库的一个版本，独立的 PyTorch 环境；安装时用环境里的 CUDA 13.2 编译器
  编译仓库自带的 pointops2（近邻查询的 CUDA 算子，几分钟）。下载 Hugging Face 上的 tapip3d_final.pth（309 MB，锁定版本）。
- 不装它自己从画面估深度和相机的 MegaSaM / MoGe / UniDepth 流程：深度和相机由 Lab2Shot 的节点接进来。

## 许可证说明

代码 Apache-2.0，权重（Hugging Face zbww/tapip3d 模型卡）MIT：可以商用。仓库里的 pointops2 是 MIT。
接进来的深度、相机节点有它们自己的许可证（比如 ViPE 非商用）。

## 参考

- 论文：https://arxiv.org/abs/2504.14717
- 项目主页：https://tapip3d.github.io/
- 代码：https://github.com/zbw001/TAPIP3D
- 权重：https://huggingface.co/zbww/tapip3d

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
