+++
team = "Google DeepMind 与 Google（合作：德国宇航中心 DLR、Mila 魁北克人工智能研究所、慕尼黑工业大学）"
people = "Sebastian Jung, Artem Zholus, Martin Sundermeyer, Carl Doersch, …, Federico Tombari"
paper = "TAPNext++: What's Next for Tracking Any Point (TAP)?（CVPR 2026 Findings）"
paper_url = "https://arxiv.org/abs/2604.10582"
website = "https://tap-next-plus-plus.github.io"
repo = "https://github.com/google-deepmind/tapnet"
year = 2026
+++

## 这是什么

TAPNext 是 Google DeepMind 的 TAP 系列里最新、最强、最快也最简单的跟点模型：它把「跟住任意一个点」写成「预测下一个 token」，靠在网络里逐帧往前传递来跟点。TAPNext++ 是它微调出来的权重，稳定跟点的时长长 40 倍，能跟穿遮挡，重新检出的能力也强，是在 1024 帧的合成序列上微调的。

在 Lab2Shot 里，我们拿它做长距离的 2D 跟踪点：在画面上撒一片网格点，或者自己点几个点，它一帧一帧把这些点跟到镜头结束，每个点每帧给出位置和看得见还是被挡住。结果可以辅助匹配移动（给 3DE / SynthEyes 提供 2D 特征点）、平面跟踪（CornerPin、稳像）和 roto 跟形。上游从查询帧往后跟，我们把镜头倒过来再跟一遍，所以查询帧之前的帧也有轨迹。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一段视频加一组查询点（每个点写成帧号加坐标），见官方 colab `torch_tapnextpp_demo.ipynb`（README:41、169）；
  模型是在线的，一帧一帧往前推，固定大小的状态在帧之间传递（`tapnet/tapnext/tapnext_torch.py:241-300`）。
  输入尺寸 256 × 256（TAPNext++ 原版）或 512 × 512（微调版，README:168-170）。
- **给**：每个点每帧的位置（`tracks`）和可见性（`visible_logits`）。

**我们怎么接的**

- 「RGB」= 上游那段视频；「参考帧」「网格点数」「手动点」造出上游的查询点；**「遮罩」口不进模型**，它只决定网格点撒在哪。
- 「2D 跟踪点」= 上游的 `tracks`，可见性和把握程度装在同一份数据里（不另立一个口），「可见门槛」用它给的概率。
- 「模型」参数 = 上游那两个权重（256 / 512），「处理分辨率」= 送进去的方形边长（上游的预处理就是压成方形）。
- **不一样的一点**：上游从查询帧往后跟；我们把镜头倒过来再跟一遍，所以查询帧**之前**的帧也有轨迹（`adapters/tapnext/worker.py:11-16`）。

**出处**：简介抽自 `third_party/tapnext/repo/README.md:15`（「TAPNext is our latest, most capable, fastest, yet simplest tracker…」）
和 :17（「TAPNext++ is an improved TAPNext checkpoint that has a 40x longer stable tracking performance…」）；
输入输出依据同一份 README:41、:168-170、`repo/tapnet/tapnext/tapnext_torch.py:241-300` 和 `adapters/tapnext/nodes.py:12-18`、`worker.py:7-20`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → TAPNext++ 2D 跟踪点 → 2D 跟踪点输出设置（3DEqualizer 2D 点文件或 CSV），
  导进 3DE 相机解算，或者在 Nuke 里做 Tracker / CornerPin。模板「2D 跟踪点 · TAPNext++」就是这条链。
- 手动加点：显示这个节点，在 2D 视图里点要跟的位置（鼠标变十字），每点一次加一个点，
  名字依次是 user_01、user_02…，导出时排在网格点前面；点错了在参数「手动点」里删掉。
- 只想跟一个物体或一块表面：把遮罩（例如 SAM 3 的抠像结果）接到遮罩输入上，网格就只撒在遮罩里，
  而且网格铺满遮罩的外框，小物体也能分到足够多的点。
- 自己点的点可以在任意一帧，点之前的帧会倒着往回跟，整段镜头每帧都有位置。
- 关键参数：`网格点数`（每边几个点，默认 0 = 只跟手动点；选 20 就是 400 个点）；`参考帧`（网格撒在哪一帧）；
  `模型`（512：高分辨率版，默认，精度好；256：论文原版，更快）；
  `可见门槛`（留空用模型自己的遮挡判断；点被误判成挡住就调低。改它不用重新跑模型）。
- 素材：TAPNext++ 是逐帧往前推的（在线）跟踪器，显存不随镜头长度增长，几千帧的长镜头也能跑。
  它把每帧压成正方形再看，精度大约是画面宽 / 高的 1/256 量级：做 roto、贴片够用，
  精确匹配移动建议和 CoTracker3 对比着用。

## 效果和局限

RTX 4090 实测（grid 20 = 400 个点）：

- sh020（iPhone 长焦手持跟拍，1080x1920，1001–1120 共 120 帧）：模型 512 每帧跟点 47 ms、显存峰值 2.6 GB；
  模型 256 是 21 ms/帧、1.7 GB。可见比例 60%（跟拍镜头很多点出画，最后一帧还剩 35%）。
  楼面上的点跟着手持晃动一起走；逐帧拟合单应矩阵后的残差中位数 0.66 px（256 版 0.60 px，
  CoTracker3 离线 0.52 px），和 CoTracker3 的结果相差中位数 3.4 px（1080x1920 画面上）。
- sh010（固定机位跳舞，864x480，124 帧）：背景点全程可见，偏离起始位置中位数 0.38 px，
  每个背景点整段最大偏移的中位数 0.67 px（理想值 0）。整体可见比例 95%。
- sh020 300 帧：显存不变（2.6 GB），45 ms/帧，时间和帧数成正比；从中间一帧（1200）撒点时前后双向都能跟。
- 已知问题：坐标是在 256 格的网格上预测的，每个点相对你点的位置有零点几像素的固定偏差，
  Lab2Shot 已把整条轨迹平移到正好穿过你点的位置（不会在起始帧跳一帧）；
  跳舞人物的衣服这类会变形、会转过去的部位，轨迹比刚体表面乱；首次加载模型约 5–11 秒（2.5 GB 权重）。

## 团队

Google DeepMind 的 TAP（Tracking Any Point）团队，和 Google、德国宇航中心、Mila 等合作完成。
这个团队做了点跟踪领域的一系列基础工作：TAP-Vid 评测数据集、TAPIR、BootsTAP、RoboTAP、TAPVid-3D 和 TAPNext，
TAPNext++ 是其中最新的在线跟踪模型，在 1024 帧的长序列上训练。

## 模型下载和安装

- 自动安装：`lab2shot ext install tapnext`。会下载 Google DeepMind 的 tapnet 仓库（锁定版本）、
  独立的 PyTorch 环境（约 5 GB，和其他扩展共用下载缓存），以及两个权重：
  `tapnextpp_512.ckpt`（2.5 GB，默认）和 `tapnextpp_ckpt.pt`（2.5 GB，256 版）。
  权重放在 Google 的云存储上，速度波动很大（我们实测一个 2.5 GB 的文件快时约 4 分钟，慢时要一个多小时）；
  中断了再运行一次同样的命令会接着下载，下完会自动校验。
- 不需要申请权限，不需要自行下载任何东西。

## 许可证说明

代码和两个 TAPNext++ 权重都是 Apache-2.0：可以商用，可以修改和再分发，保留版权和许可证声明即可。
Lab2Shot 只用到仓库里的 PyTorch 推理代码，不装 JAX、TensorFlow 等训练依赖。
训练数据（Kubric、PointOdyssey）不随扩展分发，不影响使用。

## 参考

- 论文：https://arxiv.org/abs/2604.10582
- 项目主页：https://tap-next-plus-plus.github.io
- 代码：https://github.com/google-deepmind/tapnet （TAPNext++ 说明：https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapnextpp ）
- 官方 Colab 演示：https://colab.research.google.com/github/deepmind/tapnet/blob/main/colabs/torch_tapnextpp_demo.ipynb
- 前作 TAPNext：https://arxiv.org/abs/2504.05579 ，主页 https://tap-next.github.io/
- TAP 系列（TAPIR 等）：https://deepmind-tapir.github.io

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
