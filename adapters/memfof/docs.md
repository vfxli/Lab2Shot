+++
team = "莫斯科国立大学图形与媒体实验室（MSU Graphics & Media Lab）"
people = "Vladislav Bargatin, Egor Chistov, Alexander Yakovenko, Dmitriy Vatolin"
paper = "MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation（ICCV 2025 Highlight）"
paper_url = "https://arxiv.org/abs/2506.23151"
website = "https://msu-video-group.github.io/memfof"
repo = "https://github.com/msu-video-group/memfof"
year = 2025
+++

## 这是什么

上游自己的话：MEMFOF 是一个面向 Full HD 视频的**省显存光流**方法，兼顾高精度和低显存占用。论文摘要写明：它在多帧估计和显存占用之间找到了合适的折中，1080p 输入推理只要 2.09 GB 显存、训练 28.5 GB，因而能在原生 1080p 上训练，不必裁切或降采样；投稿时在 Spring 榜单排第一（1 像素外点率 3.289），Sintel（clean）端点误差 0.963，KITTI-2015 的 Fl-all 误差 2.94%。论文 ICCV 2025 Highlight。

在 Lab2Shot 里，它一次看三帧（上一帧、这一帧、下一帧），前后两个方向的光流一起算出来，交出合成里常说的**运动矢量**，和 Nuke 的 VectorGenerator 做的是同一件事。代码和权重都可以**商用**（BSD-3）。

## 输入输出

**官方要什么、给什么**

- 吃：**三帧一起**。`forward(images, iters=8, flow_gts=None, fmap_cache=...)`，
  `images` 是 `[B, 3, 3, H, W]`、值在 0–255
  （`third_party/memfof/repo/memfof/model.py`）。demo 就是这么攒的：
  逐帧读进来，凑够三帧送一次（`demo.py`）。
- 给：一个字典——`flow`（每次迭代一份，形状 `[B, 2, 2, H, W]`，
  第一个 2 是**后向和前向**两个方向）、`info`、`nf`（只在训练时）、`fmap_cache`。
  demo 取的是中间那一帧的前向运动 `output["flow"][-1][:, 1]`。

**我们怎么接的**

- 「RGB」口就是 `images`：worker 按上游的规矩三帧一组滑过整段，「处理分辨率」决定送进去的边长。
- 「运动矢量」口（4 通道）里装的是上游同一次算出的**前向和后向**两份矢量，不拆成两个口——
  Lab2Shot 的运动矢量数据类型本来就是一包两向。
- 「置信度」口来自上游的 `info` 头：SEA-RAFT 那一路输出的两支 Laplace 混合的 logits 和对数尺度，
  按它算成 0–1 的可信度（`worker_sdk/lab2shot_worker/optical_flow.py`），单位是模型自己的像素——
  换算在 Lab2Shot 这边，数据本身是上游的。上游的 `nf` 只在训练时有，没有开口。

出处：简介来自 `third_party/memfof/repo/README.md`（Overview）；
输入输出依据 `third_party/memfof/repo/memfof/model.py`、`demo.py`，
置信度的换算在 `worker_sdk/lab2shot_worker/optical_flow.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → MEMFOF 运动矢量 → 「序列图输出设置」（写 EXR）或「多层 EXR 输出设置」→「输出」。
  写出来的就是 Nuke 的 motion 通道：`forward.u` `forward.v`（到下一帧）、`backward.u` `backward.v`（到上一帧），
  单位像素 / 帧，u 向右、v 向上（Nuke 的坐标）。Nuke 里 Read 进来给 VectorBlur（uv 选 motion）、
  IDistort（UV 选 forward，源接 TimeOffset +1，就是把下一帧对齐到这一帧）。
- 「置信度」：模型对每个矢量有多大把握（0–1，大约是 1 / 预计误差的像素数），低的地方（遮挡边缘、出画、运动模糊）要小心。它是置信度类型，不是遮罩：视图里是热图，要当遮罩先接「置信度转遮罩」。
- 往下接通用节点（任何光流都能用）：
  - 「遮挡遮罩」：前后一致性检查，得到「遮挡」（这一帧下一帧会被挡住或出画的像素）和「新露出」两张遮罩；
  - 「运动矢量变形」：把下一帧 / 上一帧拉到这一帧上（IDistort），检查光流、推一帧遮罩或修补；
  - 「运动矢量转 ST-map」：一帧接一帧串回参考帧，在参考帧上画一次，用「STMap」贴到整段（SmartVector 的思路）。
    串得越长误差越大，几十帧以上用 AllTracker。
- `处理分辨率`：留空 = 原尺寸（推荐），可选 960、1920。4K 素材太慢或显存不够时选 1920，矢量会跟着放大回原尺寸。
  画面长边超过 1920 时，留空也会自动按 1920 封顶（更大的原生分辨率没有验证过）。

## 效果和局限

RTX 4090 上（「运动矢量变形」把下一帧拉回这一帧，在「遮挡遮罩」以外的像素上和这一帧比，误差是 0–255 的平均差；
不对齐时的差写在括号里）：

| 镜头 | 模型 | 处理分辨率 | 每帧 | 显存峰值 | 拉回误差 |
|---|---|---|---|---|---|
| 手持长焦 1080×1920，每帧约 15 px | MEMFOF | 原尺寸 | 0.48 s | 2.3 GB | 3.07（16.2） |
| | MEMFOF | 长边 960 | 0.15 s | 0.6 GB | 3.21 |
| | WAFT | 原尺寸 | 2.30 s | 14.3 GB | 3.10 |
| | WAFT | 长边 960（默认） | 0.33 s | 2.2 GB | 3.05 |
| 固定机位跳舞 864×480 | MEMFOF | 原尺寸 | 0.09 s | 0.5 GB | 0.44（0.74） |
| | WAFT | 原尺寸 | 0.24 s | 2.0 GB | 0.49（0.78） |
| 人脸 772×855 | MEMFOF | 原尺寸 | — | — | 1.31（3.00） |
| | WAFT | 原尺寸 | 0.41 s | 2.7 GB | 1.42（3.01） |

- 两个模型拉回以后都只剩 0.4–3 左右的差（主要是噪点、运动模糊和亮度变化），比不对齐小 2–5 倍；MEMFOF 和 WAFT 准确度相当，
  MEMFOF 原尺寸就快、显存小，WAFT 在它训练的 960 上最划算。
- 置信度有用：置信度最低的 20% 像素，拉回误差是其余像素的 2–13 倍（手持长焦镜头 MEMFOF 5.6 对 3.0，WAFT 7.0 对 2.6；
  跳舞镜头 MEMFOF 3.8 对 0.29）。WAFT 的置信度区分得更开。
- 前后一致性检查标出的遮挡：手持长焦镜头约 2–4%，跳舞镜头约 1.5–2%（跳舞的手脚边缘、出画的边）。
- 串起来传播（「运动矢量转 ST-map」，手持长焦镜头从第一帧往后 120 帧）：和 AllTracker 在同样的像素上比，60 帧时两者的 ST-map
  相差中位数约 5 px、100 帧以上约 10 px，贴过去的误差也比 AllTracker 大（14.5 对 11.3）；追得回参考帧的像素
  到 120 帧只剩 20%（AllTracker 35%）。几十帧以内可以用，更长用 AllTracker。

已知局限：
- 它是逐三帧算的，每一帧的矢量只描述到相邻一帧的运动；串很多帧（「运动矢量转 ST-map」）误差会累积。
- 帧号不连续（隔帧的序列）时，矢量是到序列里相邻那一帧的，节点会提醒。
- 画面边缘出画的部分、被挡住的部分没有真值，矢量是模型猜的：用「遮挡遮罩」或「置信度」把它们找出来。

## 团队

莫斯科国立大学图形与媒体实验室（Dmitriy Vatolin 组），长期做视频质量、视频处理和光流的评测与方法。
MEMFOF 基于 SEA-RAFT（Princeton），在 Spring 基准上提交时排名第一，重点是高分辨率训练和省显存。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install memfof`。锁定 GitHub 仓库的一个版本，独立的 PyTorch 环境（和其他扩展共用下载缓存），
  下载作者在 Hugging Face 上发布的 MEMFOF-Tartan-T-TSKH 权重（303 MB，作者推荐用于真实视频）。不需要申请权限。
- 不用 PTLFlow 里的 MEMFOF 权重：那一份是 PTLFlow 自己训练的，CC BY-NC-SA（非商用）。

## 许可证说明

代码 BSD-3-Clause，权重（Hugging Face 模型卡）同样 BSD-3：可以商用、修改和再分发，保留版权声明即可。

## 参考

- 论文：https://arxiv.org/abs/2506.23151
- 项目主页：https://msu-video-group.github.io/memfof
- 代码：https://github.com/msu-video-group/memfof
- 权重：https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
