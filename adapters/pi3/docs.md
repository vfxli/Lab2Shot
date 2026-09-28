+++
team = "上海人工智能实验室（Shanghai AI Lab）+ 浙江大学 + 上海创智学院（SII）"
people = "Yifan Wang, Jianjun Zhou, Haoyi Zhu, …, Chunhua Shen, Tong He"
paper = "π³: Permutation-Equivariant Visual Geometry Learning（2025）"
paper_url = "https://arxiv.org/abs/2507.13347"
website = "https://yyfz.github.io/pi3/"
repo = "https://github.com/yyfz/Pi3"
year = 2025
+++

## 这是什么

π³ 是一个前馈网络，去掉了对固定基准视角的依赖：传统做法要指定一个基准帧，基准帧挑得不好就容易不稳甚至失败。它用完全置换等变的架构，从一组无序的画面预测仿射不变的相机位姿和尺度不变的逐帧点图，在相机位姿估计、单目和视频的深度估计、稠密点图估计上都取得了当时最好的成绩。新版 Pi3X 换成卷积输出头，点云更平滑；置信度改成预测连续的质量等级；支持可选注入相机位姿、内参和深度图；并支持近似的真实尺度。

在 Lab2Shot 里，用它给整段镜头一次解出每帧相机和深度图：镜头按「每段最多帧数」切成有重叠的段送进网络，段之间用相似变换拼接，官方自己的世界点图和置信度也一并交出来。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一个 `torch.Tensor`，形状 B × N × 3 × H × W，像素值在 `[0, 1]`（README「Model Input & Output」）。
  命令行入口 `example.py` / `example_mm.py` 的 `--data_path` 收一个画面文件夹或一个 .mp4，`--interval` 隔帧取样。
- **给**：一个 dict，四个键——`points` 世界点图（B × N × H × W × 3）、`local_points` 每帧相机空间的点图、
  `conf` 置信度（原始 logits，取 `sigmoid` 得 0–1）、`camera_poses` 相机到世界的 4 × 4 矩阵（OpenCV 约定）。
- Pi3X 另外**可以**注入条件：相机位姿、内参、深度图（命令行 `--conditions_path`）。

**我们怎么接的**

- 「RGB」口（3 通道图像）就是上游那个 B × N × 3 × H × W 张量：整段镜头按「每段最多帧数」切成有重叠的段送进去，段之间用相似变换拼接。
- 上游给的四样，四个口一样不少（`adapters/pi3/nodes.py`）：「深度图」= `local_points[..., 2]`，「点云」= 官方世界点图 `points`，
  「置信度」= `conf` 取 `sigmoid`，「相机」= `camera_poses`。
- **和官方不一样的两点**：① 上游不给 Focal Length，「相机」里那条逐帧 Focal Length 是从 `local_points` 用最小二乘反算的（`adapters/pi3/worker.py`）；
  ② Pi3X 的条件注入（相机位姿 / 内参 / 深度图）没有接，节点上没有这三个口。
- 模型本身不吃遮罩，所以节点上没有遮罩口。只算画面的一部分，在送进去之前把别处涂黑（「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→「RGB」口）。

**出处**：简介来自 `third_party/pi3/repo/README.md`（Overview 前两句和 Pi3X 那四条）；
输入输出依据同一份 README 和 `adapters/pi3/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → Pi3 深度与相机 → 和 ViPE 相机解算、VGGT、COLMAP 的结果对比，互相校验。「点云」口交的是官方自己的点图（`local_points`，官方的世界点图 `points` 就是它乘上相机位姿），不是从深度图反投影的。
- 两个权重：`pi3x`（默认，Pi3X，点云更平滑，尺度大致是米）和 `pi3`（原版 π³，尺度任意）。
- 适合：有视差的镜头、静止机位，长焦也能用（Focal Length 会偏小 10–15%，见下）。不适合：变焦、畸变大的素材。
- 关键参数：`max_frames` 一次送进网络的帧数（可选 50 / 100 / 150，默认 150，控制在 20 GB 以内），更长的镜头自动分段，用重叠帧对齐拼接，接缝处相机和深度图做渐变过渡；`step` 隔帧取样，长镜头用 2–4 省时间。
- **手填每段最多帧数的上限是 150 帧**：`max_frames` 一次性看完整段（不是流式模型），24 GB 显卡上 150 帧用 13.8 GB；服务器拒绝更大的数字。
- 模型本身不接受遮罩，节点上也没有遮罩 / 人物框输入口：人照样参与计算，结果里也不会把它们抹掉。想只算画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口，图上一眼看得见。

## 效果和局限

RTX 4090 上，和 ViPE 相机解算对比（相机轨迹先做相似变换对齐再算误差）：

| 素材 | Pi3X（默认） | 原版 Pi3 |
|---|---|---|
| iPhone 长焦跟拍，1080×1920，150 帧（真实 Focal Length 约 5484 px，ViPE 5133 px） | Focal Length 4814 px（偏小 12%）；轨迹和 ViPE 差 0.04 m / 4.8 m 行程（0.9%），朝向差最大 0.8°；一次算完 38 秒，显存峰值 13.8 GB | Focal Length 4723 px；轨迹差 0.06 m（1.2%），朝向差最大 1.4°；36 秒，13.2 GB |
| 固定机位跳舞，864×480，124 帧（真实 Focal Length 约 560–590 px，ViPE 1053 px） | Focal Length 691 px；相机基本不动：位移为场景深度的 0.4%，转动最大 0.12°；27 秒，12.8 GB | Focal Length 663 px；位移 1.0%，转动 0.29°；25 秒，11.9 GB |

- Pi3X 的"真实尺度"只是大致的：跟拍镜头上它给的相机行程约 2.0 m，ViPE 是 4.8 m，两者差了 2.4 倍，用之前要核对（比如拿已知尺寸的物体量一次）。
- 更长的镜头自动分段：相邻两段共用 1/4 的帧，用这些帧的相机朝向和点云把后一段对齐到前一段，接缝处相机、Focal Length、深度图做渐变。同一镜头故意切成 5 段时，轨迹误差从 0.9% 增大到 5.9%，朝向差最大 1.2°——能一次算完就别分段。
- Focal Length 逐帧估，镜头内会慢慢漂几个百分点（真实镜头没变焦）；要准 Focal Length 用 ViPE / COLMAP 或实拍参数。
- 「回环闭合」（默认关，和 VGGT、Pi3、Depth Anything 3 共用一套）：分段拼接时一段接一段误差会积累；打开后找出隔得远却拍到同一处的两段，把这两处的画面放在一起再算一次，用它把各段拉回一致（和 VGGT-Long 的做法相同，但不需要另外的检索模型：按缩略图像不像、按目前的拼接结果互相看不看得见来找；放在一起算对不上的、优化后仍和别的对不上的会被丢掉）。临时文件放在缓存里、算完就删（792 帧约 1.3–2.2 GB）。在没有回到原处的镜头上验证过：固定机位镜头切成 4 段，找到 3 处都用上，相机几乎没变（最大移动 13.4 → 14.6 cm，转动 0.21° → 0.15°）；长焦跟拍整段 792 帧（5 段），5 处候选只用上 1 处，和 ViPE 的位置差 4.3% → 4.8%（略差），朝向差中位 8.0° → 7.6°，多花 40% 时间。这类镜头上没有好处，所以默认关；真正绕一圈回到原处的实拍镜头还没有验证过。
- 用的是纯 PyTorch 版的 RoPE 位置编码（没有编译 CUDA 版），速度已经够用。

## 团队

上海人工智能实验室（Shanghai AI Lab）三维视觉团队主导，浙江大学沈春华教授组、上海创智学院参与，通讯作者何通（Tong He）。同一团队还做过 Aether（几何感知的世界模型）、DeepVerse 等三维 / 世界模型工作。π³ 发布后很快成为 VGGT 之后最常被对比的前馈重建模型之一，2025 年底又推出了改进版 Pi3X。

## 模型下载和安装

- 运行 `uv run lab2shot ext install pi3`：下载 π³ 代码（锁定版本）、建独立 Python 环境（PyTorch 2.10，约 7 GB，和 VGGT 共用下载缓存），再下载 Pi3X 权重（5.4 GB）和原版 Pi3 权重（3.8 GB）。不需要申请权限；网速正常时十来分钟，权重下载完会自动校验。

## 许可证说明

- **非商用，只能用于研究。** Pi3 和 Pi3X 的权重都是 CC BY-NC 4.0（官方写明"严格非商用"，因为训练数据的限制），再分发也必须保留非商用限制并署名。
- 代码主体是 BSD-3-Clause（可以商用），但其中位置编码文件 `pi3/models/layers/pos_embed.py` 来自 Naver 的 DUSt3R/CroCo，是 CC BY-NC-SA 4.0 非商用；DINOv2 部分是 Apache-2.0，PRoPE 部分是 MIT。
- 注意：原版 Pi3 的 Hugging Face 模型卡标签写的是 bsd-2-clause，但卡片正文写明商用须联系作者，以代码仓库 README 里的许可表（权重 CC BY-NC 4.0）为准。

## 参考

- 论文：https://arxiv.org/abs/2507.13347
- 项目主页：https://yyfz.github.io/pi3/
- 代码：https://github.com/yyfz/Pi3
- 代码许可证：https://github.com/yyfz/Pi3/blob/main/LICENSE
- 权重许可证 CC BY-NC 4.0：https://creativecommons.org/licenses/by-nc/4.0/
- Pi3X 模型卡：https://huggingface.co/yyfz233/Pi3X
- 原版 Pi3 模型卡：https://huggingface.co/yyfz233/Pi3
- 在线演示：https://huggingface.co/spaces/yyfz233/Pi3
- 同团队的 Aether（世界模型）：https://arxiv.org/abs/2503.18945
- 同团队的 DeepVerse：https://arxiv.org/abs/2506.01103
