+++
team = "Robbyant（蚂蚁集团旗下具身智能公司）"
people = "Lin-Zhuo Chen, Jian Gao, Shangzhan Zhang, …, Yujun Shen, Yao Yao, Yinghao Xu"
paper = "LingBot-Map: Geometric Context Transformer for Streaming 3D Reconstruction（ECCV 2026 Oral）"
paper_url = "https://arxiv.org/abs/2604.14141"
website = "https://technology.robbyant.com/lingbot-map"
repo = "https://github.com/Robbyant/lingbot-map"
year = 2026
+++

## 这是什么

上游自己的话：LingBot-Map 是一个前馈的三维基础模型，做**流式**的三维场景解算。它的几何上下文 Transformer（Geometric Context Transformer）把坐标定位、稠密几何线索和长程漂移校正统一在一个流式框架里，靠的是锚点上下文、位姿参考窗口和轨迹记忆；前馈结构配合分页 KV 缓存注意力，在 518×378 分辨率、超过 10000 帧的长序列上稳定保持约 20 FPS。

在 Lab2Shot 里，它是「LingBot-Map 深度与相机」：每看一帧就交出这一帧的相机（位置、朝向、Focal Length）和一张深度图（深度图配上相机就是点云）。尺度是相对的，不是真实尺度，按节点上的「尺度」参数换算成厘米。

## 输入输出

**官方要什么、给什么**

- 吃：一段画面。官方 `demo.py` 收 `--image_folder` 或 `--video_path`，加 `--model_path`（必填）、
  `--image_size`（默认 518）、`--stride`、`--first_k`、`--mode`（`streaming` 或 `windowed`）、
  `--max_frame_num`（默认 1024）。
  模型这一层是 `forward(images, ...)`：`images` 是 `[S,3,H,W]` 或 `[B,S,3,H,W]`、值在 0–1
  （`lingbot_map/models/gct_base.py`）。
- 给：`pose_enc`（相机位姿编码 `[B,S,9]`）、`depth [B,S,H,W,1]`、`depth_conf [B,S,H,W]`，
  文档串里还写了 `world_points` 和 `world_points_conf`。
  `pose_enc` 由上游自己的 `pose_encoding_to_extri_intri` 换成外参加内参。

**我们怎么接的**

- 「RGB」口就是 `images`，「处理分辨率」= `--image_size`、「隔帧」= `--stride`、
  「每段最多帧数」是节点这一侧的分段长度（`--max_frame_num` 由节点按每段要记住的关键帧数自己算）。
- 「深度图」= `depth`、「置信度」= `depth_conf`、「相机」= `pose_enc` 换算出的外参加内参。
- **上游写了、拿不到的一样**：`world_points`（世界点图）——放出来的权重
  `lingbot-map.pt` / `lingbot-map-long.pt` 里**没有点图那个头**（键只有 aggregator、camera_head、depth_head），
  流式模型默认也不建它（`lingbot_map/models/gct_stream.py` 的 `enable_point=False`），所以节点上没有「点云」口。
- **上游没有、节点上也没有**：遮罩和人物框输入口。`forward` 里那个 `mask` 参数是给 `ordered_video` 用的，不是画面遮罩。

出处：简介来自 `third_party/lingbotmap/repo/README.md`；输入输出依据 `third_party/lingbotmap/repo/demo.py`、
`lingbot_map/models/gct_base.py`、`lingbot_map/models/gct_stream.py` 和 `adapters/lingbotmap/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → LingBot-Map 深度与相机 → 相机 / 深度图；点云接「深度转点云」。结果和 ViPE 相机解算、VGGT、MapAnything 互相对照。
- 用途定位是"超长镜头快速出一版"：几千帧的航拍、步行漫游、车拍，别的前馈模型要切很多段、ViPE 很慢时，它可以一口气流过去。交付级相机仍以 ViPE / 3DE 为准。
- 适合：有移动的镜头、普通到中长焦、室内外场景都行。不适合：长焦（Focal Length 会估小，见下）、变焦、镜头畸变大的素材；逐帧相机有明显抖动，需要平滑后再用。
- 尺度不是米：模型把开头几帧里场景点到相机的平均距离当作 1 个单位。要换成厘米，按 ViPE 或实测距离调节点上的"尺度"。
- 关键参数：`max_frames` 留空时 3000 帧以内一口气跑完（官方说这是它最稳的范围），更长的镜头自动分段、段与段共用约 16 个关键帧再拼接；`step` 隔帧取样，长镜头用 2 更快，轨迹也不会更差（见下）。其余（关键帧间隔、开头定尺度的帧数、记忆窗口）一般不用改。
- 模型本身不接受遮罩，节点上也没有遮罩 / 人物框输入口：画面里的人照样参与计算，结果里也不会把它们抹掉。想只算画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口。「去掉天空」参数是模型自带的天空分割。
- 节点上没有「点云」口：LingBot-Map 放出来的两份权重里没有点图头，官方的流式模型默认也不建它。要点云接「深度转点云」。

## 效果和局限

RTX 4090 上（默认 long 权重，518 分辨率），和 ViPE 相机解算对比（相机轨迹先做相似变换对齐再算误差）：

| 素材 | 结果 |
|---|---|
| iPhone 长焦跟拍，1080×1920 竖画面，300 帧（真实 Focal Length 约 5484 px，ViPE 5479 px） | Focal Length 3924 px（偏小 28%）；轨迹和 ViPE 的差 0.43 m / 13.3 m 行程（3.3%），朝向差平均 3.7°、最大 7.4°；逐帧位置抖动约 0.14 m（ViPE 0.01 m）；58 秒（每帧约 0.18 秒），显存峰值 11.7 GB |
| 同上，`step` = 2（150 帧） | 轨迹误差 2.6%，朝向差平均 3.3°，26 秒 |
| 同上，每段 100 帧强制分段（4 段拼接） | 轨迹误差 4.3%，朝向差平均 3.4°，50 秒 |
| 固定机位跳舞，864×480，124 帧 | Focal Length 646 px（ViPE 在这个不动的镜头上给 1051 px，不可靠）；相机应该不动：漂移是场景深度的 1.2%、转动最大 0.3°（ViPE 约 0.2°）；19 秒，显存 11.5 GB |
| 长镜头（固定机位素材来回播放拼成 992 / 2976 / 5084 帧） | 显存峰值 11.7 / 11.7 / 11.7 GB（124 帧时 11.5 GB），内存 1.8–2.9 GB，都不随长度增长；时间按长度线性增加，每秒约 7 帧（5084 帧 12 分钟，自动分 2 段）；相机应该一直不动，5084 帧后漂移是场景深度的 3.7% |
| 长焦跟拍来回播放 4 遍（1200 帧） | 回到同一画面时相机位置差 0.0066（行程 1.16 单位的 0.6%）、朝向差 0.5–0.7°：来回不会越跑越偏 |

- 另一份权重 balanced（论文用的版本）：长焦跟拍 Focal Length 4470 px（更接近真实），但轨迹误差 4.4%、朝向差平均 5.0°，整体转动只有 ViPE 的四分之三；固定机位 Focal Length 726 px。默认用 long。
- 竖画面：模型是按横画面训练的，竖画面上它估的横向视场明显偏宽（长焦跟拍横向 Focal Length 只有 2100 px，纵向 3900 px）。输出按长边方向的 Focal Length、方形像素给相机。把竖画面转 90° 再算，轨迹误差反而变成 13.6%，所以不这样做。
- 长焦 Focal Length 估小，深度图和点云会相应被压扁；知道实拍 Focal Length 时，相机建议用 ViPE（可以填 Focal Length）。
- 注意力用 PyTorch 自带的实现（没有装官方的 FlashInfer 加速版：要么现场编译 CUDA，要么多下 1.2 GB 预编译包），速度约为官方宣称的一半到四分之一：竖画面每秒约 5.5 帧，横画面约 7 帧。
- 开「去掉天空」时天空分割在 CPU 上逐帧跑，300 帧的镜头多花约 50 秒。

## 团队

Robbyant（官网署名 Shanghai Ant Robbyant Technology Co., Ltd.）是蚂蚁集团旗下做具身智能（机器人）的公司，同一系列还有深度估计 LingBot-Depth、视频生成世界模型 LingBot-World、机器人大模型 LingBot-VLA / LingBot-VA 等，并且自己做机器人硬件。论文由 Yao Yao 和 Yinghao Xu 担任通讯作者（Yinghao Xu 是项目负责人），入选 ECCV 2026 口头报告。

## 模型下载和安装

- 运行 `lab2shot ext install lingbotmap`：下载 LingBot-Map 代码（锁定版本，约 450 MB，仓库里带论文 PDF 和示例）、建独立 Python 环境（PyTorch 2.8 / CUDA 12.8，约 6 GB，和其他扩展共用下载缓存）、从魔搭社区 ModelScope 下载两份权重 lingbot-map-long 和 lingbot-map（各 4.6 GB，和 Hugging Face 上的文件一模一样，sha256 校验），再从 Hugging Face 下载天空分割模型 skyseg.onnx（176 MB）。国内网络下权重每秒 100 MB 以上，一两分钟；Hugging Face 上那个小文件可能反而最慢。
- 不需要申请权限，也不需要自行下载任何模型。

## 许可证说明

- 代码 Apache-2.0，**可以商用**；权重 lingbot-map-long / lingbot-map 也按 Apache-2.0 发布（官方模型卡写明，网络是从 DINOv2 初始化后自己训练的，没有用 VGGT 的非商用权重）。
- 代码里有一部分文件（几何工具、相机编码、预测头等）来自 Meta 的 VGGT，文件头写着 Meta 版权。VGGT 代码的许可证（VGGT License）允许商用，但要遵守它的使用政策：禁止军事、武器、关键基础设施等用途。
- 天空分割模型 skyseg.onnx 是 MIT。
- 训练数据里有 ScanNet、Matterport3D、HM3D、Waymo 等研究用途的数据集；官方仍按 Apache-2.0 发布权重，商用项目如需谨慎可自行评估。
- 不依赖 SMPL 人体模型、nvdiffrast 等有限制的组件。

## 参考

- 论文：https://arxiv.org/abs/2604.14141
- 项目主页：https://technology.robbyant.com/lingbot-map
- 代码：https://github.com/Robbyant/lingbot-map
- 许可证原文：https://github.com/Robbyant/lingbot-map/blob/main/LICENSE.txt
- 权重（Hugging Face）：https://huggingface.co/robbyant/lingbot-map
- 权重（魔搭 ModelScope，本扩展从这里下载）：https://www.modelscope.cn/models/Robbyant/lingbot-map
- 演示数据：https://huggingface.co/datasets/robbyant/lingbot-map-demo
- 天空分割模型：https://huggingface.co/JianyuanWang/skyseg
- VGGT 许可证（部分代码来源）：https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
