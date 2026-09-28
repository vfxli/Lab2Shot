+++
team = "加州大学伯克利分校（UC Berkeley）+ Google DeepMind"
people = "Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, Angjoo Kanazawa"
paper = "Continuous 3D Perception Model with Persistent State（CVPR 2025 Oral）"
paper_url = "https://arxiv.org/abs/2501.12387"
website = "https://cut3r.github.io/"
repo = "https://github.com/CUT3R/CUT3R"
year = 2025
+++

## 这是什么

CUT3R 的论文题目是《Continuous 3D Perception Model with Persistent State》（CVPR 2025 Oral）。
摘要里作者这样介绍它：一套统一的框架，能解一大类三维任务；做法是让模型保有一份状态，每来一次
新的观测就把这份状态更新一次。给一串画面，这份不断演进的状态能以在线的方式，为每一张新画面给出
真实尺度的点图（pointmap，逐像素的三维点）；这些点图位于同一个坐标系下，可以累积成一份连贯、
稠密的场景，并随新画面到来而更新。它接受长短不一的输入，既可以是视频流，也可以是无序的照片，
画面里既有静止内容也可以有运动内容。

在 Lab2Shot 里，它是「CUT3R 深度与相机」：一帧一帧往下看，每看一帧就报出这一帧的相机（位置、
朝向、Focal Length）和这一帧的点。因为上游的训练数据里本来就有走动的人和开过的车，画面里有运动物体时
它照样出相机。

## 输入输出

**官方要什么、给什么**

- 吃：一个画面序列的文件夹。官方 `demo.py` 收 `--seq_path`（画面序列所在目录）、
  `--size`（画面缩放到的边长，512 或 224）、`--model_path`、`--vis_threshold`。读进来只组成 `img` 一项送进模型，
  **模型不吃遮罩、不吃框、不吃相机**。
- 给：每帧四样——`pts3d_in_self_view`（这一帧自己相机坐标里的点图）、
  `pts3d_in_other_view`（同一份几何在世界坐标里的点图）、`conf_self` / `conf`（两张置信度）；
  相机位姿是它的位姿头 `camera_pose`，Focal Length 由 `estimate_focal_knowing_depth` 从自视角点图估出来。

**我们怎么接的**

- 「RGB」口就是 `--seq_path` 那个序列，「处理分辨率」就是 `--size`。
- 「深度图」口取的是 `pts3d_in_self_view` 的 `Z`，「置信度」口是 `conf_self`，
  「相机」口是 `camera_pose` 加那个估出来的 Focal Length，「点云」口是官方的另一张点图 `pts3d_in_other_view`
  （worker 用这一帧官方自己的位姿把它放回相机空间，再按拼好的相机摆回世界——同一份数据换坐标系，没有别的计算）。
- 上游一次读完整段；节点按「每段最多帧数」分段算再拼，是因为模型读得越久越漂（它是用 4–64 帧训练的，显存不随段长增长），模型这一侧的输入没有变。
- **上游没有、节点上也没有**：遮罩和人物框输入口。要只算画面的一块，接「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→「RGB」口。

出处：简介来自论文摘要（arXiv 2501.12387）；输入输出依据 `third_party/cut3r/repo/demo.py` 和
`adapters/cut3r/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → CUT3R 深度与相机 → 相机 / 深度图 / 点云，和 ViPE 相机解算、MonST3R、VGGT 的结果互相对照。「点云」口交的是官方自己的世界点图（`pts3d_in_other_view`），不是从深度图反投影的。
- 用途定位是"快速出一版"：纯前馈、不做平差，120 帧的镜头半分钟出结果，适合视图、粗匹配、交叉检验；最终交付的相机还是以 ViPE / 3DE 为准。
- 适合：普通焦段、有移动的手持 / 跟拍镜头，画面里有人走动也可以。不适合：长焦（Focal Length 会明显估小，见下）、变焦、镜头畸变大的素材；固定机位能用，但相机会有轻微漂移。
- 关键参数：`max_frames` 每段帧数（默认 64，官方模型就是用 4–64 帧训练的），更长的镜头自动分段、段与段共用 `overlap`（默认 16）帧拼接。分段比一口气读 120 帧更准（见下），一般不用改。`step` 隔帧取样，很长的镜头用 2–4 省时间。
- 「记忆更新」选 **TTT3R** 是长镜头模式：记忆按和新画面的相关程度一点点更新，能连续读得更久（默认每段 200 帧），固定机位和几百帧的长镜头更稳，见下面的数字。它是 TTT3R 论文（MIT 许可）的更新规则，用的还是 CUT3R 的权重和环境，不另外下载。
- 模型本身不接受遮罩，节点上也没有遮罩 / 人物框输入口：它本来就是在有人走动的素材上训练的，运动的人照样参与计算，结果里也不会把它们抹掉。想只算画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口。

## 效果和局限

RTX 4090 上，和 ViPE 相机解算对比（相机轨迹先做相似变换对齐再算误差）：

| 素材 | 结果 |
|---|---|
| iPhone 长焦跟拍，1080×1920，120 帧（真实 Focal Length 约 5484 px，ViPE 5100 px） | Focal Length 3930 px（偏小 28%）；轨迹和 ViPE 的差 0.29 m / 4.0 m 行程（7.3%），朝向差平均 1.9°、最大 2.9°；分 3 段，共 32 秒（含 9 秒载入模型，推理每帧 0.1 秒），显存峰值 3.6 GB |
| 固定机位跳舞，864×480，124 帧（真实 Focal Length 约 560–590 px） | Focal Length 470 px（偏小约 18%，ViPE 在这个镜头上是 1053 px）；相机应该不动，CUT3R 的漂移是场景深度的 3.7%、转动最大 1.3°（ViPE 分别是约 1% 和 0.2°）；28 秒，显存 3.6 GB |

- **长镜头模式：「记忆更新」选「TTT3R」**（TTT3R，ICLR 2026，同一份 CUT3R 权重，只改"每读一帧记忆写进多少"）。和 ViPE 按比例+旋转+位置对齐后比较：

  | 素材 | CUT3R，每 64 帧一段 | CUT3R 一口气读完 | TTT3R 一口气读完 | TTT3R 每 200 帧一段 |
  |---|---|---|---|---|
  | 长焦跟拍整段 792 帧（1080×1920，60 fps，ViPE 行程 18.7 米） | 位置差 5.8%，朝向差中位 34° | 26.6%，方向转丢（180°） | 15.9%，方向转丢（148°） | **4.4%**，朝向差中位 25°（段间重叠 32 帧时 4.4%、15°） |
  | 同一镜头前 300 帧 | **4.2%**，9° | 22%，方向转丢 | 5.8%，7° | 4.7%，14° |
  | 固定机位 124 帧（应该不动） | 漂 10.9 cm、转 1.1° | — | **漂 4.0 cm、转 0.36°** | （同左，一段） |

  - 固定机位上 TTT3R 明显更稳（漂移少 2–3 倍）；长镜头上它比原版一口气读完好得多，但一口气读 792 帧也会转丢方向，所以默认每 200 帧一段再拼接：792 帧的长镜头上位置差比原版 64 帧一段小约四分之一；300 帧的镜头上反而略差（4.7% 对 4.2%），几百帧以内用原版即可。朝向和 ViPE 仍差十几度，主要是长焦上 Focal Length 估小（视角 19–21°，实际约 11°）带来的，TTT3R 管不到；要准的相机仍用 ViPE / COLMAP。
  - 速度、显存和原版一样（每帧约 0.1 秒；一段 200 帧时显存约 4.3 GB，一口气 792 帧约 7.4 GB）。
- 同一个镜头一口气读 120 帧（不分段）时轨迹误差反而增大到 11.5%；每段 32 帧时是 6.4%；"每段读两遍"（revisit=2）没有改善、时间翻倍。
- 尺度接近米但不准：长焦跟拍上 CUT3R 的行程只有 ViPE 的一半左右（两者谁更接近真实无法确认），当成"大致真实尺度"用。
- 长焦镜头 Focal Length 估小很多，深度图和点云会被相应压扁；知道实拍 Focal Length 时建议用 ViPE（可以填 Focal Length）或 MonST3R 的"已知 Focal Length"参数。
- 远处大面积平整区域（楼面、天空）置信度很低，不会进输出点云。
- 相机轨迹逐帧有小抖动（不做平差），需要平滑时在下游处理。
- 官方建议编译的 cuRoPE 加速核没有编译：本扩展用同一公式的 PyTorch 版本替代，结果完全一致。

## 团队

加州大学伯克利分校（UC Berkeley）Angjoo Kanazawa 和 Alexei Efros 两位教授的实验室，和 Google DeepMind（Aleksander Holynski）合作，一作 Qianqian Wang、Yifei Zhang。Efros 是 pix2pix、CycleGAN 等图像合成经典工作的作者之一；Kanazawa 组做过 HMR（单图人体网格）、Nerfstudio、4DHumans；Qianqian Wang 还做过 OmniMotion（视频全局点跟踪）和 IBRNet。CUT3R 是 CVPR 2025 口头报告论文。

## 模型下载和安装

- 运行 `lab2shot ext install cut3r`：下载 CUT3R 代码（锁定版本）、建独立 Python 环境（PyTorch 2.10，约 7 GB，和 VGGT / Pi3 / MonST3R 共用下载缓存，缓存已有时 1 分钟内），再从作者的 Google Drive 下载 cut3r_512_dpt_4_64 权重（3.2 GB，下载后校验 sha256）。Google Drive 单连接很慢时要半小时左右。
- 不需要申请权限，也不需要自行下载任何模型。

## 许可证说明

- **非商用**，只能用于研究：CUT3R 的代码和权重都是 CC BY-NC-SA 4.0（署名、禁止商用、改编后的作品必须用同样的许可证发布）。
- 代码里包含的 Naver DUSt3R / CroCo 同样是 CC BY-NC-SA 4.0；CroCo 里有两个文件（pos_embed.py、blocks.py）含 Meta MAE 的 CC BY-NC 4.0 部分和 timm 的 Apache-2.0 部分。
- 没有其他隐藏依赖（不需要 SMPL 人体模型、nvdiffrast 等）。
- 「记忆更新」选「TTT3R」时的更新规则来自 TTT3R（MIT，可商用），但它用的仍是 CUT3R 的权重，所以整个节点仍然非商用。

## 参考

- 论文：https://arxiv.org/abs/2501.12387
- 项目主页：https://cut3r.github.io/
- 代码：https://github.com/CUT3R/CUT3R
- 许可证原文：https://github.com/CUT3R/CUT3R/blob/main/LICENSE
- 权重（作者的 Google Drive）：https://drive.google.com/file/d/1Asz-ZB3FfpzZYwunhQvNPZEUA8XUNAYD/view
- 它的基础 DUSt3R（Naver）：https://github.com/naver/dust3r
- 长镜头模式 TTT3R：论文 https://arxiv.org/abs/2509.26645 ，代码 https://github.com/Inception3D/TTT3R （MIT；Lab2Shot 按它的公式在 worker 里重写了更新规则，数学上和它的实现相同，没有和它的程序逐帧对比过）
