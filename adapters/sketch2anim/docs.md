+++
team = "爱丁堡大学 · Snap Inc. · 东北大学"
people = "Lei Zhong, Chuan Guo, Yiming Xie, Jiawei Wang, Changjian Li"
paper = "Sketch2Anim: Towards Transferring Sketch Storyboards into 3D Animation（ACM TOG 44(4) / SIGGRAPH 2025）"
paper_url = "https://arxiv.org/pdf/2504.19189"
repo = "https://github.com/zhongleilz/Sketch2Animation"
year = 2025
+++

## 这是什么

Sketch2Anim 要解决的是把分镜草图转成三维动画：分镜里的 2D 草图是动画师做三维动作时照着的参考，这个过程反复试错，费时又吃经验。它由草图约束理解和动作生成两个模块组成——一个三维条件动作生成器同时用三维关键姿势、关节轨迹和动作词来做精细的动作控制；再加一个神经映射器，把用户画的 2D 草图和它对应的三维关键姿势、轨迹对到同一个嵌入空间里，第一次做到用 2D 控制动作生成。

在 Lab2Shot 里，它用来**从火柴人出一段三维骨架动画**：火柴人是在「手画简笔画」`core.draw_figure` 那个节点上画的（一帧一个关键姿势，髋关节连起来就是身体走的路线），画完接到这个解算器的「草图」口；再加一句英文说清是什么动作。生成出来的 22 关节全身动作，用上游自己的 IK 转成带旋转的骨架动画，重采样到节点要的帧率，单位厘米。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`demo.sh` 执行的 `demo_kp_traj_2d.py`——一句英文描述和它的语义角色解析写在 `demo/demo.json` 里，
  控制信号是 HumanML3D 的 22 关节做成的 2D 关键姿势和关节轨迹（`demo_kp_traj_2d.py` 的 `joints_2d` / `hint_2d`）。
  那份 2D 是把三维姿势按 `Rx(angle_x) · Ry(angle_y)` 转过再投影得到的（`utils.py` 的 `rotate_pose` / `project2D`），
  训练时 `angle_x` 在 0–30、`angle_y` 在 -45–45（`dataset.py`）。
  **上游没有任何一处读画面的像素**（`demo_kp_traj_2d.py` 里进去的是 `batch['pose_2d']` / `batch['hint_2d']` 和 `batch['text']`）。
- **给**：HumanML3D 的 22 关节三维动作（20 fps）。

**我们怎么接的**

- **这个节点只有一个输入口「草图」**（2D 跟踪点），和上游一致：它就是上游那组 2D 控制关节
  （`batch['pose_2d']` / `batch['hint_2d']`），一帧一副 18 关节的火柴人，按画面像素给。
  **画它的是另一个节点**「手画简笔画」`core.draw_figure`——上游没有「从图里认关节」这一步，
  官方 demo 喂的是一份动捕的三维关节投影出来的 2D（`demo_kp_traj_2d.py` 里的 `./demo/kick.pkl`），
  所以在 Lab2Shot 里那一份由人画，是一个**显式的小工具节点**，不长在解算器身上
  （解算器的输入和官方一致，额外的输入输出都由单独的小工具节点提供）。
  「提示词」= 上游那句英文，「草图视角」= 上游的 `angle_x` / `angle_y`（只开放它训练时见过的范围）。
- 换算都在 worker 里，按上游自己的做法：18 个画出来的关节按 `utils.py convert_kps_joint` 补成 22 个；
  按 HumanML3D 数据集自己的均值姿势（`Mean_raw.npy`）把像素换算成米；归一化照训练代码（`mld/data/humanml/dataset.py`），不是照它的演示脚本。
- 「骨架动画」= 上游那 22 关节动作，用它自己的 IK（`visualization/joints2bvh.py`）转成带旋转的骨架，重采样到节点要的帧率，单位厘米。
- **不一样的一点**：上游的 2D 控制是从数据集里的三维姿势投影出来的，这里的是艺术家画出来的——同一种数据，来源不同。

**出处**：简介来自论文摘要（arXiv 2504.19189：「Storyboarding is widely used for creating 3D animations…」那一段里介绍方法的两句：
3D-based motion generator with keyposes / joint trajectories / action descriptions，和 a neural mapping that aligns 2D sketches with 3D data in a shared space）；
输入输出依据 `third_party/sketch2anim/repo/Readme.md`、`repo/demo.sh`、`repo/demo_kp_traj_2d.py`、`repo/utils.py`
和 `adapters/sketch2anim/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：「读取序列」（分镜图或参考画面）→「手画简笔画」→ **Sketch2Anim 动作生成** →「USD 输出设置」→「输出」。模板「动作生成 · Sketch2Anim 画线」就是这套（模板里没接画面，在空白画布上画）。
- **画火柴人**（在「手画简笔画」上）：双击显示那个节点，在参数面板的「关键姿势」上点「添加帧」——
  当前帧上出现一个站好的火柴人（T-pose，18 个关节，真人比例，不用自己拖出来）；再在 2D 视图里
  把关节一个个拖到你要的姿势，拖髋关节整个人一起走，右键点在火柴人身上删掉它。
- **多个关键姿势**：时间线拖到另一帧，再点一次「添加帧」；要接着上一个姿势改就点「基于前一帧」——
  把上一帧那个姿势原样复制过来。**一帧只能有一个姿势**（这一帧已经有了，两个按钮就变灰并写明原因）。
  参数面板上一帧一个「第 N 帧」的小标签，点它时间线就跳过去，方便来回切帧看画得连不连贯；
  时间线的尺子上那几个记号也是它们。**两个以上的火柴人，髋之间就是身体走的路线**（中间按直线补），
  姿势和路线一起约束生成。只画一个就只有姿势，身体大致留在原地。
- **底图只是照着画的**：「手画简笔画」的「图像」口接一段序列（分镜图、参考画面），
  时间线拖到哪一帧就显示那一帧，你在那一帧上画。**那张图一个像素都不参与计算**——
  上游只吃火柴人的关节坐标。不接也能画，那就是一张空白画布。
- **动作的长度 = 解算器上的「起始帧号」「结束帧号」**：这个节点不吃画面，长度由这两个参数说了算（按 24 帧/秒换算，节点上没有帧率参数）；
  你在第几帧画的火柴人，动作就在第几帧摆那个姿势（画在这个范围外面的，贴到最近的一端）。
- **草图视角**：画的火柴人是从哪个角度看身体的。「正面」「侧前 30」「侧前 45」三档，都在模型训练过的角度范围里（俯角 0–30°、偏角 −45–45°）。选错了生成出来的动作朝向会歪。
- **提示词**：一句英文，最好以「A person …」开头。文字编码器是 sentence-t5，只认英文。和画出来的姿势矛盾时，「贴合草图」大就听草图、「贴合描述」大就听文字。
- **画大画小不影响结果**：比例尺是第一个火柴人的身高（头到脚），换算成数据集里标准身材的 1.46 米；之后所有火柴人都用这把尺子量，所以起作用的是姿势和位置的**变化**。
- **交付的骨架**：22 个关节、Mixamo 的骨骼名（Hips / LeftUpLeg / Spine / LeftArm …）、CG 的关节朝向（主轴沿骨头），和各解算节点交出的「蒙皮角色」是同一套，Maya、Houdini 的重定向认得出来。单位厘米、Y 轴向上、地面在 y = 0。
- **接着往下做**：要套到镜头里那个人的体型上，先接「动作重定向」（「动作」接这段骨架动画，「目标」接 GVHMR / TRAM / WHAM 解出的人），它交出那个人骨架上的骨架动画；要看蒙皮再接「线性蒙皮变形」（「骨架动画」接重定向的结果，「蒙皮角色」接同一个人）。要给 Maya 就接「FBX 输出设置」。

## 效果和局限

RTX 4090 上（960×540 的分镜、96 帧 / 24 帧每秒 = 4 秒的动作，「去噪步数」2 的官方默认配置，本机实测）：

| 量 | 数值 |
|---|---|
| 去噪本身 | **0.45 秒**（「去噪步数」4 是 0.51 秒、8 是 0.56 秒——这是个 LCM 模型，多几步几乎不多花时间，也几乎没提升） |
| 头一次还要读模型 | 约 6 秒（`@resident`，同一个 worker 进程里之后的任务不再读） |
| 显存峰值 | **1.6 GB**（三个步数档位都一样） |
| 关节旋转（上游 IK） | 0.1 秒，拟合误差 **0.9 厘米** |
| 生成的姿势和画的差多少 | **5.7 厘米**（「提示词」和姿势一致时）；换视角档位时 9.2（正面）／ 12.6（侧前 30）／ 14.1（侧前 45） |
| 走的路线和画的那条线差多少 | **1.6 厘米**（两个火柴人之间那条直线） |
| 交出来的骨架 | 22 个关节、Mixamo 骨骼名、地面 y = 0（最低关节 −0.8 ~ +1 厘米）、头 1.50 米 |
| 环境占用 | 权重约 1.4 GB + 独立 Python 环境 |
| 支持的显卡架构 | sm_70 / 75 / 80 / 86 / 90 / 100 / **120**（torch 2.8 + cu128）：**4090 和 5090 都能跑** |

**草图只管你看到的那两个方向。** 火柴人是正交投影下的一张图，纵深（离镜头远近）它说不出来，由模型自己定：
上面那条「走的路线差 1.6 厘米」说的是**投影之后**贴得多准；同一次里身体在三维里一共走了 2.06 米，
其中约 1.8 米是往纵深走的——「a person walks forward」这句话定的。要管纵深，把它写进「提示词」，
或者换一个视角档位再画一次。

- 做不了的：
  - **手指、表情、手型**：模型只有 22 个身体关节，没有手指、下巴和眼睛。
  - **一次最长 9.8 秒**（模型自己的 196 帧 / 20 帧每秒）：按 24 帧每秒换算，起始帧号到结束帧号最多 235 帧，超了当场拒绝并告诉你能填多少。更长的镜头分段生成。
  - **比 2 秒还短的段**（24 帧每秒的 48 帧）模型没训练过，照算但节点上会警告。
  - **不是精确的姿势匹配**：它按草图的意思生成一段自然动作，不是把火柴人逐关节复刻出来。要一帧一帧精确对位，还是得 K 关键帧（模板「动作补帧 · Kimodo」那条路）。
  - 只会 HumanML3D 里有的日常动作；打斗、特技、和道具互动这类会退化。
  - 一个人。多人要分别生成再合成场景。

## 团队

爱丁堡大学（Lei Zhong、Changjian Li）、Snap Inc.（Chuan Guo，text-to-motion 和 HumanML3D 的作者之一）、东北大学（Yiming Xie，OmniControl 的作者之一）。这一组人同时也是 MotionLCM、OmniControl、HumanML3D 这条线的作者，所以代码就是从 MLD / MotionLCM 长出来的。

## 模型下载和安装

- 自动安装：`lab2shot ext install sketch2anim`，下载：
  - 锁定版本的官方代码（`zhongleilz/Sketch2Animation` @ 5b781ee2）；
  - 独立 Python 环境（PyTorch 2.8 + CUDA 12.8）；
  - 官方权重 `adapter.ckpt`（357 MB）和 `pretrain_22joint_combine_adapter.ckpt`（303 MB），来自作者的公开 Google Drive 文件夹（按 sha256 校验）；
  - 文字编码器 `sentence-transformers/sentence-t5-large`（1.3 GB，Apache-2.0）。
- 不需要申请，装好后完全离线计算。总占用约 1.4 GB 权重 + 环境。
- **上游 demo 里的这些不装**：HumanML3D 数据集（要先向 AMASS 申请再自己跑一遍预处理）、glove、SMPL 网格、t2m 评测器、画图和出视频的库。模型推理只要仓库里已经带着的四份归一化数（`datasets/humanml3d/Mean.npy`、`Std.npy`、`humanml_spatial_norm/Mean_raw.npy`、`Std_raw.npy`），worker 自己读它们；上游那个数据加载器只是为了把这四份数递出来，才要整份数据集。
- **t2m 评测器不装**：上游 `MLD.__init__` 无条件加载它（453 MB），但它只喂论文里的 FID / R-precision 指标，生成这条路一个张量都不读。worker 里覆写 `_get_t2m_evaluator` 跳过，权重里那 96 个评测器张量留在原地，**别的一个都不少**（非评测器的缺失张量 0 个）。要复现论文数字时再单独装。
- **原始仓库一个字节都没改**：numpy 早就删掉的 `numpy.core.umath_tests`（上游 `visualization/Animation.py` 还在 import）由 worker 在 `sys.modules` 里补一个只有 `matrix_multiply = np.matmul` 的壳；`np.float`、`np.int` 这些删掉的别名靠把 numpy 钉在 1.23.5 解决，不打补丁。

## 许可证说明

**仅限研究。** 代码是 MIT（Copyright (c) 2025 Lei Zhong，原文允许商用、修改和再发布），但官方预训练权重是用 HumanML3D 训练的，HumanML3D 的动作来自 AMASS，AMASS 的许可只许非商业的学术研究（与同样用 AMASS 的 StableMotion 同一档），所以权重和生成出来的动作只能用于研究和评估。文字编码器 sentence-t5-large 是 Apache-2.0（可商用），不改变上面的结论。要商用得用有授权的动捕数据重新训练。

## 参考

- 论文：https://arxiv.org/pdf/2504.19189
- 项目页：https://zhongleilz.github.io/Sketch2Anim/
- 代码（**真代码在这里**）：https://github.com/zhongleilz/Sketch2Animation
- 底座 MotionLCM：https://github.com/Dai-Wenxun/MotionLCM ，MLD：https://github.com/ChenFengYe/motion-latent-diffusion
- HumanML3D：https://github.com/EricGuo5513/HumanML3D

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
