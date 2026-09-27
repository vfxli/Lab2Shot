+++
team = "浙江大学 CAD&CG 国家重点实验室（周昆组）"
people = "Jia Qin, Youyi Zheng, Kun Zhou"
paper = "Motion In-betweening via Two-stage Transformers（ACM TOG / SIGGRAPH Asia 2022）"
paper_url = "https://doi.org/10.1145/3550454.3555454"
repo = "https://github.com/victorqin/motion_inbetweening"
year = 2022
+++

## 这是什么

Two-stage Transformer 是一套基于深度学习的框架，分两个阶段合成动作补帧。给定若干上下文帧和一个目标帧，它以非自回归的方式生成长度可变、合理的过渡动作。框架由两个基于 Transformer Encoder 的网络组成：第一阶段的 Context Transformer 依据上下文生成粗略的过渡，第二阶段的 Detail Transformer 再精修动作细节。它对动画师友好，支持在过渡段内施加完整或局部的姿势约束；基准数据是 LAFAN1。

在 Lab2Shot 里，这个节点叫「Two-stage Transformer 动作补帧」：接一段骨架动画，在「关键帧」标出的每两个端点之间补出中间帧。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：LAFAN1 骨架上的动作——若干上下文帧、一个目标帧，和要补多少帧：
  `python eval_detail_model.py lafan1_detail_model lafan1_context_model -t <过渡帧数> -i <片段序号>`（README:73、93）。
  LAFAN1 数据集要先下载解压到 `datasets/lafan1`（README:11-13），预训练权重解压到 `experiments`（README:15）。
  一次调用最多 65 帧。
- **给**：补出来的过渡动作——位置加旋转一对（`packages/motion_inbetween/train/detail_model.py:515-584` 的 `pos_new` / `rot_new`），
  外加基准统计；`post_process` 是它自己的参数（同一段 :517）。

**我们怎么接的**

- 「动画」口（骨架动画或蒙皮角色）按契约先转到 LAFAN1 的骨架和 30 fps；接进去的在上游那边就是 `positions` + `rotations` 一对，
  补出来的是 `pos_new` + `rot_new` 一对——一个口对应的是这一对，不是两样数据。
- 「关键帧」参数 = 上游的两个端点：每两个关键帧之间是上游方法的一次完整计算（Context Transformer 然后 Detail Transformer），
  按它自己评估脚本的摆法——10 帧上下文接到前一个关键帧、中间是过渡、后一个关键帧作为目标，再加它后面一帧。
- 「衔接平滑」= 上游的 `anim_post_process` 曲线偏移。
- **不一样的两点**：① 上游一次最多 65 帧，所以两个关键帧最多隔 54 帧（1.8 秒），节点上按这个拦；
  ② 上游的上下文来自数据集里真实的前 10 帧，动画师只给关键帧，所以第一段的上下文是我们按
  「从第一个关键帧朝第二个关键帧的直线往回延」造出来的，后面每一段的上下文是前一段补出来的结果（`adapters/two_stage_transformer/worker.py:7-18`）。

**出处**：简介抽自论文摘要前两句（Motion In-betweening via Two-stage Transformers，ACM TOG 41(6) 184：
两阶段框架、给上下文帧和目标帧、非自回归地生成长度可变的过渡，Context Transformer 出粗略过渡、Detail Transformer 精修细节）；
输入输出依据 `third_party/two_stage_transformer/repo/README.md:11-15、:73、:93`、
`repo/packages/motion_inbetween/train/detail_model.py:515-584` 和 `adapters/two_stage_transformer/nodes.py:17-23`、`worker.py:7-20`。

## 在 Lab2Shot 里怎么用

- 典型接法：「导入 USD」或「导入 FBX」（Maya 导出的角色，只导关键帧那几帧；骨架动画、蒙皮角色都行）→ **Two-stage Transformer 动作补帧** → 「USD 输出设置」→「输出」。模板「动作补帧 · Two-stage Transformer」就是这套，Maya 插件以后也调用它。
- **关键帧**：留空时，文件里动画有记录的帧就是关键帧；动画每帧都烘焙过的文件，在「关键帧」里写要保留的帧，如 `1001, 1012, 1030` 或 `1001-1100x8`（每 8 帧一个）。
- **关节映射**：模型用的是 LaFAN1 的 22 个关节。接上人物后，表格每一行显示自动猜到的人物关节（「自动 · Character1_LeftUpLeg」这样）；猜错了在下拉里选。Maya HumanIK、Mixamo、Unreal、SMPL 这些命名都认得，扭转骨骼、手指、面部骨骼不参与，**保持原来的动画不动**。人物的绑定姿势是 T-pose 还是 A-pose、每个关节怎么定向都没关系，Lab2Shot 按骨头方向对齐。
- **关键帧精确**（默认开）：每个关键帧和原来的姿势完全一样；**脚锁定**（默认开）：模型判断脚着地的帧，用两骨 IK 把脚踝钉住。
- **地面**：模型默认人站在 y = 0 的地面上（Y 轴向上）。视频解出来的人物（GVHMR、SAM 3D Body……）先接「自动落地」。
- **帧率**：模型按 30 帧/秒生成，结果换算回人物自己的帧率（24、25、29.97……）。
- **第一个关键帧前面的上下文**：模型需要过渡开始前 10 帧的动作（它是从一段动作「接着往下补」的）。动画师只给了关键帧，所以 Lab2Shot 用前两个关键帧之间的直线路径往前延伸 10 帧当作这段上下文（相当于角色以前两个关键帧之间的平均速度走进第一个关键帧）；之后每段的上下文就是前面补出来的动作。最后一个关键帧后面的一帧同样按直线延伸。
- **两个关键帧最多隔 1.8 秒**（30 帧/秒的 54 帧，模型一次最多看 65 帧）：隔得更远的地方要加关键帧，或者用「Kimodo 动作补帧」。

## 效果和局限

- 速度：一段 10 秒的镜头约 0.2–0.8 秒，显存约 0.2 GB。
- 局限：
  - 只会 LaFAN1 里有的动作类型；打斗、乐器、特技这类会退化。输出是动捕风格，不是手 K 风格。
  - 对输入的“干净程度”敏感：视频解算出来、地面不平、脚浮在空中的动作（比如 GVHMR 在手机长焦镜头上的结果），它补出来的中间帧几乎不动、到关键帧才跳过去；这类动作用 Kimodo。
  - 手指不动（LaFAN1 没有手指）；人物的手指保持原来的动画。

## 团队

浙江大学计算机辅助设计与图形学（CAD&CG）国家重点实验室，秦佳（Jia Qin）、郑友怡、周昆。周昆老师的组在图形学和动画上发表了大量 SIGGRAPH 论文；秦佳后来又做了基于扩散模型和物理仿真的补帧（2025）。

## 模型下载和安装

- 自动安装：`lab2shot ext install two_stage_transformer`，下载：
  - 锁定版本的官方代码；
  - 独立 Python 环境（PyTorch 2.8 + CUDA 12.8，约 4 GB）；
  - 官方预训练权重（GitHub Releases，Context 和 Detail 两个 Transformer，214 MB）；
  - Ubisoft LaFAN1 数据集（144 MB）：worker 从里面读 LaFAN1 骨骼的骨长和一个站立的参考姿势，用来和你的人物对齐。
- 不需要申请，装好后完全离线运行。

## 许可证说明

**不能商用。** 代码是 MIT，但官方权重是用 Ubisoft LaFAN1 训练的，LaFAN1 是 CC BY-NC-ND 4.0（署名、非商用、禁止演绎），所以权重和补出来的动作只能用于研究和评估。要商用，得用有授权的动捕数据（比如工作室自己的动捕，或者条款允许的 BONES-SEED）重新训练；需要可商用的补帧直接用「Kimodo 动作补帧」。

## 参考

- 论文：https://doi.org/10.1145/3550454.3555454
- 代码和预训练权重：https://github.com/victorqin/motion_inbetweening
- LaFAN1 数据集：https://github.com/ubisoft/ubisoft-laforge-animation-dataset
- 前作 Robust Motion In-betweening（Ubisoft La Forge，SIGGRAPH 2020）：https://montreal.ubisoft.com/en/automatic-in-betweening-for-faster-animation-authoring/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
