+++
team = "InterDigital R&D France + Inria（Rennes）"
people = "Lucas Mourot, Ludovic Hoyet, François Le Clerc, Pierre Hellier"
paper = "Deep Learning for Foot Contact Detection, Ground Reaction Force Estimation and Footskate Cleanup（SCA 2022 / Computer Graphics Forum 41.8）"
paper_url = "https://doi.org/10.1111/cgf.14635"
website = "https://github.com/InterDigitalInc/UnderPressure"
repo = "https://github.com/InterDigitalInc/UnderPressure"
year = 2022
+++

## 这是什么

UnderPressure 这篇论文处理的是从动捕数据自动判断脚接触。作者先公开了 UnderPressure——一套带压力鞋垫标注的动捕数据库，鞋垫数据是脚与地面接触的可靠依据；再用深度学习训练出一个神经网络，从动作数据估计施加在两只脚上的地面反作用力，据此得出准确的脚接触标记。在此之上给出一套全自动的脚滑清理流程：先由估计出的地面反作用力得到脚接触标记，再用优化式逆向运动学（IK）求解脚部约束，把脚滑去掉，并保证与估计出的地面反作用力一致。

在 Lab2Shot 里，这个节点叫「UnderPressure 脚滑清理」，属于动作清理这一族：接一段骨架动画，交出清理过的动画、每帧的足底力和脚接触。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：动捕动作——关节角、骨架和根轨迹，经 `anim.FK` 算成关节位置再送进网络（`demo.py`）；
  采样率是它训练用的那一个（`data.FRAMERATE`）。
- **给**：`model.vGRFs(positions)`——每帧、每只脚、16 个鞋垫格子的垂直地面反作用力（单位是体重倍数）；
  由它得出每帧的脚接触；`footskate.Cleaner` 再按这些接触做优化式 IK，交出清理过的动作
  （`cleaner(item["angles"], item["skeleton"], item["trajectory"])`——进出是同一组数据）。

**我们怎么接的**

- 「动画」口按契约重定向到上游自己的 23 关节骨架（静止姿势对齐、腿长缩放），并重采样到它训练的那个采样率。
  这一步不能省：按 24 帧每秒原样喂进去，网络会把动作读成近乎静止的姿势、几乎每帧都判成接触
  （和重采样后的正确答案只有 69% 一致，而且不报错，见 `adapters/underpressure/worker.py`）。
- 「足底力」= 上游的 vGRFs，「脚接触」= 由它得出的每帧四个接触，「动画」输出 = `footskate.Cleaner` 清理过的动作。
- 「接触余量」= 接触往两边各放宽几帧。
- 上游只给接触和足底力两样（`demo.py` 只有 contacts 和 vGRFs），节点不另外输出「脚滑」曲线。
- 这个模型是四层卷积，**在 CPU 上算**（500 帧清理 CPU 2.4 秒、RTX 5090 2.7 秒）。

**出处**：简介来自论文摘要（CGF 41.8 / arXiv 2208.04598：「This paper addresses automatic foot contact label detection from motion capture data with a deep learning based method…」
「…a fully automatic workflow for footskate cleanup… an optimisation-based inverse kinematics (IK) approach」）；
输入输出依据 `third_party/underpressure/repo/demo.py`、`repo/README.md`
和 `adapters/underpressure/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：「导入 FBX」（动捕棚或重定向出来的角色动画）→ **UnderPressure 脚滑清理** →「FBX 输出设置」→「输出」。模板「脚滑清理 · UnderPressure」就是这套。
- **接在谁后面最有用**：
  - 重定向之后（体型换了，脚就开始飘）；
  - 视频解出来的人体动作之后（「GVHMR 全身动作」「TRAM 全身动作」→「提取骨架」→ 这个节点）；
  - 「StableMotion 动捕清理」之后——先修坏帧，再钉脚，这是生产里的顺序，模板「动捕清理 · StableMotion + UnderPressure」就是这条链。
- **关节映射**：它的骨骼是 Xsens MVN 的 23 个关节。接上人物后表格里每行显示自动猜到的人物关节，猜错了在下拉里改。手指、面部、扭转骨骼不参与，**保持原来的动画不动**。
- **脚接触**：判断结果作为四条曲线（左脚跟、左脚尖、右脚跟、右脚尖）一起交出来，0 是抬起、1 是着地。拿到 Nuke 或 Maya 里就能驱动 IK 开关，也可以先看看它判得对不对再决定调不调参数。
- **足底力**：模型估的每一帧、每只脚、16 个鞋垫单元各承了多少力（占体重的比例），32 条曲线（左脚 1–16、右脚 1–16）。「脚接触」就是由它判出来的，所以它更原始：要自己定着地阈值、或者想看压力分布时用它。
- **整段都会按模型清理一遍**，没有「只改问题帧」的选项。只想改真正在滑的帧，需要先量出脚滑再挑帧，目前没有现成节点做这件事。
- **接触余量**：每段着地前后各多算几帧算作着地。脚离地的瞬间还在抖就调大；碎步、原地小跳调小。
- **地面在 y = 0**：模型按人站在 y = 0 的地面上判断。视频解出来的人物先接「自动落地」。
- **帧率**：模型按 100 帧/秒训练。节点自己把动作升采样到 100 帧/秒判断接触，结果再换算回镜头自己的帧率，节点上会提醒一句升采样了。

## 效果和局限

### 数字（节点在 CPU 上跑，不占显卡）

素材：AMASS 里 8 段真实走路动捕（各 5 秒，24 帧/秒）和 LAFAN1 的 `walk1_subject1`。

- **脚滑清得掉**：8 段着地帧的脚滑中位数合计 0.794 → 0.198 cm/帧（24 帧/秒，即 19 → 4.8 cm/秒），少 75%。其中真正在滑的那一段（BMLrub，5.56 cm/帧 = 133 cm/秒）清到 0.76 cm/帧，**少 86%**。
- **本来不滑的动捕会被多改一点**：整段清一遍会让干净的动捕多滑一点（0.025 → 0.174 cm/帧），干净的动捕上要留意这一点。
- **代价**：姿势最大变动平均 7.8 cm（单个关节、单帧），真正在滑的那一段 27 cm——脚要钉住，腿和根就得动。
- **支撑力估得准**：整段平均 0.93–1.05 倍体重（走路一个完整步态周期的平均值理论上正好等于体重），双脚同时着地时 1.08–1.33 倍，跑动有腾空的段落最低接近 0。这一条也是节点自带的体检：算出来不在 0.6–1.5 之间就说明骨骼比例或关节映射不对，节点会警告。
- **速度**：LAFAN1 60 秒的动捕 34 秒，30 秒的 19 秒，10 秒的 15 秒（CPU，6 线程）。
- **帧率**：24、25、30 帧/秒的素材升采样到 100 帧/秒再判断，和原生 100 帧/秒的结果一致度 98.8%–99.1%（F1 0.983–0.987）。**不升采样、把 24 帧/秒当成 100 帧/秒喂进去只有 69%**——那是个看不出来的错误答案，所以节点一定会自己重采样。
- **体型不影响**：骨架放大缩小 ±15%（约 1.45 m 到 1.97 m），接触判断变化不到 1%。
- **地面**：模型把接触钉在高度 0 上。接进来的动捕不在地面上时（比如 AMASS 的某段比地面低了 24 cm），节点先把整段放到地面再清理、清理完原样移回去；不这么做，整个人会被拽起来，最大变动 35 cm、脚反而更滑。

### 局限

- **只管脚**。上半身、手、头它不判断也不修；
- 训练数据是日常动作（走、跑、上下楼梯、蹲、搬重物），**卡通动作、特技、悬空的表演**判不准；
- 赤脚和穿鞋的重量分布不同，模型按穿鞋的数据训练；
- 官方演示的迭代次数（100）是最稳的一档：50 次和 200 次在 12 段官方样本上分别清掉 30%、31%、27% 的脚滑，**没有「次数越多越干净」这回事**（它自己的权重会把姿势往原样拉回去），所以节点不把它做成参数；
- 它**不判断这一帧的姿势是不是坏的**：抖动、穿模、关节跳变要用「StableMotion 动捕清理」。

## 团队

InterDigital R&D 法国实验室和 Inria Rennes 的 MimeTIC 组。第一作者 Lucas Mourot 同一时期还做过角色动画的综述和 UnderPressure 数据库本身（10 个人、约 5 小时、压力鞋垫 + Xsens 动捕，是这个方向上少见的带真值的公开数据）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install underpressure`。
- **什么都不用下载**：训练好的网络（4 MB）就在官方仓库里（`pretrained.tar`），跟着代码一起拉下来；环境里只装 PyTorch。
- 磁盘：仓库 45 MB + 环境约 5 GB（PyTorch cu128）。

## 许可证说明

**仅限研究**。代码和预训练网络都在 InterDigital 的「Limited Software Evaluation License」下：只允许 “fundamental research work”，**明文排除一切商业用途**（包括放进任何提供给第三方的产品或服务，不论收不收费）。比一般的「非商用」更严，所以节点上标「仅限研究」。

许可证还要求：任何由此产生的发表要注明「UnderPressure is an InterDigital product」并引用原文；不得修改软件本身——Lab2Shot 不改原仓库的任何文件，所有适配都在自己的 worker 里做。

数据库（`files.inria.fr/UnderPressure/`）另有自己的条款，Lab2Shot 不下载它，节点也不需要它。

## 参考

- 论文：https://doi.org/10.1111/cgf.14635
- 仓库：https://github.com/InterDigitalInc/UnderPressure
- 许可证：https://github.com/InterDigitalInc/UnderPressure/blob/main/LICENCE.txt
