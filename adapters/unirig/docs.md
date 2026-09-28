+++
team = "清华大学 + Tripo（VAST AI Research）"
people = "Jia-Peng Zhang, Cheng-Feng Pu, Meng-Hao Guo, Yan-Pei Cao, Shi-Min Hu"
paper = "One Model to Rig Them All: Diverse Skeleton Rigging with UniRig（SIGGRAPH 2025 / TOG）"
paper_url = "https://arxiv.org/abs/2504.12451"
website = "https://zjp-shadow.github.io/works/UniRig/"
repo = "https://github.com/VAST-AI-Research/UniRig"
year = 2025
+++

## 这是什么

UniRig 是 TOG（SIGGRAPH）上的自动绑定统一框架，由清华大学和 Tripo 开发。给三维模型做绑定——生成骨架、分配蒙皮权重——是三维动画里关键却复杂耗时的一步，UniRig 借助大型自回归模型，用一套统一的框架把这件事在各类三维资产上自动化。整套系统分两个阶段：一是骨架预测，用 GPT 式的 transformer，按新提出的 Skeleton Tree Tokenization 自回归地预测出拓扑合法的骨架层级；二是蒙皮权重与属性预测，用 Bone-Point Cross Attention，依据预测出的骨架和输入网格几何算出每个顶点的蒙皮权重。

在 Lab2Shot 里，这个节点叫「UniRig 自动绑定」，是绑定这一族唯一的节点：接一块没有骨骼的模型，交出蒙皮角色。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一个模型文件（`--input examples/giraffe.glb`）。三条命令依次执行：
  `generate_skeleton.sh` 出骨架（`--seed` 换一副骨架），
  `generate_skin.sh` 拿上一步的骨架文件出蒙皮，
  `merge.sh --source <蒙皮结果> --target <原模型>` 把结果并回原模型。
- **给**：两份 npz——`predict_skeleton.npz`（joints / parents / names，`src/system/ar.py`）和
  `predict_skin.npz`（skin / vertices / joints，`src/system/skin.py`）；官方就是靠这两份合成一副能动的绑定，没有别的产物。

**我们怎么接的**

- 「模型」口（三角网格，厘米、Y 向上、世界坐标）= 上游第一步用 Blender 打开模型文件读出来的那份网格：
  网格我们手里已经有了，就不再读一遍文件，按它自己的 `save_raw_data`（`src/data/extract.py`）清一遍，
  面数超过 50000 就降到 50000（上游 `faces_target_count` 的默认值）。
- 「蒙皮角色」= 上面那两份 npz 合起来的绑定；「影响骨骼数」是每个顶点留几根骨头，「随机种子」= 上游的 `--seed`。
- **不一样的两点**：① 上游的 `merge.sh` 是把权重按最近顶点搬回原网格，我们改调它自己的 `src.system.skin.reskin`
  （参数和它的 SkinWriter 一模一样：median、alpha 2.0、threshold 0.03），那个函数本来就是「从采样点铺到任意网格的顶点上」，更准；
  ② 上游内部把模型归一化到 `[-1, 1]` 的立方体、骨架和权重两个阶段各归一化一次，我们不去猜它的公式，
  从它自己写出的那几份 npz 里把这个相似变换解出来再换回我们的单位，误差写进报告。

**出处**：简介来自 `third_party/unirig/repo/README.md`（「This repository contains the official implementation for the SIGGRAPH'25 (TOG) UniRig framework, a unified solution for automatic 3D model rigging」）
和（「While UniRig uses separate stages for skeleton prediction and skinning…」）；
输入输出依据同一份 README、`repo/src/data/raw_data.py`、`repo/src/system/ar.py`、`repo/src/system/skin.py`
和 `adapters/unirig/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：「导入 USD」（或「导入 FBX」）选中模型 → **UniRig 自动绑定** → 「USD 输出设置」和「FBX 输出设置」→「输出」。模板「自动绑定 · UniRig」就是这套。
- **网格、UV、分区、材质全都保留**：Lab2Shot 不走 UniRig 自己的 Blender 导出，而是把骨架和蒙皮权重写进**你导进来的那份 USD**，只多出一个 SkelRoot、一副骨架和每个网格上的蒙皮。交出去在 Maya 里就是 skinCluster，在 Houdini 里就是 captured 的网格。
- **一个模型里有好几块网格**（身体、衣服、头发片、机械的几十个零件）没关系：节点先把它们按世界坐标拼成一张网格交给模型，权重算完按网格切回去，各写各的。
- **骨骼名是 bone_0、bone_1…**：这一版公开的权重（articulation-xl）**不带部位名**（作者的 mixamo 命名分支没有训练）。要接 Maya 的 HumanIK 或 Mixamo 重定向，得在 DCC 里按部位改名。这是这个项目现在最大的不方便，Lab2Shot 不去猜哪根是大腿——猜错了比不猜更糟。
- **影响骨骼数**（默认 4）：每个顶点最多由几根骨骼带动，就是 Maya 的 maxInfluences。权重最大的几根留下，其余归零后重新归一。
- **随机种子**（默认 12345，官方默认值）：骨头长错地方、少了尾巴或翅膀，换一个数字重算，挑一副最好的。
- **面数超过 5 万**：按官方的做法先降面（`fast_simplification`）再算骨架和权重，节点上会提醒一次；**权重最后铺回原来那张网格的每一个顶点**（用的是 UniRig 自己的 `reskin`），一个面都没少。
- 只要骨架不要网格：接「提取骨架」。要把绑定好的结果烘成每帧的点缓存：接「烘焙成模型」。

## 效果和局限（RTX 4090 / RTX 5090 上）

| 素材 | 网格 | 交出的骨骼 | 用时 | 显存峰值 |
|---|---|---|---|---|
| CesiumMan（Khronos 官方人形样本） | 3 273 点 / 4 672 面，1 块网格 | 19 根 | 4090 约 41–48 秒；5090 约 58 秒（新机器上第一次运行约 85 秒） | 3.7 GB |
| Fox（Khronos 官方四足动物） | 1 728 点 / 576 面 | 19 根 | 4090 约 55 秒 | 3.7 GB |
| BrainStem（Khronos 官方机器人，59 块网格） | 34 159 点 / 61 666 面 → 降到 50 000 面 | 22 根 | 4090 约 46 秒 | 3.7 GB |

- **和真值比**：CesiumMan 文件自带 19 根骨骼，UniRig 也给 19 根；关节位置的对称 Chamfer（J2J）**1.06 cm**，最大 2.04 cm，占这个人物身高 150.7 cm 的 **0.70%**。差得最多的是脖子那两根。
- **显存和镜头长短无关**：它看的是一块静止的网格，不是一段序列，所以 3.7 GB 是个定数，和面数关系也不大（网络在固定数量的采样点上算）。
- **局限**：
  - 骨骼没有部位名（见上）；
  - 骨架是生成的，同一个模型换种子会给出不同的骨架，根数也会变；细长的附属结构（尾巴、翅膀、辫子）有时会少给几根，作者建议换种子重试或手工补；
  - 权重阶段**吃骨架的质量**：骨架不好，权重跟着不好（作者原话：建议先把骨架改好再刷权重）；
  - 只做骨架和权重，**不做控制器、不做 IK、不做表情**；
  - 官方还没放出论文正文里那个 Rig-XL / VRoid 上训练的完整检查点，现在能用的是 Articulation-XL 2.0 上训练的这一个。
  - 作者已经放出后继项目 **SkinTokens**（把两个阶段合成一条自回归序列，官方说蒙皮精度提升 98%–133%、骨骼 17%–22%）。它和 UniRig 输入输出完全一样，将来接进来就是「绑定」家族里多一个节点，不用动核心。

## 团队

清华大学计算机系（胡事民组）和 Tripo（VAST AI Research）一起做的，SIGGRAPH 2025（TOG）。同一批人后来放出了它的后继 SkinTokens：把骨架和蒙皮合成一条自回归序列，官方说蒙皮精度提升 98%–133%、骨骼 17%–22%。

## 模型下载和安装

- `lab2shot ext install unirig`，下载：
  - 锁定版本的官方代码（提交 6793c664）；
  - 独立 Python 环境（Python 3.11 + PyTorch 2.8 + CUDA 12.8，含 spconv、torch_scatter、torch_cluster、FlashAttention 2.8.3 的官方预编译轮子）；
  - 两个权重（Hugging Face VAST-AI/UniRig）：骨架 1.4 GB、蒙皮 4.6 GB；
  - facebook/opt-350m 的 `config.json`（只要这一个文件：骨架网络按它的结构建网络，权重来自 UniRig 自己的检查点）。
- **不装 Blender（bpy）**。官方的命令行用 Blender 读模型文件、写 FBX，Lab2Shot 两头都用自己的 USD，所以那一层不需要；顺带也避开了 240 MB 的依赖。
- 装好后完全离线，不连网。

## 许可证说明

**可以商用，但有一处要注意。** UniRig 自己的代码和两个权重都是 MIT（Hugging Face 模型卡写明 MIT）。**仓库里自带的形状编码器 `src/model/michelangelo` 是 GPL-3.0**，两个阶段都用它；Pointcept 点云 transformer 是 MIT。GPL-3.0 不限制拿结果去做商业镜头，它限制的是再分发这份软件本身——Lab2Shot 只在本机跑、不分发它，所以交付物可以商用。其余依赖：spconv Apache-2.0、FlashAttention BSD-3-Clause、Open3D MIT、trimesh MIT。训练数据 Articulation-XL 2.0 来自 Objaverse-XL，里面每个模型有各自的许可。

## 参考

- 论文：One Model to Rig Them All: Diverse Skeleton Rigging with UniRig，https://arxiv.org/abs/2504.12451
- 项目页：https://zjp-shadow.github.io/works/UniRig/
- 代码：https://github.com/VAST-AI-Research/UniRig（本版锁在 6793c664）
- 权重和训练数据：https://huggingface.co/VAST-AI/UniRig 、https://huggingface.co/datasets/Seed3D/Articulation-XL2.0
- 后继项目 SkinTokens：https://github.com/VAST-AI-Research/SkinTokens
