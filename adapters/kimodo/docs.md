+++
team = "NVIDIA 研究院（多伦多 AI 实验室 / 空间智能实验室）"
people = "Davis Rempe, Mathis Petrovich, Ye Yuan, Haotian Zhang, Xue Bin Peng, …, Jan Kautz, Simon Yuen, Sanja Fidler"
paper = "Kimodo: Scaling Controllable Human Motion Generation（技术报告，2026）"
paper_url = "https://arxiv.org/abs/2603.15546"
website = "https://research.nvidia.com/labs/sil/projects/kimodo/"
repo = "https://github.com/nv-tlabs/kimodo"
year = 2026
+++

## 这是什么

上游自己的话（Overview）：Kimodo 是一个运动学动作扩散模型（kinematic motion diffusion model），在大规模（700 小时）、可商用的光学动捕数据上训练；它生成高质量的三维人体和机器人动作，控制方式是文字提示，外加一整套约束——全身姿态关键帧、末端位置和旋转、2D 路径、2D 路点。

在 Lab2Shot 里，它是「Kimodo 动作生成」：动画师只 K 关键姿势，中间的动作由 Kimodo 生成，关键帧保持原样；也可以再加一句英文描述，说这段动作是什么、什么风格。不接动画时它还能只按一句文字生成一整段（上游 `constraint_lst` 自己写的「unconstrained generation」）。代码和七个权重里的六个可以商用，SMPL-X 那一档只限研究（见下面「许可」）。

## 输入输出

**官方要什么、给什么**

- 吃：一句（或几句）文字描述，加一套可选的约束。命令行是 `kimodo_gen`（或 `python -m kimodo.scripts.generate`），
  关键参数：`prompt`（必填）、`--model`、`--duration`（秒）、`--num_samples`、
  `--constraints`（约束文件，网页 demo 存出来的那种）、`--diffusion_steps`、`--cfg_type` / `--cfg_weight`、
  `--no-postprocess`、`--seed`（`third_party/kimodo/repo/README.md`）。
  约束在代码里是 `constraint_lst`（`EndEffectorConstraintSet` / `FullBodyConstraintSet`，
  `kimodo/constraints.py`），官方列的约束种类有：全身姿态关键帧、末端位置和旋转、2D 路径、2D 路点。
- 给：一个 NPZ——`posed_joints [T,J,3]`、`global_rot_mats [T,J,3,3]`、
  `local_rot_mats [T,J,3,3]`、`foot_contacts [T,4]`、`smooth_root_pos [T,3]`、`root_positions [T,3]`、
  `global_root_heading [T,2]`。

**我们怎么接的**

- 「动画」输入口接的就是上游的约束：节点把接进来的骨骼动画按「关键帧」选出的那几帧转成 `constraint_lst`
  （`third_party/kimodo/repo/kimodo/postprocess.py`）。「提示词」是**参数**，不是口——
  上游的 `prompt` 本来就是一句文字。
- 「动画」输出口是 `local_rot_mats` 加根位置转成的骨骼动画，和官方吐的是同一份数据。
- 「模型」参数对应 `--model` 那七个权重、三副骨架（SOMA 30 关节四个、SMPL-X 22 关节一个、G1 机器人 34 关节两个；
  `third_party/kimodo/repo/kimodo/skeleton/definitions.py`），「去噪步数」= `--diffusion_steps`、
  「贴合关键帧」= `--cfg_weight` 那一路、「模型后处理」= `--no-postprocess` 的反面、「随机种子」= `--seed`。
- 上游还有、节点没有开口的：`foot_contacts`、`global_root_heading`、`smooth_root_pos`。

出处：简介来自 `third_party/kimodo/repo/README.md`；输入输出依据同一份 README、
`third_party/kimodo/repo/kimodo/postprocess.py` 和 `adapters/kimodo/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：「导入 USD」或「导入 FBX」（Maya 导出的角色，只导关键帧那几帧；骨架动画、蒙皮角色都行）→ **Kimodo 动作生成** → 「USD 输出设置」→「输出」。模板「动作补帧 · Kimodo」就是这套。
- **关键帧**：留空时，文件里动画有记录的帧就是关键帧；动画每帧都烘焙过的文件，在「关键帧」里写要保留的帧，如 `1001, 1012, 1030` 或 `1001-1100x8`。
- **关节映射**：骨架跟着「模型」走——默认的四个 Rigplay / SEED 权重是 NVIDIA 的 SOMA（30 个关节），SMPL-X 那一档是 SMPL-X 的 22 个身体关节（接 SMPL 人体不用猜关节），G1 机器人那两档是 Unitree G1 的 34 个关节（一条腿三个独立的髋关节轴，人物骨骼对不上，所以**只能用来生成、「动画」口是灰的**）。接上人物后，表格每一行显示自动猜到的人物关节，猜错了在下拉里选。Maya HumanIK、Mixamo、Unreal、SMPL 这些命名都认得；扭转骨骼、手指、面部骨骼不参与，**保持原来的动画不动**（Kimodo 本身也不动手指）。绑定姿势是 T-pose 还是 A-pose、关节怎么定向都没关系。
- **提示词**（可选）：一句英文，最好以「A person …」开头，比如「A person walks forward cautiously」。不写就只按关键帧补（Kimodo 官方基准里「无文字约束」的用法，关键帧命中和写了文字一样好），这时完全不需要文字模型。写了描述时，Lab2Shot 用 Kimodo 自己的文字编码器（LLM2Vec + Meta Llama 3 8B）在 CPU 上把这句话算成一个向量，**同一句话只算一次**，以后任何镜头用同一句都复用（Kimodo 自己的演示程序也是这样缓存的）。第一次约一两分钟，要占约 16 GB 内存。
- **关键帧精确**（默认开）：每个关键帧和原来的姿势完全一样；**脚锁定**（默认开）：Kimodo 判断脚着地的帧，用两骨 IK 把脚踝钉住。**模型后处理**（默认开）是 Kimodo 自己的脚滑清理。
- **随机种子**：扩散模型每次结果不同，同一个种子得到同一个结果；不满意换个数字重算。
- **地面**：模型默认人站在 y = 0 的地面上（Y 轴向上）。视频解出来的人物先接「自动落地」。
- **长镜头**：Kimodo 一次最长生成 10 秒、最好少于 20 个关键帧；更长的镜头 Lab2Shot 分段生成，相邻两段共用边界关键帧，后一段还贴着前一段最后 5 帧生成再交叉淡化（Kimodo 自己多段衔接的做法）。
- **帧率**：模型按 30 帧/秒生成，结果换算回人物自己的帧率。

## 效果和局限

- 速度：模型加载约 3 秒（之后常驻显存）；一段 10 秒的镜头约 4–8 秒（100 步；关键帧越密，分段越多），显存约 1.2 GB。
- 局限（官方说明）：不能保证贴合关键帧（Lab2Shot 的「关键帧精确」保证）；仍有一些脚滑；不认识场景（台阶、障碍物）；竖直方向控制弱（上楼梯）；卡通、违反物理的动作做不出来；动作风格是写实动捕，细腻的表演节奏（预备、过冲、停顿）未必对。

## 团队

NVIDIA 研究院多伦多 AI 实验室和空间智能实验室（Sanja Fidler、Jan Kautz 等），和 NVIDIA 的动画、机器人团队一起做的；同一批人还做了实时版 ARDY 和机器人动作生成 MotionBricks。训练数据来自 Bones Studio（Rigplay 动捕库）。

## 模型下载和安装

- 自动安装：`lab2shot ext install kimodo`，下载：
  - 锁定版本的官方代码，和它的动作后处理模块 MotionCorrection（C++，安装时编译）；
  - 独立 Python 环境（PyTorch 2.8 + CUDA 12.8）；
  - 七个 Kimodo 模型（Hugging Face，各 1.1 GB；Kimodo-SOMA-RP-v1.1 是默认，SMPL-X 那一档是受限仓库，要先申请访问）；
  - 文字编码器（只在写文字描述时用）：两个 LLM2Vec 适配权重（330 MB）和 Meta Llama 3 8B Instruct（16 GB）。Llama 3 在 Hugging Face 上要先申请：打开 https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct 同意 Meta 的条款，通过后再运行一次安装。没有它也能用，只是不能写文字描述。
- 装好后完全离线运行（Kimodo 自己默认会先连一个网上的文字编码服务，Lab2Shot 关掉了）。

## 许可证说明

**可以商用（一个权重除外）。** 代码 Apache-2.0；七个权重里六个（SOMA 四个 + G1 两个）是 NVIDIA Open Model License，允许商用；训练数据是 Bones Studio 授权的动捕。写文字描述时用到的 Meta Llama 3 是 Llama 3 社区许可证：可以商用，但月活超过 7 亿的公司需要另外向 Meta 申请，产品里要标注「Built with Meta Llama 3」，并遵守它的使用政策；LLM2Vec 适配权重 MIT。Kimodo 的 SMPL-X 版（Kimodo-SMPLX-RP-v1）也装了，「模型」里就能选，但它那一档的许可和别的六个不一样：NVIDIA Internal Scientific Research and Development Model License，只限研究，而且是受限仓库，要本人在 Hugging Face 页面申请访问。

## 参考

- 技术报告：https://arxiv.org/abs/2603.15546
- 项目主页和文档：https://research.nvidia.com/labs/sil/projects/kimodo/
- 代码：https://github.com/nv-tlabs/kimodo
- 模型卡：https://huggingface.co/nvidia/Kimodo-SOMA-RP-v1.1 、https://huggingface.co/nvidia/Kimodo-SOMA-SEED-v1.1
- 最佳用法和局限：https://research.nvidia.com/labs/sil/projects/kimodo/docs/key_concepts/limitations.html
- NVIDIA Open Model License：https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
