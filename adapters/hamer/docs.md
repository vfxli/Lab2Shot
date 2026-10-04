+++
team = "加州大学伯克利分校（UC Berkeley），与密歇根大学、纽约大学合作"
people = "Georgios Pavlakos, Dandan Shan, Ilija Radosavovic, Angjoo Kanazawa, David Fouhey, Jitendra Malik"
paper = "Reconstructing Hands in 3D with Transformers（CVPR 2024）"
paper_url = "https://arxiv.org/abs/2312.05251"
website = "https://geopavlakos.github.io/hamer/"
repo = "https://github.com/geopavlakos/hamer"
year = 2024
+++

## 这是什么

上游自己的话（论文摘要）：HaMeR 从单目输入解出三维的手。它采用完全基于 transformer 的结构，精度和稳定性比以往的方法显著提高；关键在于同时扩大训练数据和网络容量——训练数据合并了多个带 2D 或 3D 手部标注的数据集，模型用的是大规模 Vision Transformer。论文《Reconstructing Hands in 3D with Transformers》。

在 Lab2Shot 里，它是「HaMeR 手部动作」，按官方演示的流程接：先用它自带的 ViTDet 检测器在画面里找人，再在每个人身上找左右手的关键点（ViTPose），然后 HaMeR 对每只手的特写小图估计手型和姿态。每只手各自成一个物体：网格、骨骼、每根骨骼相对父骨骼的旋转、蒙皮权重和 MANO 参数，落进我们的「蒙皮角色」类型，可以做成带骨骼的手部动画。

## 输入输出

**官方要什么、给什么**

- 吃：一个画面文件夹。官方 `demo.py` 收 `--img_folder`、`--out_folder`、
  `--body_detector`（默认 `vitdet`）、`--rescale_factor`（框放大倍数，默认 2.0）、`--batch_size`。
  **人是它自己检的**：按 `--body_detector` 建起 ViTDet-H，每帧取类别 0、分数大于 0.5 的框，
  再用 ViTPose 的手腕关键点定出每只手的框。
- 给：每只手一份 `pred_mano_params`（MANO 手部参数）和 `pred_vertices`（网格顶点），
  以及相机那一路的中间量（`hamer/models/hamer.py`）；关键点是 `vitposes['keypoints']`。

**我们怎么接的**

- 「RGB」口就是 `--img_folder`：和上游一样，**只有画面**。「手部框放大」就是 `--rescale_factor`。
- 「蒙皮角色」口是 `pred_mano_params` 转成的骨骼动画，网格就是 `pred_vertices`（蒙皮角色里的网格，没有单独的网格口），
  「2D 关键点」口是 `vitposes['keypoints']`。
- **「人物框」是可选输入口**：官方仓库的 `ViTPoseModel.predict_pose(image, det_results)`（`vitpose_model.py`）
  收的就是人框，官方 demo 只是先用 ViTDet 检出人框再喂给它。接了就按框找手、不再自己检人；不接就照官方 demo 的路自己检（同一个 ViTDet-H）。
- 节点上的「已知 Focal Length」「Filmback」对应上游用的那一个 Focal Length 数值，所以这里没有「相机」输入口。

出处：简介来自论文摘要（arXiv 2312.05251）和仓库 `README.md`；
输入输出依据 `third_party/hamer/repo/demo.py`、`hamer/models/hamer.py` 和 `adapters/hamer/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → HaMeR 手部动作 → USD 输出设置 → 输出，导进 Maya / Houdini 做手部动画参考、手部替身或和身体解算（SAM 3D Body、GVHMR）的结果拼在一起。
- **「人物框」可选**：不接时人由官方自己的 ViTDet 检测器找，画面里每个人的手都会解。手的编号跟着人的编号走（第 N 个人的左手 = 2N，右手 = 2N+1），整段镜头不变。
- 只想解某一个人：「ViTDet 人物框」→（在「选人」里点他）→ HaMeR 的「人物框」口，就只找他的手。另一条路是把别人从画面里去掉——「ViTDet 人物框」→「选人」→「人物框转遮罩」→「图像合成」（留下）→ HaMeR 的「RGB」口。
- Focal Length 尽量填（「已知 Focal Length」配合「Filmback」，也可以接一条 Focal Length 进来）：手离镜头的远近由 Focal Length 决定。不填时按全画幅 50 mm 镜头估算（HaMeR 自带的超长焦假设会把手放到约 100 米外，节点不用它）。
- **节点上没有「相机」的进出口**：官方只有一个 Focal Length（`demo.py` 的 `scaled_focal_length`），没有相机。结果留在相机空间；要把手放进某台相机（ViPE / 3DE）的世界、和全身结果对在一起，后面接核心节点「相机空间转换」（`camera_space`）——这一步在节点图上看得见。
- 「手部框放大」默认 2.0（官方演示的值）：手被裁得太紧、手指露不全时调大一点。
- 「2D 关键点」输出：找手用的 ViTPose+ 手部 21 点（手腕 + 五指的指节和指尖，带把握值），画面上的点（不是三维结果的投影），接「2D 跟踪点输出设置」交给 3DEqualizer / Nuke，也可以叠在画面上看解出来的手贴不贴。按人分组，一个人一组。
- 输出「蒙皮角色」是手的骨骼带动的网格（能在 DCC 里改动画），「网格」是逐帧变形的模型（模型的精确网格，点缓存）。模板「手部动作 · HaMeR」就是这条链。
- 适合：手在画面里至少几十像素、手指看得清的镜头。手太小、糊、被挡住或者握着物体时，手指姿态不可靠；每帧单独计算，没有时序平滑，手指会有细小抖动。

## 效果和局限

- RTX 4090 上（跳舞镜头，864×480，124 帧，一个人）：全流程约 72 秒（含加载模型），逐帧找手 + 解算约 0.07 秒/帧，显存峰值约 5.9 GB。左手找到 116 帧、右手 115 帧，投影回画面和手的位置吻合。
- 每只手单独估计深度图，同一个人的两只手、手和身体之间的前后关系不保证一致；要和身体对齐，最好填准 Focal Length，或在场景里按手腕位置对齐。
- 左手是把画面镜像后当右手算，再镜像回来（官方做法），左手文件里的参数已经换成 MANO_LEFT 的约定。
- 输出的骨骼蒙皮不包含 MANO 的姿态修正形变，用骨骼驱动时和原始网格有 1–2 毫米级的差别（极端弯曲处可到 1 厘米多）。

## 团队

伯克利 Jitendra Malik 和 Angjoo Kanazawa 的计算机视觉组，第一作者 Georgios Pavlakos（现在德州大学奥斯汀分校）。这个团队做过人体反求的一系列经典工作：HMR、SPIN、SMPLify-X、4D-Humans（HMR 2.0），HaMeR 就是把 4D-Humans 的思路用在手上。

## 模型下载和安装

- 自动安装：`lab2shot ext install hamer`。下载原始仓库、独立 Python 3.10 环境（torch 2.8 + CUDA 12.8，编译 mmcv / mmpose）和权重：HaMeR 模型 2.7 GB、ViTPose+-H 全身关键点 3.8 GB（都来自作者自己的 Hugging Face Space，和官方 6 GB 压缩包里的文件完全一致），共约 6.5 GB。网络好的话 20–30 分钟。
- 需要手动下载：MANO 手部模型。到 https://mano.is.tue.mpg.de 注册登录，在 Download 页面下载「Models & Code」（mano_v1_2.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，后台管理页「扩展包」里的「手动下载」写着这个文件夹在哪），Lab2Shot 会自动解压并找到里面的 `MANO_RIGHT.pkl`。

## 许可证说明

- 不能商用。HaMeR 代码本身是 MIT，但它离不开 MANO 手部模型：MANO 只允许非商用的科研、教学和艺术项目，禁止再分发，商用要找马普所（ps-license@tue.mpg.de）。
- HaMeR 权重作者没有单独写许可，训练用了 FreiHAND、InterHand2.6M（CC-BY-NC）等非商用数据，只按研究用途使用。
- 找手用的 ViTPose+-H：代码 Apache-2.0，权重从 MAE 预训练（MAE 权重 CC-BY-NC-4.0）。smplx 代码是马普所非商用许可（只用来读 MANO）。
- 作者的 Hugging Face Space 里也放了 MANO_RIGHT.pkl，Lab2Shot 不从那里下载：MANO 要从官网自己注册获取。

## 参考

- 论文：https://arxiv.org/abs/2312.05251
- 项目主页：https://geopavlakos.github.io/hamer/
- 代码：https://github.com/geopavlakos/hamer
- 在线演示和权重（作者的 Hugging Face Space）：https://huggingface.co/spaces/geopavlakos/HaMeR
- ViTPose：https://github.com/ViTAE-Transformer/ViTPose
- HInt 手部数据集：https://github.com/ddshan/hint
- MANO：https://mano.is.tue.mpg.de
