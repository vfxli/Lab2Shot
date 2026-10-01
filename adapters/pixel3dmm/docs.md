+++
team = "慕尼黑工业大学（TUM）视觉计算与人工智能组（Matthias Nießner 实验室），与伦敦大学学院合作"
people = "Simon Giebenhain, Tobias Kirschstein, Martin Rünz, Lourdes Agapito, Matthias Nießner"
paper = "Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction（ICLR 2026）"
paper_url = "https://arxiv.org/abs/2505.00615"
website = "https://simongiebenhain.github.io/pixel3dmm/"
repo = "https://github.com/SimonGiebenhain/pixel3dmm"
year = 2025
category = "face"
+++

## 这是什么

Pixel3DMM 做的是从单张 RGB 画面还原三维人脸：一组高度泛化的视觉 transformer 逐像素预测几何线索，用来约束三维可变形面部模型（3DMM）的优化。它取 DINO 基础模型的隐层特征，配一个专门的表面法线和 UV 坐标预测头；再用一套 FLAME 拟合的优化，从 UV 坐标和法线的预测解出 3DMM 参数。作者报告，在带表情的面部几何精度上它比最有竞争力的基线好 15% 以上。

在 Lab2Shot 里，我们拿它做面部动作：上游那五步在一个进程里连着执行，按整段镜头拟合，交出带表情形变的蒙皮头、每帧网格、它自己解出的那台相机，以及两张屏幕空间先验（法线图和 UV 坐标图）。和「SMIRK 面部动作」怎么选，见下面那张表。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`scripts/run_preprocessing.py --video_or_images_path` 收一个 .mp4 或一个画面文件夹（README:100、103），
  做裁切、面部关键点、分割并执行 MICA；再 `scripts/network_inference.py model.prediction_type=normals / uv_map`
  逐帧预测两张屏幕空间先验（README:109-110，脚本假定方形画面、先缩到 512 × 512，README:115）；
  最后 `scripts/track.py` 把 FLAME 拟合到整段（README:121），`iters` / `global_iters` 是逐帧和整段联合的优化步数（README:136）。
- **给**：两张预测——法线图和规范面部 UV 坐标图（README:116 写明**法线在 FLAME 坐标系里，不是相机空间**）；
  `track.py` 拟合出的 FLAME 参数（脸型、表情、下巴、脖子、眼球、眼皮）和整段共用的一台相机（含解出的 Focal Length）。

**我们怎么接的**

- 「RGB」口就是上游那个画面文件夹：worker 把每帧按 `00000.png` 这样链过去（不重编码），上游自己那五步依次执行，
  和上游脚本一样每步一个独立进程（`adapters/pixel3dmm/worker.py` 的 `run_step`、`steps.py`）。
- 「法线图」「UV 坐标图」= 上游两个 ViT 预测头那两张（`scripts/network_inference.py:146-154` 的 `output['normals']` / `output['uv_map']`）；
  「蒙皮角色」「网格」「相机」都来自 `track.py` 拟合出的那份 FLAME 和那台相机。
- **不一样的两点**：① 上游的法线在 FLAME 坐标系里，我们按 README:116 那句把它转成相机空间再交出去
  （`adapters/pixel3dmm/worker.py` 的 `write_maps`），和别的法线图节点一个约定；
  ② 上游是三条命令分三次跑，我们在一次解算里把五步依次跑完。
- 上游 tracker 一次最多 1000 帧、且要求每帧都找得到脸，这两条我们照搬：不足 16 帧在网页上提交前就拦下，超过 1000 帧在开始解算时就报错。

**出处**：简介抽自论文摘要前三句（arXiv 2505.00615：「We address the 3D reconstruction of human faces from a single RGB image…」）；
输入输出依据 `third_party/pixel3dmm/repo/README.md:100-121、:136、:115-116` 和 `adapters/pixel3dmm/nodes.py` 的 `Official`、`worker.py` 的模块说明和 `write_maps`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → Pixel3DMM 面部动作 → USD 输出设置 / Alembic 输出设置 / FBX 输出设置 → 输出，导进 Maya / Houdini 做面部动画参考、表情驱动或替换头的对位。模板「面部动作 · Pixel3DMM」就是这条链。
- **要求整段每一帧都有一张清晰的脸**：它先用 PIPNet 在每帧找脸，按整段的平均框裁出**一个固定的正方形裁切框**，再算。有一帧找不到脸（人出画、完全侧脸、被手挡住）就会停下来报错，让你把帧范围缩到脸一直在的那一段。
- **一次最多 1000 帧**（上游 tracker.py 写死的上限）；不足 16 帧也算不了（第二阶段一次要取 16 帧一起算）。少于 16 帧在网页上就会拦下。
- 「精度」三档：快 / 标准 / 精细，改的是迭代步数（逐帧 100 / 200 / 400 步，联合 1500 / 5000 / 10000 步）。标准就是官方默认。显存三档一样，时间成倍差。
- Focal Length：不填的话它自己解（整段一个 Focal Length，联合优化时一起解）；填了「已知 Focal Length」（或接一条 Focal Length 进来），就用你给的那个并锁住不再动——这时它只解头的位置和姿态。节点上写着这一次的 Focal Length 是谁给的。
- 输出：
  - **人物**：带 102 个表情形变（100 个 FLAME 表情 + 左右眼皮）和五根骨头（头、脖子、下巴、两个眼球）的蒙皮头，表情权重做成动画，DCC 里能继续调；
  - **网格**：每帧的精确网格（点缓存），和画面贴得最紧的那一份；
  - **相机**：它解出来的那台相机——**一台固定相机**，头在它前面动。Focal Length 是解出来的；镜头中心跟着裁切框走，所以脸不在画面正中时，交付的相机带一个镜头中心偏移（USD 的 aperture offset），Nuke / Maya 里照样对得上；
  - **法线图**：相机空间的单位法线（和 MoGe、Sapiens2 同一个约定），只在裁切框那一块有值；
  - **UV 坐标图**：FLAME 官方的 UV 展开，0–1 两通道，只在裁切框那一块有值。Nuke 里用 STMap 贴图，Houdini / Maya 里做纹理传递——它自己就给出这一张，不用再接「规范坐标转 UV」。

## 效果和局限

- **实测（RTX 4090 和 RTX 5090，sh030 面部特写 772×855）**：见下面「实测数字」。
- **上游两个还没修的已知问题**（原仓库 issue 里作者确认过，我们没有改动算法）：
  1. **头顶偏高、脖子姿态不准**。头发以下的脸部很准，但头顶的轮廓常常比真人高一点，脖子的转动也不可靠——做替换头或对位时，脸用它的，头顶和脖子要人工核对。
  2. **法线图和 UV 图在背景上有网格状伪影**。脸以外的区域（背景、头发、衣服）这两张图本来就没有意义，我们只把裁切框那一块交出来，但框里的背景部分仍然会有网格纹路；用这两张图时配合遮罩，不要整块拿去用。
- 解出来的是头和表情，不含头发、牙齿；眼球的转动它解，眼皮用两个形变。
- 用骨骼 + blendshape 还原网格时，不含 FLAME 的姿态修正形变，也不含表情引起的关节微小位移，和原始网格相差不到 1 毫米（和 SMIRK 一样）。
- 裁切框是整段一个、固定的：脸在画面里移动很大的镜头，框会变得很大，脸在 512×512 里就变小，精度会掉。这类镜头先分段。

## 实测数字

RTX 5090，sh030 面部特写（772×855，整段 113 帧），裁切框 676×676：

| 精度 | 整段用时 | 平均每帧 | 显存峰值 |
|---|---|---|---|
| 快 | 414 秒 | 3.7 秒 | 19.8 GB |
| 标准（默认） | 706 秒 | 6.2 秒 | 19.8 GB |
| 精细 | 1291 秒 | 11.4 秒 | 19.8 GB |

- **显存峰值在「面部分割」那一步**（facer / FaRL），19.8 GB，和镜头长短、精度档都没关系。24 GB 的卡能跑，
  但这一步跑的时候这张卡基本占满了，别的任务要排在它后面。其余四步都小：找脸 0.1 GB、MICA 0.8 GB、
  两张预测图 2.8 GB、拟合 2.7 GB。
- **时间里有一大块和帧数无关**：同一条镜头「快」档 20 帧用 339 秒、113 帧用 414 秒——多出来的 93 帧只花了
  76 秒（0.8 秒/帧），其余 322 秒是五个步骤各自加载模型、torch.compile 和整段联合优化的固定开销。
  所以镜头越长，平均每帧越便宜；上面表里的「平均每帧」是按 113 帧这一条算的。
- Focal Length 是它自己解出来的：这条镜头解出 2426 像素（772 宽的画面，约合全画幅 113 mm）。镜头中心解在
  (365, 395)，画面中心是 (386, 428)——差 21 / 32 像素，交付的相机带着这个偏移。
- 4090 和 5090 都跑通（两张卡的 kernel 都编了）。

**肉眼核对**：把解出来的网格按交付的相机投回原画面，眼眶、鼻梁、嘴角、下巴、脖子都落在对的地方；
头转动的那几帧也跟得上。

## 和「SMIRK 面部动作」怎么选

两张模板卡都标「非商用」，选哪一张看这一条：

| | SMIRK 面部动作 | Pixel3DMM 面部动作 |
|---|---|---|
| 误差（NeRSemble 单视角带表情，Chamfer L1） | 2.276 mm | **1.659 mm** |
| 速度 | **0.13 秒/帧** | 约 6.2 秒/帧（标准档） |
| 时序 | 每帧单独算，会抖 | 整段联合优化 + 时序平滑，稳 |
| 表情 | 50 个 FLAME 表情 + 2 眼皮 | 100 个 FLAME 表情 + 2 眼皮，另有脖子和眼球转动 |
| 相机 | 正交相机换算成的针孔，Focal Length 要你填 | 自己解出 Focal Length 和镜头中心 |
| 对素材 | 找不到脸的帧沿用最近一帧，能出结果 | 有一帧找不到脸就停下来 |
| 帧数 | 不限 | 16–1000 帧 |

**要快、要先看个大概、素材不稳**：SMIRK。**要交付、要准、要不抖**：Pixel3DMM。

## 团队

慕尼黑工业大学 Matthias Nießner 的实验室。第一作者 Simon Giebenhain 之前做过 NPHM 和 MonoNPHM（神经参数化头模），Tobias Kirschstein 做过 NeRSemble（多视角人脸数据集和评测榜）和 DiffusionAvatars。这一条线上的工作是目前单目面部精度最高的一批。

## 模型下载和安装

- 自动安装：`lab2shot ext install pixel3dmm`。下载原始仓库和三个预处理仓库（facer、MICA、PIPNet，都钉死版本）、独立 Python 3.10 环境（torch 2.9 + CUDA 13.0），编译 pytorch3d 和 nvdiffrast，再下载权重：Pixel3DMM 的两个预测网络 2.1 GB + 1.4 GB、MICA 身份网络 479 MB、InsightFace antelopev2 344 MB、PIPNet 关键点 47 MB，外加几个小模型。一共约 4.5 GB，环境加权重约 13 GB。编译要十几分钟。
- 需要手动下载：FLAME 面部模型。到 https://flame.is.tue.mpg.de 注册登录，在 Download 页面下载「FLAME 2020」（FLAME2020.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，后台管理页「扩展包」里的「手动下载」写着这个文件夹在哪），Lab2Shot 会自动解压并找到里面的 `generic_model.pkl`。
- 安装和上游官方脚本的两处不同（都不改原仓库）：
  1. 官方的 `install_preprocessing_pipeline.sh` 用 SSH（`git@github.com`）克隆 facer / MICA / PIPNet，并把它自己的替换文件拷进那三个仓库里。Lab2Shot 改用 https 按提交号钉死克隆，再用符号链接**在旁边**拼出上游期望的那棵目录树（`third_party/pixel3dmm/codebase/`），四个检出一个字节都不改。
  2. pytorch3d 从源码编译时**去掉了它的点渲染器 pulsar**：pulsar 的显式模板实例化在 CUDA 13 的 nvcc 下生成不出主机端符号，`_C` 链接不过；Pixel3DMM 用到的是 knn / load_obj / Meshes，和 pulsar 无关。只保留 pulsar 头文件里的两个常量（`_C.EPS`、`_C.MAX_UINT`，pytorch3d 自己导入时要读）。
- 两张显卡都能跑：pytorch3d 和 nvdiffrast 的 kernel 都按本机的 sm_89（4090）和 sm_120（5090）编译。

## 许可证说明

- **不能商用。** Pixel3DMM 的代码和两个预测网络权重都是 CC BY-NC 4.0：只能用于研究和评估。
- 必须用 FLAME 面部模型（FLAME 2020）才能算，许可仅限非商用科研，需要注册后自己下载，禁止再分发。
- 脸型先验用马普所的 MICA：专有代码，只许有许可的非商用使用；权重同样非商用。它的人脸检测用 InsightFace 的 antelopev2，仅限非商用研究。
- 光栅化用 NVIDIA 的 nvdiffrast（NVIDIA Source Code License，非商用研究）。
- 其余可商用：PIPNet 关键点 MIT，facer / FaRL 面部分割 MIT，pytorch3d BSD-3-Clause，chumpy MIT。

## 参考

- 论文：https://arxiv.org/abs/2505.00615
- 项目主页：https://simongiebenhain.github.io/pixel3dmm/
- 代码：https://github.com/SimonGiebenhain/pixel3dmm
- NeRSemble 单视角面部精度榜：https://kaldir.vc.cit.tum.de/nersemble_benchmark/benchmark/svfr
- FLAME：https://flame.is.tue.mpg.de
- MICA：https://github.com/Zielon/MICA
