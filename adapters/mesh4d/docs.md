+++
team = "牛津大学 VGG（视觉几何组）+ Naver Labs Europe"
people = "Zeren Jiang, Chuanxia Zheng, Iro Laina, Diane Larlus, Andrea Vedaldi"
paper = "Mesh4D: 4D Mesh Reconstruction and Tracking from Monocular Video"
paper_url = "https://arxiv.org/abs/2601.05251"
website = "https://mesh-4d.github.io/"
repo = "https://github.com/jzr99/Mesh4D"
year = 2026
+++

## 这是什么

上游自己的话（论文摘要）：Mesh4D 是一个前馈模型，从单目视频解出 4D 网格。给一段动态物体的视频，它解出这个物体完整的三维形状和运动，运动表示成一个形变场。核心是一个紧凑的隐空间，一次编码整段动画：训练时由训练物体的骨架结构引导，给出合理形变的先验，推理时不需要骨架；编码器用时空注意力，整体形变的表示更稳。在此之上训练一个隐空间扩散模型，以输入视频和第一帧解出的网格为条件，一次预测出整段动画。那份第一帧网格来自上游仓库里的腾讯 Hunyuan3D-2.1 图生三维。

在 Lab2Shot 里，它交出的每一帧顶点数、面数、顺序完全一致，正是 DCC 里说的点缓存（Houdini 的 point cache、Maya 的 Alembic 缓存、USD 里逐帧写的 points）。它不解相机、也不知道物体多大：输出归一化在 1 个单位以内（包围盒最长边 0.9），节点把它换成 Y 轴向上、底面落在地面（Y=0）、左右前后居中，再乘节点上的「尺度」变成厘米。

## 输入输出

**官方要什么、给什么**

- 吃：**抠好的 RGBA 画面**。官方 `infer.py` 逐帧 `Image.open(路径).convert("RGBA")`，
  是 RGB 的话自己用 `rembg` 抠一张出来（`third_party/mesh4d/repo/hy3dshape/infer.py`）；
  数据集类把它拆成 `batch['image']` 和 `batch['mask']` 两样送进形变管线
  （`hy3dshape/hy3dshape/pipelines_video_newvae_all_nonalign_infer.py`）。
- 给：`registered_gen_mesh`——先由 Hunyuan3D-2.1 从一帧生成、再配准过的那一份静止网格；
  `deformed_verts_gen`——形变网络把那一个网格的顶点推到每一帧的结果。拓扑不变，就是点缓存。
  **没有相机，也没有真实尺寸。**

**我们怎么接的**

- 「图像」口 = `batch['image']`，「前景遮罩」口 = `batch['mask']`（官方的输入：官方读的就是 RGBA 的那条通道）。
- 「静止网格」口 = `registered_gen_mesh`，「网格」口 = `deformed_verts_gen`：一一对上。
- 「尺度」参数是 Lab2Shot 这一侧的：上游的结果没有真实尺寸，要摆进场景就得给一个尺度。

出处：简介来自论文摘要（arXiv 2601.05251：「We propose Mesh4D, a feed-forward model for monocular 4D
mesh reconstruction. Given a monocular video of a dynamic object, our model reconstructs the object's complete
3D shape and motion, represented as a deformation field」），仓库 `README.md` 是同一个标题；
输入输出依据 `third_party/mesh4d/repo/hy3dshape/infer.py` 和
`hy3dshape/hy3dshape/pipelines_video_newvae_all_nonalign_infer.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「网格序列 · Mesh4D」就是这一条）：
  「读取序列」→「BiRefNet 抠像」→「Mesh4D 网格序列」→「USD 输出设置」/「Alembic 输出设置」→「输出」。
- 「前景遮罩」是**必须接**的，不是可选：Mesh4D 要先按遮罩把物体裁出来、居中、叠到白底上。
  画面里只有一个主体时用「BiRefNet 抠像」最省事（它自己找主体）；要按提示词挑物体就用「SAM 3 视频分割」。
  有一帧遮罩全黑，节点会停下来并说是哪一帧。
- **一次只看 6 帧**。更长的镜头按 6 帧一段接着算，整段共用第一帧生成的那一个网格，
  所以**拓扑和点数整段不变**；但每一段的形变是各自采样出来的，接缝那一帧动作可能跳，
  节点算完会把接缝的帧号列出来。帧数不是 6 的倍数时，最后一段取的是整段最后 6 帧
  （和上一段有重叠，重叠的帧按先算出来的那一份留着）——每一帧都来自一个真正的 6 帧窗口，不是凑的。
- 少于 6 帧提交前就被拒绝（这是方法本身的前提，帧不够变不出来）。
- 「面数」默认「原样」，就是官方脚本的做法（一个面都不减）。43 万面 × 帧数的点缓存很大
  （60 帧约 155 MB 的顶点数据），视图里转起来沉、写进 Alembic 也不小；
  两档减面用的是官方自己那个减面函数，在形变**之前**减一次，整段共用减完的拓扑，
  不是逐帧抽稀。选了就会在节点上说明。
- 「静止网格」那个口交的是生成出来的那一个网格本身（没有形变），拿去重拓扑、展 UV 用这一份。
- 「尺度」：量出真实物体最长的那一边，填它的厘米数。默认 100（1 单位 = 1 米，物体约 90 厘米高）
  只是个占位，节点每次算完都会提醒一句。

## 效果和局限

- **RTX 4090 上（作者放出的 `DATA/in-the-wild/blooming_rose` 叠白底后的 6 帧 1080×1080，
  「标准」+「原样」，整条模板链）**：

  | 这一步 | 做几次 | 用时 | 显存 | 结果 |
  |---|---|---|---|---|
  | 装载两个模型 | 每个 worker 一次（之后常驻） | 约 60 秒 | 生成 7.0 GB + 形变 9.3 GB 常驻 | 形变权重 9.1 GB（fp16） |
  | 生成网格（Hunyuan3D-2.1） | 整段一次 | 约 60 秒 | 8.2 GB | 17.9 万点 / 35.8 万面 |
  | 解形变（50 步） | 每 6 帧一段 | 14 秒 | 11.3 GB | 6 帧点缓存，拓扑不变 |

  整个任务（BiRefNet 抠像 + Mesh4D）**153.3 秒**，整卡显存峰值 **21.5 GB**（PyTorch 侧 16.3 GB）。
  6 帧的一段约 150 秒；60 帧的一段约 280 秒（生成只做一次，形变 10 段）。
- **效果**：玫瑰从花苞到开放，6 帧的网格上花瓣确实一层层张开、萼片和叶子跟着动。
  交出来的 USD 里 `/shot/mesh4d` 有 6 个时间采样、每帧 178825 个点、拓扑完全一致，
  帧与帧之间顶点平均移动 0.54–0.76 厘米（尺度 100）；包围盒从地面 0 到 89.65 厘米高，左右前后对称，
  也就是站在地面上、摆在原点。
- **每次跑出来的网格不完全一样**：形状是扩散采样生成的，换个遮罩（比如 BiRefNet 的边缘和数据集自带的
  alpha 有出入）点数就会变（同一段素材，用数据集自带的 alpha 是 21.6 万点，接 BiRefNet 是 17.9 万点）。
  要复现就固定「随机种子」并保持上游不变。
- **第一帧决定一切**：形状是 Hunyuan3D-2.1 **生成**的，不是从多视角**解算**出来的。
  画面里看不见的那一面是模型编出来的；第一帧生成得不像，后面几帧只会跟着不像。
  这也是为什么它不适合做「测量」，适合做代理几何、动态遮挡物、参考。
- **一次只有 6 帧**，长镜头有接缝（见上）。
- **不解相机、没有真实尺度**，也不知道物体在画面里的三维位置：交出来的是一个摆在原点的物体。
- **一次只解一个物体**：画面里有两个主体就要分别抠出来各算一次。
- 机位一动就不可靠：它没有相机模型，镜头的转动会被当成物体的形变吸收进网格里。
- **不出贴图**：官方还能给一张 Hunyuan3D-2.1 生成的 PBR 贴图
  （`hy3dpaint`，要另外 7 GB 权重、还要编 CUDA 光栅化器和 C++ 补洞器），节点没有接这一步，只交几何——论文的贡献也是几何（4D Mesh Reconstruction and Tracking）。
- 作者放出的去噪网络是 **24 GB 的训练存档**（带优化器状态）。安装时会把它压成一份 9.1 GB 的
  fp16 推理权重（和官方脚本喂给网络的是同一批张量），所以每次计算不用再读 24 GB。
- 官方 `requirements.txt` 漏写了四个依赖（`timm`、`munch`、`plyfile`、`cython`）和 `torch_cluster`，
  少一个都起不来；扩展包自己的 `requirements.txt` 里补齐并注明了。

## 团队

牛津大学 VGG（Visual Geometry Group）Andrea Vedaldi 组，和 Naver Labs Europe 的 Diane Larlus 合作。
VGG 是做视觉几何和三维解算的老牌实验室（VGGT、Track-Anything 这条线的很多工作出自这里）。
Mesh4D 2026 年 1 月放出第一版，是这条线上第一篇把「生成一个网格 + 学一个形变场」合起来做 4D 的工作。
基准和可视化代码作者写着 coming soon，仓库里还没有。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install mesh4d`。约 45 GB（环境约 12 GB + 权重约 36 GB）。
- 权重四份：作者 Google Drive 上的形变 VAE（3.3 GB）和去噪网络（24 GB），
  Hugging Face 上腾讯 Hunyuan3D-2.1 的生成网络和形状 VAE（8.1 GB），Meta 的 DINOv2-Large（1.2 GB）。
- 安装时在这个环境里编译 im2mesh 的 5 个 Cython 模块（不需要 CUDA 编译器），
  并把 24 GB 的训练存档压成推理权重。原仓库一个字都不改：它的脚本要的那棵目录树是用符号链接在旁边拼出来的。
- 环境按官方钉死：Python 3.10、PyTorch 2.5.1+cu124。**只支持 Ada 架构（RTX 4090）**
  （`.venv-ada`）：torch 2.5.1 没有 Blackwell（RTX 5090，sm_120）的核，
  要上 5090 得连 PyTorch、pytorch-lightning 和 torch_cluster 的轮子一起换，尚未验证。

## 许可证说明

- **仅限研究**，三条理由叠在一起，取最严的一档：
  1. **Mesh4D 仓库里没有任何许可证文件**，作者也没在别处声明授权条款。没有授权就是没有给出使用许可，
     只能当作论文附带的研究代码；两个权重（形变 VAE、去噪网络）同样没有任何声明。
  2. 它的推理代码整个是腾讯 **Hunyuan3D-2.1**（`hy3dshape` / `hy3dpaint`），受
     「Tencent Hunyuan 3D 2.1 社区许可协议」约束。该协议第一行就写明**不适用于欧盟、英国和韩国**，
     并禁止在这些地区使用它的代码、权重和**输出结果**；生成网格用的 `hunyuan3d-dit-v2-1` 权重同样是这一份许可。
     （顺带一提：Mesh4D 的作者在牛津，正好在被排除的地区里。）
  3. 环境里的 **PyMeshLab 是 GPL-3.0**。
- Hunyuan3D-2.1 的 NOTICE 里列出的第三方组件（Stable Diffusion 的 MIT + CreativeML Open RAIL++-M 等）
  各自的条款也要一并遵守；分发时要带上 “Tencent Hunyuan 3D 2.1 is licensed under the Tencent Hunyuan 3D 2.1
  Community License Agreement, Copyright © 2025 Tencent.” 这句话。
- DINOv2-Large 是 Apache-2.0，可商用；但整条链按最严的算。

## 参考

- 论文：https://arxiv.org/abs/2601.05251
- 项目页：https://mesh-4d.github.io/
- 代码：https://github.com/jzr99/Mesh4D
- Hunyuan3D-2.1（它的推理代码和生成网络）：https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- Consistent4D（`DATA/` 里那几段画面的来源）：https://github.com/yanqinJiang/Consistent4D
- Motion2VecSets（形变表示的前作）：https://github.com/VVeiCao/Motion2VecSets
