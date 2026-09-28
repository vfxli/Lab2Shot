+++
team = "捷克理工大学（CTU）视觉识别组（VRG）"
people = "Jonáš Šerých, Jiří Matas"
paper = "Segmentation-Guided Homography Estimation for Long-Term Planar Tracking（ECCV 2026）"
paper_url = "https://arxiv.org/abs/2602.19624"
website = "https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/"
repo = "https://github.com/serycjon/WOFTSAM"
year = 2026
+++

## 这是什么

近来最先进的分割类方法能给出高质量、长时稳定的分割遮罩，但它们不估计平面跟踪要的那个几何表示——精确的 8 自由度单应位姿。论文把这一长处用到平面物体上，提出 SAM-H：一条免训练的流程，从分割遮罩的轮廓估计单应；用在 SAM 2 的遮罩上时，它在 PlanarTrack 基准的 p@5 指标上大幅领先（+18.4pp）。作者进一步指出，基于分割和基于对应关系的单应估计互为补充，于是提出 WOFTSAM，在 PlanarTrack 和 POT-210 上都超过以往所有方法。

在 Lab2Shot 里，这个节点叫「WOFTSAM 平面跟踪」：在一帧上标出平面的四个角，整段镜头逐帧交出这四个角和平面的单应矩阵。只限研究使用（CC BY-NC-SA 4.0）。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一段视频或一个画面文件夹，加起始帧上平面的四个角点——`demo.py` 里的 `init_coords`（4 个点，第一行是 x），
  `track_function(sam_predictor, conf, frames, init_coords, seq_name)` 里角点同样是**参数**，不是一路输入；
  `--config` 选 tracker 配置。
- **给**：每帧的平面位姿——单应矩阵 `output_H` 和随之而来的四个角；
  它内部三步之一会找到平面：从模板出发的光流、按 SAM-H 位姿预扭曲后的光流、或 SAM-H 的位姿本身。

**我们怎么接的**

- 「RGB」= 上游那段素材；「四个角」**参数** = 上游的 `init_coords`（艺术家在视图里框出来），起始帧就是画这四个角的那一帧；
  「处理分辨率」= 送进去的长边（按 8 的倍数取整，它内部的 RAFT 要求）。
- 「四个角」输出 = 上游每帧那四个角的位置和可见性，单应矩阵随平面一起带在 2D 跟踪点数据里。
- **不一样的一点**：上游把「这一帧是三步里哪一步找到的」丢掉了，这里把它那一段逐帧的处理抄过来读出这一项，
  当作每个角的把握程度（1 光流找到 / 0.5 SAM-H 找回 / 0 两者都没确认）；
  起始帧之前的帧是把镜头倒过来跟的（见 `adapters/woftsam/worker.py`）。

**出处**：简介来自 `third_party/woftsam/repo/README.md`（标题「Segmentation-Guided Homography Estimation for Long-Term Planar Tracking」）
和其中的「This repository provides the code for SAM-H and WOFTSAM achieving a state-of-the-art performance on POT-210 and PlanarTrack planar object tracking benchmarks.」）；
输入输出依据 `repo/demo.py` 和 `adapters/woftsam/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → WOFTSAM 平面跟踪 →「2D 跟踪点输出设置」（格式选 CornerPin）→「输出」。
- **框平面**：显示这个节点，在 2D 视图里拖一个框当作平面，再把四个角（黄色小方块，标着 1–4）分别拖到平面真正的角上。
  框在哪一帧，就从哪一帧开始跟，前后都会跟过去（前面的帧倒着跟）。选平面整个看得见、没被挡住、比较正对镜头的一帧。
  重新拖一个框会替换原来的；右键框里删掉。
- 输出「四个角」：四个 2D 跟踪点（corner_1…corner_4，按框的顺序），每帧的位置，外加平面的单应矩阵（从起始帧到这一帧）。
  - 「2D 跟踪点输出设置」选 **CornerPin**：写一个 CornerPin2D 节点（名字.nk，在 Nuke 里 File → Import Script，
    或者打开文件复制粘贴进节点图）：to1–to4 每帧一个关键帧，from1–from4 是起始帧的四个角。把换上去的画面接到它上面，
    就贴到了平面上；勾上 invert 就是把平面稳定下来（清理、补画用）。
  - 也能写 3DEqualizer 的点文件或 CSV。
  - 接「跟踪点转 3D」（再接一个深度图和相机）：四个角变成相机世界里的三维位置，在 Houdini / Maya 里放一张卡片。
- **挡住和找回在结果里看得见**：2D 视图里平面的轮廓，看得见的帧是黄色实线；没能确认平面的帧（被挡住、出画、变化太大）
  是红色虚线并写着「没能确认平面」，这些帧的四个角标成不可见（位置是估计的）。节点的提示里列出哪些帧是光流跟丢、
  由 SAM 2 的遮罩重新找回的，哪些帧没能确认；数据包的 `plane` 里也记着。交付前重点检查这些帧。
- 「处理分辨率」：留空按原尺寸跟点（每边对齐到 8 的倍数）；4K 素材显存不够时填 1920。画面长边超过 1920 时，
  留空会自动按 1920 封顶（更大的原生分辨率没有验证过）；要用更大的原尺寸就手动填。

## 效果和局限

RTX 4090 上（没有平面的标准答案，所以用两种办法量：同一个平面从最后一帧倒着再跟一遍、两次的四个角比；
和 AllTracker、CoTracker3 把四个角当点来跟的结果比。误差是四个角的平均距离，原图像素）：

- 手持、边走边拍 1080×1920，墙上的告示（有纹理的平面），90 帧：89 帧确认看得见；和 AllTracker 相差中位数 2.9 像素
  （90% 的帧在 4.4 像素以内），和 CoTracker3 相差 4.7 像素（两个点跟踪器之间自己也差 2.4 像素）；来回跟一遍回到起点差 5.8 像素。
  其中两帧光流跟丢，由 SAM 2 的遮罩找回。每帧 0.41–0.45 秒，显存峰值 10.1 GB。
- 固定机位跳舞 864×480，舞者身后的墙（从镜头中间开始，前后都跟）：和 AllTracker 相差中位数 4.9 像素；
  每帧 0.18–0.31 秒，显存 1.6 GB。
- **做得不好的**：
  - 手持镜头里一大片重复的蓝色墙板（人从下面走过、挡住下半边）：中途丢了，后面 60 帧都是「没能确认」（红色虚线），
    位置和 AllTracker 差几十到几百像素——结果里标出来了，但就是跟不上。
  - 固定机位镜头里几乎没有纹理的灰地面（脚踩进来）：大部分帧和 AllTracker 差 6 像素左右，最后几帧四个角缩成一团（标成没能确认）。
  - 固定机位镜头里墙的上半（几乎没纹理，舞者的头伸进来）：连续十几帧四个角歪了最多 48 像素，**却没有标出来**（光流的检查通过了），
    之后自己回到正确位置。所以「看得见」只是 WOFTSAM 自己的判断，不是保证。
- 结论：屏幕、招牌、海报这类有纹理的平面效果很好（和最好的点跟踪器差 2–5 像素，还能出 CornerPin）；
  大片单色、重复纹理的墙和地面不可靠，先用点跟踪（AllTracker）看看。

已知局限：
- 只跟**平的**物体：弯的屏幕、飘动的旗子、人脸不适合，用点跟踪（CoTracker3、AllTracker）。
- 平面整个出画、或者被完全挡住很久，找回时 SAM 2 可能先找到另一个长得像的平面（同一面墙上的另一块玻璃）。
- 四个角要按顺序（顺时针或逆时针）放在平面的角上；四个角交叉会被拒绝。
- 所有帧留在内存里：1080p 每 100 帧约 1.3 GB（外加 SAM 2 的每帧约 12 MB）。

## 团队

捷克理工大学（布拉格）视觉识别组 Jiří Matas 教授组，Jonáš Šerých 的博士工作。之前的 WOFT（加权光流平面跟踪，2023）
是同一组的工作，WOFTSAM 在它的基础上加上 SAM 2 做长时间的重新检测。

## 模型下载和安装

- 自动安装：`lab2shot ext install woftsam`。锁定 GitHub 仓库的一个版本，和作者的 SAM 2 分支（遮罩在平面不见时不更新记忆）
  一起放好，不 pip 安装；独立的 PyTorch 环境（和其他扩展共用下载缓存）。
- 下载：SAM 2.1 Hiera tiny（156 MB，Meta，Hugging Face 锁定版本），DINOv2 ViT-S/14 带 register（88 MB，SAM-H 分清四个角用）
  和 DINOv2 的网络代码；WOFTSAM 自己的加权 RAFT 权重（22 MB）在仓库里。都不需要申请权限。

## 许可证说明

WOFTSAM 的代码和它的加权 RAFT 权重是 CC BY-NC-SA 4.0：只能用于研究等非商业用途，改动后要用同样的许可发布，署名作者。
仓库里的 RAFT 部分是 BSD-3；用到的 SAM 2.1 tiny 和 DINOv2 是 Apache-2.0。整体按**非商用**对待。

## 参考

- 论文：https://arxiv.org/abs/2602.19624
- 项目主页：https://cmp.felk.cvut.cz/~serycjon/WOFTSAM/
- 代码：https://github.com/serycjon/WOFTSAM

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
