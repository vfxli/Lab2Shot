+++
team = "雅典国立技术大学（NTUA），与马普所智能系统研究所（MPI-IS）、希腊研究与技术基金会（FORTH）合作"
people = "George Retsinas, Panagiotis P. Filntisis, Radek Daněček, Victoria F. Abrevaya, Anastasios Roussos, Timo Bolkart, Petros Maragos"
paper = "SMIRK: 3D Facial Expressions through Analysis-by-Neural-Synthesis（CVPR 2024）"
paper_url = "https://arxiv.org/abs/2404.04104"
website = "https://georgeretsi.github.io/smirk/"
repo = "https://github.com/georgeretsi/smirk"
year = 2024
+++

## 这是什么

SMIRK 从单目画面解出三维人脸，面部几何能忠实还原极端、不对称和细微的表情。论文的路子叫「用神经合成做分析」：一个神经渲染模块拿预测网格渲出来的几何，加上输入画面上稀疏采样的像素，重新生成一张人脸。

在 Lab2Shot 里，它用来做逐帧的面部动作：官方那套 FLAME 参数（脸型、50 个表情分量、下巴、眼皮、头部转动）装成一副带骨骼（头、脖子、下巴、两个眼球）和表情形变的蒙皮头，另外交每帧网格和表情曲线，可以进 DCC。上游给的相机是正交的、不是一台相机，所以节点上没有相机输出口。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`demo.py` / `demo_video.py` 的 `--input_path`，一张画面或一段视频；
  `--crop` 打开时用 MediaPipe 裁出脸，`--render_orig` 把结果按原尺寸呈现。
- **给**：SMIRK 编码器出的 FLAME 2020 参数（`src/smirk_encoder.py`、`src/FLAME/FLAME.py`）——
  300 个脸型分量、50 个表情分量、下巴、眼皮、头部转动，加一台**正交**相机；由它们算出 5023 个顶点的网格。
  `--use_smirk_generator` 是它自己那个神经生成器，用来重画一张脸做可视化。

**我们怎么接的**

- 「RGB」口 = 上游 `--input_path` 那段素材，逐帧走上游 demo 的流水线；「裁切」参数 = 上游的 `--crop`
  （自动找脸：MediaPipe 找脸后按 1.4 倍裁成方形再缩到 224 × 224；整幅画面：整幅补成方形再缩）。
- 「蒙皮角色」= 官方那整套 FLAME 参数装成的蒙皮角色（脸型锁成整段中位数的静止网格 + 五根骨头的骨骼动画），
  「网格」= 每帧顶点，「表情曲线」= 50 个表情分量加两个眼皮。
- **不一样的两点**：① SMIRK 的相机是正交的，我们按「已知 Focal Length」「Filmback」把它换算成一台针孔相机（正交缩放换成距离），
  头才能摆进相机空间；**节点上没有「相机」输出口**，因为上游给的不是一台相机；
  ② `--use_smirk_generator` 那一步是它做可视化用的，我们不执行。

**出处**：简介来自 `third_party/smirk/repo/readme.md`（「SMIRK reconstructs 3D faces from monocular images with facial geometry that faithfully recover extreme, asymmetric, and subtle expressions.」）
和论文摘要里那句神经渲染模块（arXiv 2404.04104：「a neural rendering module that, given the rendered predicted mesh geometry, and sparsely sampled pixels of the input image, generates a face image」）；
输入输出依据 `repo/demo.py`、`repo/src/smirk_encoder.py`、`repo/src/FLAME/FLAME.py` 和 `adapters/smirk/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → SMIRK 解算 → USD 输出设置 / Alembic 输出设置 → 输出，导进 Maya / Houdini 做面部动画参考、表情驱动或替换头的对位。
- 「裁切」选「自动找脸」（默认）：用 MediaPipe 在每帧找脸，按官方演示的方式裁成正方形再算，脸在画面里多大都行。画面本身已经是裁好的大头特写时可以选「整幅画面」，但一般自动更准。
- Focal Length 尽量填（「已知 Focal Length」配合「Filmback」）：SMIRK 自己用的是正交相机，Lab2Shot 按 Focal Length 把头放到对应的距离上；不填时按全画幅 50 mm 镜头估算。Focal Length 只影响头离镜头的远近，头在画面上的位置和轮廓都对得上。
- **「相机」输出口**：放头用的那台针孔相机——原点、不动，焦距 = 「Focal Length」（不填按全画幅 50 mm），主点在画面中心。它不是上游解出来的（官方的 `outputs['cam']` 是 224 裁切上的弱透视三个数，上游也不吃相机），是我们把正交换成针孔时用的相机，交出来让三维视图透过它看背板、头和画面对得上（`nodes.py` 的 `official` 里登记在 `ours`）。节点没有「相机」输入口。结果留在相机空间；要把头摆进某台相机（ViPE / 3DE）的世界，后面接核心节点「相机空间转换」（`camera_space`）。
- 输出：带 52 个表情形变（50 个 FLAME 表情 + 2 个眼皮）的 USD 头，表情权重做成动画，DCC 里能继续调；另有 52 条表情曲线（CSV / .chan）。模板「面部动作 · SMIRK」就是这条链。
- 适合：脸够大、正脸到大半侧脸、光线正常的镜头。完全侧脸、脸很小或被手挡住时 MediaPipe 找不到脸（这些帧沿用最近一帧的裁切，结果仅供参考）。
- 每帧单独计算，没有时序平滑；脸型（长相）取整段镜头的中位数，保证整段是同一个人。

## 效果和局限

- RTX 4090 上（一段面部特写，772×855，113 帧）：约 15 秒（其中大部分是 CPU 上的 MediaPipe 找脸），显存峰值不到 0.6 GB。113 帧全部找到脸，网格叠回画面和五官、下巴、闭眼、张嘴都吻合。
- 表情系数逐帧估计，连续播放时会有轻微抖动，做动画时建议再平滑一遍。
- 只估计头部和表情，不含头发、牙齿和眼球转动（眼球固定朝前）。
- 用骨骼 + blendshape 还原网格时，不含 FLAME 的姿态修正形变，和原始网格相差不到 1 毫米（张嘴很大时约 4 毫米）。

## 团队

雅典国立技术大学 Petros Maragos 的实验室（George Retsinas、Panagiotis Filntisis），和马普所的 Timo Bolkart、Radek Daněček（FLAME 模型和 EMOCA 的作者）以及 FORTH 的 Anastasios Roussos 合作。这个团队之前做过 SPECTRE（口型准确的三维面部）和 EMOCA（带情绪的三维面部），SMIRK 是这条线上的新一代。

## 模型下载和安装

- 自动安装：`lab2shot ext install smirk`。下载原始仓库、独立 Python 3.10 环境（torch 2.8 + CUDA 12.8）和权重：SMIRK 预训练模型 134 MB（作者 Google Drive）、MediaPipe 面部动作关键点模型 3.6 MB。几分钟就能装好。
- 需要手动下载：FLAME 面部模型。到 https://flame.is.tue.mpg.de 注册登录，在 Download 页面下载「FLAME 2020」（FLAME2020.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，后台管理页「扩展包」里的「手动下载」写着这个文件夹在哪），Lab2Shot 会自动解压并找到里面的 `generic_model.pkl`。必须是 FLAME 2020：SMIRK 是按它训练的，FLAME 2023 的系数对不上。

## 许可证说明

- 不能商用。SMIRK 代码本身是 MIT，但它离不开 FLAME 2020 面部模型：FLAME 2020 只允许非商用科研，禁止再分发（只有 FLAME 2023 Open 是可商用的 CC-BY-4.0，但 SMIRK 用不了它）。仓库自带的 FLAME 遮罩、关键点嵌入和头部模板也来自 FLAME 官网。
- SMIRK 权重作者没有单独写许可，训练用了 LRS3、MEAD、CelebA、FFHQ 等非商用研究数据，只按研究用途使用。
- 裁脸用的 MediaPipe Face Landmarker（代码和模型）是 Apache-2.0，可商用。

## 参考

- 论文：https://arxiv.org/abs/2404.04104
- 项目主页：https://georgeretsi.github.io/smirk/
- 代码：https://github.com/georgeretsi/smirk
- 演示视频：https://www.youtube.com/watch?v=8ZVgr41wxbk
- FLAME：https://flame.is.tue.mpg.de
