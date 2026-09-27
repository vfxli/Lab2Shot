+++
team = "清华大学 + 上海科技大学 + Deemos Tech"
people = "Yuxuan Han, Xin Ming, Tianxiao Li, Zhuofan Shen, Qixuan Zhang, Lan Xu, Feng Xu"
paper = "Learning a Delighting Prior for Facial Appearance Capture in the Wild（SIGGRAPH 2026）"
paper_url = "https://arxiv.org/abs/2605.05636"
website = "https://yxuhan.github.io/OpenDelight/"
repo = "https://github.com/yxuhan/OpenDelight"
year = 2026
+++

## 这是什么

OpenDelight 是一个完全开源、高性能的去光照先验，面向面部外观采集。作者把原来那套基于模型的逆向渲染换掉：先训练一个强的去光照网络当先验，再用它约束外观采集的优化，从随手拍的视频得到高质量的反射率估计；和已有的商业模型相比，烘进结果里的阴影明显更少。

在 Lab2Shot 里，它用来逐帧去掉脸上的光影（高光、阴影、环境光染的颜色），交出基础色（去掉光照的漫反射颜色，CG 流程里也叫 albedo）和一张脸部遮罩（不含头发），用来给角色做脸部贴图或者处理面部扫描贴图。基础色是网络给的显示用 sRGB 编码，这个节点原样交出；接「序列图输出设置」写 EXR 时由输出节点转成场景线性 ACEScg，可以进贴图流程。

**非商用**：只能用于研究和评估。

## 输入输出

**官方要什么、给什么**

- 吃：一个人脸画面文件夹。官方 `test.py` 收 `--data_root`、`--save_root`、`--img_size`（默认 512）、
  `--border_scale`、`--skin_mask`、`--use_enhancer`（细节增强）等
  （`third_party/opendelight/repo/test.py`）。
  它**自己**跑抠像和人脸对齐：逐张 `align_image(img_path, mask_path, predictor)`，
  **不吃外来的遮罩**。
- 给：去掉光照之后的脸 `output_face_torch` 和一张 `final_mask`；
  官方把两样存成**同一张 RGBA**（`save_image(torch.cat([res_torch, final_mask[:, :1]], dim=1))`）。

**我们怎么接的**

- 「图像」口就是 `--data_root` 那一段画面，「处理分辨率」= `--img_size`、「细节增强」= `--use_enhancer`、
  「关键点平滑」是 Lab2Shot 这一侧按序列做的（上游逐张独立算，脸的对齐会逐帧抖）。
- 「基础色」口 = `output_face_torch`，「遮罩」口 = `final_mask`：官方存成一张 RGBA 的两样，
  节点拆成两个口，数值一个字节不改。**上游那张图在论文和代码里叫 albedo / diffuse，这里的口叫「基础色」**
  ——CG 流程里 basecolor 和 albedo 指的是同一张图，所以整个项目只用「基础色」这一个说法；
  它和「DiffusionRenderer PBR 通道」的「基础色」口是同一种数据，接线时不会当成两回事。
  数值本身没有任何换算，这个节点也不做色彩转换：网络给的是显示用的 sRGB 编码，按工作空间原样交出去；
  写 EXR 时由输出节点用 OCIO 转成场景线性 ACEScg，合成和灯光要的就是这个。

出处：简介来自 `third_party/opendelight/repo/README.md`（OpenDelight is a fully open-source… 那一句）；
输入输出依据 `third_party/opendelight/repo/test.py`。

## 在 Lab2Shot 里怎么用

- **典型接法**：读取序列 → OpenDelight → 序列图输出设置（基础色和遮罩各接一个）。内置模板「面部去光照 · OpenDelight」就是这样接的。
- **什么素材效果好**：正面或接近正面、脸够大、对焦清楚的照片或特写。
- **一帧只处理一张脸**：画面里有好几张脸时，只处理最大的（视频里是和上一帧重叠的那张），节点会提示。
- **视频会闪**：它是一帧一帧单独算的，前后帧之间没有约束，所以更适合挑几张好的帧或者扫描照片来做贴图，不适合当视频结果交付。
- **关键参数**：
  - **细节增强**：多跑一个增强网络，毛孔、皱纹等细节更好，每帧慢约 20%。只要大致颜色时可以关。
  - **处理分辨率**：512（上游默认，目前唯一一档）。脸会先裁出来缩放到这个尺寸再算，最后贴回原图位置。
  - **关键点平滑**：视频输入时平滑每帧的脸部定位，减少裁切框抖动。遇到镜头切换或者几张不相关的照片会自动断开，不会把它们混在一起。

## 效果和局限

RTX 4090 上（默认参数：细节增强开、512）：

- 面部特写（772×855、113 帧）：一共 29 秒，其中加载模型约 6 秒，约 0.15 秒/帧；显存约 6.4 GB。
- 1920×1080 的素材 20 帧：约 0.34 秒/帧，显存约 6.6 GB。

已知问题：

- 视频逐帧会闪（见上）。
- 没检测到脸的帧输出全黑，节点会列出是哪几帧。脸部裁切框以外的地方也是黑的。
- 网络输出的颜色是显示用的 sRGB 编码（写 EXR 时输出节点才转成 ACEScg 线性），并不是物理测量出来的反照率，要当贴图起点来用，不能当测量值。
- 遮罩按面部分割去掉了头发，头发区域的基础色不要用，只用遮罩里面的部分。

## 团队

第一作者 Yuxuan Han 是清华大学 Feng Xu 教授组的博士生，一直在做「在家、在户外用普通设备采集面部材质」。他之前的工作有 High-Quality Facial Geometry and Appearance Capture at Home（CVPR 2024）、Facial Appearance Capture at Home with Patch-Level Reflectance Prior（SIGGRAPH 2025）、WildCap（CVPR 2026）。合作方有上海科技大学的 Lan Xu，以及 Deemos Tech。Deemos 是上海科技大学孵化、由该校学生创办的公司，主打产品 Hyper3D 做 AI 生成三维模型。

## 模型下载和安装

- **自动安装**：运行 `uv run lab2shot ext install opendelight`，会下载：
  - OpenDelight网络 + 细节增强网络（作者的 Google Drive 压缩包，约 460 MB，下载后自动解包）；
  - DAViD 前景抠像模型（ViT-L，约 1.3 GB）；
  - FaRL 面部分割模型（LaPa，约 620 MB）；
  - ibug 面部检测（RetinaFace）和 68 点关键点（FAN）的代码和权重（约 380 MB）；
  - 一个独立的 Python 环境（含 PyTorch，约 7.4 GB），不用编译。
  - 权重在硬盘上一共约 3.4 GB（压缩包本身也留着）。
- Google Drive 偶尔会限流导致下载失败，过一会儿重新运行安装就行，下好的文件不会重下。
- 原版训练时用的 MAE 初始化权重推理时用不到，所以不下载。

## 许可证说明

- **非商用**。代码是 GPL-3.0；**权重没有声明任何许可证**，训练数据是 FaceOLAT（马普所的灯光球数据集，只能用于学术研究）和作者购买的 120 套 Light Stage 扫描。所以按非商用处理：只能用于研究和评估，作者给出商用许可之前，不能用在商业制作里。商用请联系作者 Yuxuan Han（hanyx22@mails.tsinghua.edu.cn）。
- GPL-3.0 的代码在扩展包自己的独立进程里运行，不会让 Lab2Shot 本身也变成 GPL。
- 辅助模型：DAViD 抠像（MIT）、ibug 面部检测和关键点（MIT）、FaRL 面部分割（代码 MIT，但训练数据 LaPa 只能非商用）。

## 参考

- 论文：https://arxiv.org/abs/2605.05636
- 项目主页：https://yxuhan.github.io/OpenDelight/
- 代码仓库：https://github.com/yxuhan/OpenDelight
- 许可证原文（代码）：https://github.com/yxuhan/OpenDelight/blob/main/LICENSE
- 作者发布的 Ava256-Scan 面部扫描数据集：https://huggingface.co/datasets/yxuhan76/OpenDelight-Dataset
- 第一作者主页：https://yxuhan.github.io/
- FaceOLAT 数据集：https://github.com/prraoo/FaceOLAT
- DAViD（前景抠像）：https://github.com/microsoft/DAViD
- facer / FaRL（面部分割）：https://github.com/FacePerceiver/facer
- ibug 面部检测：https://github.com/hhj1897/face_detection
- ibug 面部关键点：https://github.com/hhj1897/face_alignment
