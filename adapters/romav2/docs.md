+++
team = "瑞典林雪平大学（Linköping University）、查尔姆斯理工大学、隆德大学"
people = "Johan Edstedt, David Nordström, Yushan Zhang, Georg Bökman, Jonathan Astermark, Viktor Larsson, Anders Heyden, Fredrik Kahl, Mårten Wadenbäck, Michael Felsberg"
paper = "RoMa v2: Harder Better Faster Denser Feature Matching（2025）"
paper_url = "https://arxiv.org/abs/2511.15706"
website = "https://github.com/Parskatt/romav2"
repo = "https://github.com/Parskatt/romav2"
year = 2025
+++

## 这是什么

RoMa v2 做的是稠密特征匹配：估计同一个三维场景的两张画面之间的全部对应关系。稠密匹配精度高、稳，近年被当成这一类任务的标准做法，但已有的匹配器在很多困难的真实场景里仍然会失败或者表现不好，高精度的模型又往往很慢。作者从匹配架构和损失、更多样的训练分布、先匹配后细化的两阶段流程、自写的 CUDA kernel 和 DINOv3 基础模型几方面一起改，报告的结果比前代明显更准，刷新了当时的最好成绩。

在 Lab2Shot 里，我们拿它做图像对位：「图像」和「参考图」两张画面可以差很多（不同机位、不同时间、参考照片和画面），节点交出一张 ST-map、一张置信度，和两组同名成对的匹配点。可以**商用**（MIT；骨干 DINOv3 的许可证允许商用，有附加条件，见许可证说明）。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`RoMaV2().match(img_A_path, img_B_path)`，两张画面（README:36-45；也可以调 `model(img_A, img_B)` 走 forward）。
- **给**：`preds`，里面有 `warp_AB`（A 的每个像素在 B 里的位置，归一化到 `[-1,1] × [-1,1]`）和 `overlap_AB`（这个像素在 B 里看不看得见）；
  `model.sample(preds, 5000)` 取样出 `matches, overlaps, precision_AB, precision_BA`，`model.to_pixel_coordinates` 把匹配换成像素坐标（README:47-51）。
  上游还出反向的 `warp_BA` / `overlap_BA` / precision（`src/romav2/romav2.py:365-368`）。

**我们怎么接的**

- 「图像」= A（结果都在它的像素上），「参考图」= B。画面是序列时按帧号配对；参考是一张照片时和每一帧配对（`adapters/romav2/worker.py:7-10`）。
- 「ST-map」= 官方的 `warp_AB`（`romav2.py:344`）换算到画面本来的尺寸；「置信度」= `overlap_AB`；
  「画面上的匹配点」「参考图上的匹配点」= 同一份 `matches_AB`（`romav2.py:384`）的两半，同名的是一对。
- 「精度」参数 = 上游那几档方形输入（快 512 × 512、标准 640 × 640、精细 800 × 800 匹配加 1280 × 1280 细化、双向，官方默认）。
- **不一样的两点**：① 上游在自己的方形网格上给结果，我们放大回画面的尺寸再输出；取样固定随机种子，同一次计算结果可复现；
  ② 上游反向的 `warp_BA` / `overlap_BA` / precision **我们没有对应的输出口**（`adapters/romav2/nodes.py:14-21`）。

**出处**：简介抽自论文摘要第 1、2 句（arXiv 2511.15706：「Dense feature matching aims to estimate all correspondences between two images of a 3D scene…」）
和标题；输入输出依据 `third_party/romav2/repo/README.md:36-51`、`repo/src/romav2/romav2.py:301-395` 和 `adapters/romav2/nodes.py:14-21`。

## 在 Lab2Shot 里怎么用

- 两个输入：「图像」（结果都在它的像素上）和「参考图」（要对齐过来的那张：见证机、照片、补拍）。
  画面是序列时按帧号配对；参考图是一张照片时和每一帧配对。
- 输出「ST-map」：画面的每个像素去参考图的哪里取色。接「STMap」，「源」接参考图，就得到对齐到画面上的参考图
  （Nuke 里 STMap 节点，src 接参考图）。「置信度」：这个像素在参考图里有没有、准不准（0–1，被挡住、出了参考图的画面接近 0；当遮罩用先接「置信度转遮罩」）。匹配点也带每个匹配的置信度。
- 输出「画面上的匹配点」和「参考图上的匹配点」：两组点，同名的是同一对（`match_<帧号>_<序号>`），每个点只在它那一对的帧上可见。
  接「2D 跟踪点输出设置」写 3DE（两台相机共用点名）或 CSV，给 COLMAP、PnP、3DE 的见证机解算用。
  逐帧匹配时每个点只有一帧，写出时把「最少可见帧数」设成 1。
- 用法举例：见证机、参考照片对齐到画面；补拍对齐英雄素材；在一个 take 上画的 roto 传到另一个 take（roto 接「源」）；
  摆 LiDAR 时给 2D ↔ 3D 对应。
- `精度`：快（512×512）、标准（640×640）、精细（800×800 匹配、1280×1280 细化、双向，官方默认）。两张图都先压成正方形给模型，
  结果再放大回画面的尺寸。

## 效果和局限

RTX 4090 实测（参考图用 ST-map 贴到画面上，在置信度 > 0.5 的像素上和画面比，误差是 0–255 的平均差）：

- sh020 手持长焦 1080×1920，第 1060 帧对齐到第 1001 帧（中间镜头摇了约 470 px）：精细 2.3 s 一对、显存峰值 9.7 GB；
  两帧重叠的部分约 35% 被判为可信，在这些像素上误差 11.0（不对齐 61.1）；快（512×512）1.0 s、2.5 GB，误差 11.5，差不多。
- sh010 固定机位跳舞 864×480，第 1100 帧对齐到第 1001 帧（人动了）：65% 可信，误差 2.3（不对齐 7.7）。
- 每对取样 2000 个匹配点，两边同名。

已知局限：
- 匹配是逐对算的，没有时间上的连贯：逐帧对齐一段序列时，ST-map 可能有轻微的逐帧抖动。
- 模型的网格最细 1280×1280，放大到 1080p 以上时细节是插值的。
- 没有装作者可选的融合相关性加速核（fused-local-corr 只有 torch 2.11 的版本），用的是等价的纯 PyTorch 实现，结果一样，稍慢。

## 团队

Johan Edstedt（林雪平大学计算机视觉实验室，Michael Felsberg 组）主导，合作者来自查尔姆斯理工大学（Fredrik Kahl）
和隆德大学（Viktor Larsson）。之前的 DKM、RoMa 都出自这个团队，是稠密匹配里最常用的方法之一。

## 模型下载和安装

- 自动安装：`lab2shot ext install romav2`。锁定 GitHub 仓库 v2.0.1，另外锁定 RoMa v2 用到的 DINOv3 网络代码
  （facebookresearch/dinov3 的一个版本，原来是运行时联网拉取，现在安装时下载好），独立的 PyTorch 环境，
  下载作者 GitHub 发布页的 romav2.0.1.pt（1.1 GB，里面包含 DINOv3 ViT-L 骨干的权重），下完按 sha256 校验。不需要申请权限。

## 许可证说明

RoMa v2 的代码和权重是 MIT。权重里的 DINOv3 ViT-L 骨干受 Meta 的 **DINOv3 License** 约束：允许商用、修改和再分发；
再分发时要附带许可证原文，发表论文要注明用了 DINOv3；禁止用于军事、武器、核工业、间谍等用途，受制裁方不能使用。
Lab2Shot 按可商用标注，附条件写在节点的许可说明里。

## 参考

- 论文：https://arxiv.org/abs/2511.15706
- 代码：https://github.com/Parskatt/romav2
- DINOv3 License：https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
