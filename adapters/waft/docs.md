+++
team = "普林斯顿大学视觉与学习实验室（Princeton Vision & Learning Lab）"
people = "Yihan Wang, Jia Deng"
paper = "WAFT: Warping-Alone Field Transforms for Optical Flow（ICLR 2026 Oral）"
paper_url = "https://arxiv.org/abs/2506.21526"
website = "https://github.com/princeton-vl/WAFT"
repo = "https://github.com/princeton-vl/WAFT"
year = 2026
+++

## 这是什么

WAFT（Warping-Alone Field Transforms）是一种简单有效的光流方法。它与 RAFT 相似，但把代价体换成高分辨率的扭曲，精度更好、显存占用更低；这个设计也质疑了「必须构造代价体才有好性能」这一惯常看法。它是一个简单灵活的元架构，归纳偏置和专门设计都很少。与已有方法相比，WAFT 在 Spring、Sintel、KITTI 三个基准上均排第一，在 KITTI 上零样本泛化最好，同时比性能相当的方法最多快 4.1 倍。

在 Lab2Shot 里，这个节点叫「WAFT 运动矢量」。上游是两帧方法，节点对每一帧各算一遍到下一帧、一遍到上一帧，拼成前后双向的运动矢量，把握程度写成置信度通道。非商用，见许可证说明。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：相邻两帧画面——`model.calc_flow(image1, image2)`（`demo.py`）；
  命令行 `demo.py --cfg --ckpt --dataset --scale` 是在标准基准集上按两帧一对评估用的。
- **给**：这一对之间的光流 `output['flow']`，外加 `output['info']`——上游自己的 `get_heatmap` 就是拿它算的，
  也就是每个像素的把握程度。

**我们怎么接的**

- 「RGB」= 整段镜头；节点按两帧一对送进去：第 i 帧的前向是 `flow(i, i+1)`、后向是 `flow(i, i-1)`，一批算完
  （`adapters/waft/worker.py`）。
- 「运动矢量」= 上游的 `flow`，装成 Nuke 能用的四个通道（前向两个 + 后向两个）；
  「置信度」= 上游的 `output['info']`——它在 Lab2Shot 的运动矢量数据里是一条置信度通道，不另立一个口。
- 「处理分辨率」= 送进去的长边（默认 960，WAFT 的训练尺寸）。
- **不一样的两点**：① 上游 `--dataset` 是在基准集上评估，这里喂的是节点图上的素材；
  ② 上游建网络时会去加载 Depth Anything V2 和 timm 的 ResNet-18 预训练部件，而它的权重文件里这些权重都有，
  所以这里不加载那些、严格读权重（数值一样，见 `adapters/waft/worker.py`）。

**出处**：简介来自 `third_party/waft/repo/README.md`（「We introduce Warping-Alone Field Transforms (WAFT), a simple and effective method for optical flow. WAFT is similar to RAFT but replaces cost volume with high-resolution warping, achieving better accuracy with lower memory cost… WAFT ranks 1st on Spring, Sintel, and KITTI benchmarks」）；
输入输出依据 `repo/demo.py` 和 `adapters/waft/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

和「MEMFOF 运动矢量」完全一样：同一个节点家族、同样的输出（Nuke 的 motion 通道：`forward.u/v`、`backward.u/v`，
像素 / 帧，u 向右、v 向上）和同样的下游通用节点（遮挡遮罩、运动矢量变形、运动矢量转 ST-map）。两者可以互换，
按镜头挑：MEMFOF 可以商用、显存小；WAFT 在细节和大运动上有时更准，但只能研究用。对比数字见下面。

- `处理分辨率`：默认 960。WAFT 在 432×960 的画面上训练，1080p 素材在 960 上算和原尺寸一样准，快得多、显存小得多（矢量放大回原尺寸）；留空 = 原尺寸。
  画面长边超过 1920 时，留空自动按 1920 封顶（更大的原生分辨率没有验证过）。

## 效果和局限

RTX 4090 上（「运动矢量变形」把下一帧拉回这一帧，在「遮挡遮罩」以外的像素上和这一帧比，误差是 0–255 的平均差；
不对齐时的差写在括号里）：

| 镜头 | 模型 | 处理分辨率 | 每帧 | 显存峰值 | 拉回误差 |
|---|---|---|---|---|---|
| 手持长焦 1080×1920，每帧约 15 px | MEMFOF | 原尺寸 | 0.48 s | 2.3 GB | 3.07（16.2） |
| | MEMFOF | 长边 960 | 0.15 s | 0.6 GB | 3.21 |
| | WAFT | 原尺寸 | 2.30 s | 14.3 GB（SDPA 后约 9.7） | 3.10 |
| | WAFT | 长边 960（默认） | 0.33 s | 2.2 GB | 3.05 |
| 固定机位跳舞 864×480 | MEMFOF | 原尺寸 | 0.09 s | 0.5 GB | 0.44（0.74） |
| | WAFT | 原尺寸 | 0.24 s | 2.0 GB | 0.49（0.78） |
| 人脸 772×855 | MEMFOF | 原尺寸 | — | — | 1.31（3.00） |
| | WAFT | 原尺寸 | 0.41 s | 2.7 GB | 1.42（3.01） |

- 两个模型拉回以后都只剩 0.4–3 左右的差（主要是噪点、运动模糊和亮度变化），比不对齐小 2–5 倍；MEMFOF 和 WAFT 准确度相当，
  MEMFOF 原尺寸就快、显存小，WAFT 在它训练的 960 上最划算。
- 置信度有用：置信度最低的 20% 像素，拉回误差是其余像素的 2–13 倍（手持长焦镜头上 MEMFOF 5.6 对 3.0，WAFT 7.0 对 2.6；
  固定机位镜头上 MEMFOF 3.8 对 0.29）。WAFT 的置信度区分得更开。
- 前后一致性检查标出的遮挡：手持长焦镜头约 2–4%，固定机位镜头约 1.5–2%（跳舞的手脚边缘、出画的边）。
- 串起来传播（「运动矢量转 ST-map」，手持长焦镜头从第一帧往后 120 帧）：和 AllTracker 在同样的像素上比，60 帧时两者的 ST-map
  相差中位数约 5 px、100 帧以上约 10 px，贴过去的误差也比 AllTracker 大（同一帧上 14.5 对 11.3）；追得回参考帧的像素
  到 120 帧只剩 20%（AllTracker 35%）。几十帧以内可以用，更长用 AllTracker。

已知局限：
- 两帧方法，前后两个方向各算一遍，比 MEMFOF 多一倍计算。
- 权重没有写明许可，而且用只许研究使用的数据集训练，按非商用处理：商业项目请用 MEMFOF。

## 团队

普林斯顿大学 Jia Deng 教授的视觉与学习实验室，光流领域最有影响的方法 RAFT（ECCV 2020 最佳论文）和 SEA-RAFT 都出自这里。

## 模型下载和安装

- 自动安装：`lab2shot ext install waft`。锁定 GitHub 仓库的一个版本，独立的 PyTorch 环境（和其他扩展共用下载缓存），
  从作者的 Google Drive 下载 README 推荐"用于实际应用"的 WAFT-a1 tar-c-t 权重（257 MB，Depth Anything V2 ViT-S 特征），
  下完按 sha256 校验。不装 xformers（官方建议装它只为加速注意力；少一个依赖）：Depth Anything V2 的注意力在 worker 里改走 torch 的 scaled_dot_product_attention（同一算式，省掉 tokens×tokens 的显式矩阵），1920 档显存 14.3 → 9.7 GB，不下载 Depth Anything V2 和 timm 的预训练权重（权重文件里已经有）。
- 不用 PTLFlow 里的 WAFT 权重（PTLFlow 自己训练的是 CC BY-NC-SA）。

## 许可证说明

代码 BSD-3-Clause，可以商用。权重放在作者的 Google Drive 上，没有单独写许可证，而且是在 Sintel、KITTI、Spring、
TartanAir 等只许研究使用的数据集上训练的，所以 Lab2Shot 按**非商用**标注。骨干 Depth Anything V2 ViT-S 本身是 Apache-2.0。

## 参考

- 论文：https://arxiv.org/abs/2506.21526
- 代码：https://github.com/princeton-vl/WAFT
- 权重（Google Drive，a1 文件夹）：https://drive.google.com/drive/folders/1joBWKGoH2RUdCgcge8Tz2osOHcQUX5m_

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
