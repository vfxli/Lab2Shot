+++
team = "丰田研究院（TRI）、斯坦福大学、卡内基梅隆大学等"
people = "Adam W. Harley, Yang You, Xinglong Sun, Yang Zheng, Nikhil Raghuraman, Yunqi Gu, Sheldon Liang, Wen-Hsuan Chu, Achal Dave, Pavel Tokmakov, Suya You, Rares Ambrus, Katerina Fragkiadaki, Leonidas J. Guibas"
paper = "AllTracker: Efficient Dense Point Tracking at High Resolution（ICCV 2025）"
paper_url = "https://arxiv.org/abs/2506.07310"
website = "https://alltracker.github.io/"
repo = "https://github.com/aharley/alltracker"
year = 2025
+++

## 这是什么

上游 README 这样介绍它：AllTracker 是一个点跟踪模型，比其他同类模型更快、更准，并且在
高分辨率上输出稠密（全像素）的结果。它估计长程点轨迹的做法是：算出查询帧与视频里其他每一帧
之间的流场。和已有的点跟踪方法不同，它交出的是高分辨率、稠密（全像素）的对应场，可以按流场图
来看；和已有的光流方法不同，它把一帧对应到之后的几百帧，而不只是下一帧。

在 Lab2Shot 里，它是「AllTracker 跟踪贴片」：在参考帧上画一次，贴着画面跟到整段镜头——
纹身、logo、假体接缝、衣服上的威亚、脸上的修补，对应 Nuke 里 SmartVector + VectorDistort 或
Mocha Remove 那条流程。代码和权重都是 MIT，可以商用。

## 输入输出

**官方要什么、给什么**

- 吃：一段画面。官方 `demo.py` 收的是一个 mp4（`--mp4_path`）加一个参考帧号（`--query_frame`）、
  最大边长（`--image_size`，默认 1024）、最多帧数（`--max_frames`，默认 400）。模型这一层是
  `model.forward_sliding(rgbs, iters, sw, is_training)`：**只吃画面**，没有查询点、没有遮罩。
- 给：`traj_maps_e`（`B,T,2,H,W`，每帧每个像素回到参考帧的位置）和 `visconf_maps_e`
  （`B,T,2,H,W`，可见性和置信度）。参考帧之前的那一段是把画面倒过来再算一遍拼上的。
  demo 里的 2D 点是把稠密结果按 `--rate` 隔点抽样出来的。

**我们怎么接的**

- 「图像」口就是上游的 `rgbs`，「参考帧」参数就是 `--query_frame`，「处理分辨率」就是 `--image_size`。
- 「ST-map」口（2 通道）就是 `traj_maps_e`，「置信度」口就是 `visconf_maps_e`，
  「2D 跟踪点」口是按「网格点数」在参考帧上撒点、从稠密结果上取值得到的（对应 demo 的 `--rate` 抽样）。
- **上游没有、节点上也没有**：遮罩口。要只跟一块区域，在节点图上接「人物框转遮罩 → 图像相乘」把画面挡住再送进来。

出处：简介来自 `third_party/alltracker/repo/README.md`；输入输出依据 `third_party/alltracker/repo/demo.py`
和 `nets/alltracker.py`，以及 `adapters/alltracker/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 输出「ST-map」：每一帧一张，指回参考帧——这一帧的每个像素，去参考帧的哪里取色（Nuke STMap 的约定，R = x/W，G = 1 − y/H）。
  在参考帧上画好修补（Nuke 里画，写成一张图，带 alpha 的话 alpha 另存成遮罩），用「读取序列」读进来（一张图），
  接「STMap」的「源」，ST-map 接这个输出：修补就贴到了每一帧上。Nuke 里同样：STMap 节点，src 接画好的那一帧。
- 输出「置信度」：这一帧的像素能不能追回参考帧（AllTracker 的可见度 × 置信度，0–1；被挡住、出画、模型没把握的地方接近 0）。贴修补时先接「置信度转遮罩」变成遮罩，0 的地方要补画。2D 跟踪点也带每个点每帧的置信度。
- 输出「2D 跟踪点」：从同一份结果里取样出网格点（`网格点数`，撒在参考帧上）和在 2D 视图里点的点
  （点在任何一帧都行），接「2D 跟踪点输出设置」写 3DE / CSV。不要点也不要网格就填 0。
- `参考帧`：在哪一帧上画。每一帧都指回这一帧；前后都会跟过去（前面的帧倒着跟）。
- `处理分辨率`：长边缩到这个像素再跟点，结果放大回原尺寸。论文里 768×1024 要 40 GB 显存；Lab2Shot 分块取数据、
  帧留在内存里，1024 在 RTX 4090 上放得下（见下）。放大是插值：比处理分辨率一个像素更细的细节（发丝、细纹理）不会跟得更准。

## 效果和局限

RTX 4090 上（把参考帧用 ST-map 贴到每一帧，在置信度 > 0.5 的像素上和那一帧比，误差是 0–255 的平均差）：

- 手持长焦 1080×1920，120 帧，参考帧 1001，处理分辨率 576×1024：每帧 0.095 s，显存峰值 11.1 GB（论文说 768×1024 要
  40 GB，这里帧留在内存里、按窗口送上显卡）。镜头一直在摇，到第 1060 帧画面已经移了约 470 px：
  第 1011 / 1061 / 1120 帧贴过去的误差 5.4 / 12.8 / 17.1（不对齐是 46.6 / 60.7 / 79.3），可信的像素 89% / 71% / 35%
  （后面大部分内容已经出画）。
- 同一段镜头和 MEMFOF 运动矢量一帧帧串起来（「运动矢量转 ST-map」）比，在两边都可信的像素上：AllTracker 的误差每一帧都更小
  （第 1061 帧 11.3 对 14.5，第 1101 帧 13.3 对 17.8），两者的 ST-map 在 60 帧时相差中位数约 5 px、100 帧以上约 10 px，
  可信的像素也多（120 帧时 35% 对 20%）：串光流每帧会漂零点几个像素，AllTracker 对到参考帧，不累积。
- 固定机位跳舞 864×480，124 帧，参考帧在中间（1062，前后各跟 60 帧）：贴过去的误差 1.1–2.4（不对齐 5.2–9.1），
  约 90% 的像素可信（串光流 74–88%）。
- 误差里有亮度变化、运动模糊和噪点，不全是位置不准；两帧之间拿来就用光流拉回的误差大约是 1–3，可以当作底线。

已知局限：
- 处理分辨率决定精度上限：1080p 素材用 1024 长边处理时，一个处理像素约等于原图 1.9 个像素。
- ST-map 是把"参考帧 → 每一帧"的对应反过来求的（每个像素找它在参考帧的位置）；被挡住、新露出来的地方没有答案，置信度为 0。
- 很长的镜头（几百帧）内存里要放下全部帧和结果：1024 长边时约每 100 帧 1.5 GB。

## 团队

Adam W. Harley（斯坦福、TRI）主导，合作者来自丰田研究院、卡内基梅隆大学、斯坦福 Leonidas Guibas 组等。
Harley 之前做了 PIPs、PIPs++ 等点跟踪方法；AllTracker 把点跟踪和光流合在一起，输出高分辨率的稠密结果。

## 模型下载和安装

- 自动安装：`lab2shot ext install alltracker`。锁定 GitHub 仓库的一个版本，独立的 PyTorch 环境（和其他扩展共用下载缓存），
  下载作者在 Hugging Face 上的 alltracker.pth（66 MB，完整训练的模型）。不需要申请权限；不下载 ImageNet 预训练的 ConvNeXt（权重文件里已经有）。

## 许可证说明

代码和权重都是 MIT：可以商用、修改和再分发，保留版权声明即可。

## 参考

- 论文：https://arxiv.org/abs/2506.07310
- 项目主页：https://alltracker.github.io/
- 代码：https://github.com/aharley/alltracker
- 权重：https://huggingface.co/aharley/alltracker

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
