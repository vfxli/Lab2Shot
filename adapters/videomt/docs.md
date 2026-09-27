+++
team = "埃因霍温理工大学移动感知系统实验室、亚琛工业大学（TU Eindhoven Mobile Perception Systems Lab, RWTH Aachen）"
people = "Narges Norouzi, Idil Esen Zulfikar, Niccolò Cavagnero, Tommie Kerssies, Bastian Leibe, Gijs Dubbelman, Daan de Geus"
paper = "VidEoMT: Your ViT is Secretly Also a Video Segmentation Model（CVPR 2026）"
paper_url = "https://arxiv.org/abs/2602.17807"
website = "https://github.com/tue-mps/videomt"
repo = "https://github.com/tue-mps/videomt"
year = 2026
+++

## 这是什么

VidEoMT 是一个只有编码器的轻量模型，用于在线视频分割，建在普通的 Vision Transformer（ViT）之上。空间和时间两方面的推理都在 ViT 编码器内部完成，不依赖专门跟住物体的模块，也不用沉重的任务专用头。它沿时间传递的办法是复用上一帧的 query，再和一小组学到的、与帧无关的 query 融合。这样的设计在精度有竞争力的同时，比已有做法快 5–10 倍，ViT-L 骨干最高可到 160 FPS。

在 Lab2Shot 里，这个节点叫「VidEoMT 整画面分割」：不用提示词、不用点选，整段素材的每个像素都分到一类，交出全景分割和类别分割两张分割图。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：视频的每一帧（`videomt/videomt.py` 里的 `images`）。按它的测试设置缩放（短边一个定值，长边不超过它的 1333/720 倍）、
  归一化、补到 32 的倍数；评估时窗口大小为 1，一帧一帧走，query 从上一帧接着用。
- **给**：视频全景分割——`pred_masks` 加 `pred_logits`（同一段）。每个 query 就是整段里的一个片段，这是编号稳定的来源；
  每个 query 的类别，上游取它在 clip 每一帧上的类别 logits 的平均再 softmax。类别表是 VIPSeg 的 124 类，写死在权重里。
  **上游只吃画面，没有别的输入。**

**我们怎么接的**

- 「图像」= 上游那段素材；「处理分辨率」= 上游测试设置的短边；「检测阈值」= 留下一个 query 需要的类别分数；
  「完整度门槛」= 一个 query 最终保住自己遮罩面积的比例。
- 「全景分割」= 每个片段一个编号（`pred_masks`），「类别分割」= 每个像素的 VIPSeg 类别（`pred_logits`）。
- **类别分数照官方算**（`adapters/videomt/worker.py`）：每个 query 的类别是它在**所有帧**上的
  类别 logits 简单平均再 softmax，分母就是帧数。
- **不一样的两点**：① 多一条官方没有的过滤：只在少于 3 帧（`MIN_FRAMES`）里是一个物体的 query 不留，挡掉一闪而过的误检，
  它不碰分数、只决定留不留；
  ② 上游一次把整段拿进来，这里分几遍算（网络是确定性的，再算一遍结果一样），让整段镜头按画面本来的尺寸出结果。

**出处**：简介来自 `third_party/videomt/repo/README.md`（`We introduce Video Encoder-only Mask Transformer (VidEoMT), a lightweight encoder-only model for online video segmentation built on a plain Vision Transformer (ViT)`）
和同一段的「VidEoMT propagates information over time by reusing queries from the previous frame and fusing them with a compact set of learned, frame-agnostic queries.」）；
输入输出依据 `repo/videomt/videomt.py` 和 `adapters/videomt/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- **挑出要的物体**：接「分割转遮罩」。上游算过以后参数里列出这段镜头里有的类别和物体，点，或者在 2D 视图里点画面：
  - 换天空：「类别分割」→ 分割转遮罩 选 sky（天空）；
  - 植被、道路、建筑的遮挡（holdout）：选 tree、grass、other_plant、road……可以多选；
  - 某一个人或某一辆车：「全景分割」→ 分割转遮罩 选 person 2。
- **写成 Cryptomatte**：两张分割图都接「多层 EXR 输出设置」（再接「输出」），按 Cryptomatte 1.2 写（MurmurHash3 编号 + 文件头清单），Nuke 的 Cryptomatte 节点认，点物体就能取遮罩。「图层名」按接线顺序填，比如先接画面再接两张分割图时填 `, CryptoObject, CryptoMaterial`（画面留空保持 rgba，全景一层、类别一层）。「读取序列」读回来还是同样的分割图和类别表。
- **边缘要精修**：模型按短边 720 左右的分辨率算，遮罩边是它的精度（放大到原画面时按概率平滑过渡，不是锯齿，但头发、树叶、运动模糊的软边它给不了）。要软边就接抠像：
  - 人：分割转遮罩 → **MatAnyone 2 精细抠像** 的「粗遮罩」，得到带发丝的 alpha；
  - 任何主体：**BiRefNet 抠像** 的 alpha 和分割转遮罩的遮罩（先「遮罩调整」扩边 10–20 px）做「遮罩合并 · 相交」，BiRefNet 的细边只留在这个物体上。
- **和 SAM 3 的分工**：SAM 3 按你给的词找物体（任何词，如 red umbrella），VidEoMT 不用词、整帧每个像素都有归属，还包括天空、道路这类“物体”（stuff）。要“把画面按类别拆开”用 VidEoMT，要“找某个特定物体”用 SAM 3。
- **和 SegAnyMo 的分工**：VidEoMT 按“是什么”分，停着的车也是 car；只要“正在动的”物体用 SegAnyMo。
- **参数**：
  - **处理分辨率**：短边缩到多少像素再算，默认 720（作者评测用的尺寸）。调大小物体和边缘好一点，但模型没在那么大的画面上训练过，而且更慢。
  - **检测阈值**：物体整段的类别把握要超过它才输出（默认 0.8）。有物体没分出来就调低；分出了不存在的物体就调高。
  - **完整度门槛**：一个物体自己的遮罩里至少要有这么大比例最后归它，否则整段去掉（默认 0.8）。

## 效果和局限

- 类别是 VIPSeg 的 124 类（日常街景、室内），天空、树、建筑、道路、人、车这些常见的很稳；专门的道具、特效条目、少见物体会被归到相近的类（other_construction、other_machine、other_plant）。
- 边缘精度是短边约 720 像素的网络精度；4K 素材上细节要靠抠像精修。
- 物体的类别按它在**所有帧**上的类别分数平均算（官方口径）：半路走进画面的人，前面「什么都不是」的帧也算进分母，
  整段只露面很短的物体类别分数会偏低、可能被检测阈值挡掉，这时调低阈值；只在少于 3 帧里出现的物体直接不留。
  同一个查询前后换成了另一个物体（很少见），类别按多数算。
- 离开画面再回来的物体会拿到新编号（它是逐帧往后看的模型，不会认回来）。

## 团队

埃因霍温理工大学（TU/e）移动感知系统实验室，和亚琛工业大学（RWTH Aachen）的 Bastian Leibe 组合作。同一个实验室还做了 EoMT（CVPR 2025，“你的 ViT 其实就是一个图像分割模型”）、后续的 PMT（冻结骨干网络）和 LVMT（长视频遮挡）。

## 模型下载和安装

- 运行 `uv run lab2shot ext install videomt`，会下载：
  - VidEoMT-L VIPSeg 全景分割权重 `vipseg_vit_large_55.2.pth`（1.3 GB，Hugging Face tue-mps/VidEoMT，不用申请权限）；
  - 一个独立的 Python 环境（PyTorch 2.8.0 + CUDA 12.8 + timm，约 5 GB），不用编译任何东西。
- 只用仓库里 MIT 许可的网络部分；作者的训练和评测框架（detectron2 + 来自 MinVIS 的代码）不装也不运行。

## 许可证说明

- **代码和权重都是 MIT**，DINOv2 骨干网络是 Apache-2.0。
- **但这份全景分割权重只在 VIPSeg 数据集上训练**，VIPSeg 的许可写明“只用于非商业研究”。用研究数据训练出的权重能不能商用，法律上没有定论，Lab2Shot 按**非商用**标注；商业项目使用前请法务确认。
- 仓库里有几个文件带 NVIDIA Source Code License-NC（来自 MinVIS 的训练和评测框架），Lab2Shot 不运行它们。

## 参考

- 论文：https://arxiv.org/abs/2602.17807
- 代码仓库：https://github.com/tue-mps/videomt
- 模型：https://huggingface.co/tue-mps/VidEoMT
- VIPSeg 数据集：https://github.com/VIPSeg-Dataset/VIPSeg-Dataset
- 前作 EoMT：https://github.com/tue-mps/eomt
