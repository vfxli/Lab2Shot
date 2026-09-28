+++
team = "苏黎世联邦理工学院（ETH Zürich）计算机视觉实验室 + INSAIT + 丰田欧洲（Toyota Motor Europe）"
people = "Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos, Luc Van Gool"
paper = "UniK3D: Universal Camera Monocular 3D Estimation（CVPR 2025）"
paper_url = "https://arxiv.org/abs/2503.16591"
website = "https://lpiccinelli-eth.github.io/pub/unik3d/"
repo = "https://github.com/lpiccinelli-eth/UniK3D"
year = 2025
+++

## 这是什么

UniK3D 是第一个能为任意相机建模的通用单目三维估计方法。它引入一种球面三维表示，把相机与场景几何解得更开，对不受约束的相机模型也能算出准确的真实尺度三维结果。相机那一部分用学到的球谐叠加，给出一种与相机模型无关的光线束表示；另有一项角度损失，避免广角相机的三维输出被压缩。在大视场和全景这类难处，提升尤其明显。

在 Lab2Shot 里，这个节点叫「UniK3D 深度图」：逐帧各算各的，交出深度图、沿光线的距离图、射线场、点云和置信度，鱼眼和 360° 全景素材同样能接。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`model.infer(rgb=…, camera=…, normalize=True, rays=…)`——一张 RGB 画面，
  相机和光线场都是**可选**的（`scripts/demo.py`；等距圆柱全景走 `infer_equirectangular`）。
  同样**只收内参 / 光线，不收外参**。
- **给**：`points`（光线 × 距离，对任何镜头都成立，见 `unik3d/models/unik3d.py`）、`depth`（沿相机 Z 轴的距离）、
  `distance`（沿每条光线的距离）、`rays`（它自己那份单位光线场）和置信度。
  **上游不出 intrinsics**：它给的是那套光线，不是一个针孔内参矩阵。

**我们怎么接的**

- 「RGB」= 上游那张画面。「已知 Focal Length」「Filmback」= 上游那个可选的相机条件（按去过畸变的针孔造出光线喂进去）；同样是两个参数，不是一台相机。
- 「深度图」= `depth`，「距离图」= `distance`，「射线场」= `rays`，「点云」= `points`，「置信度」= 它的置信度。
  上游算出的这几样都原样交出，一样不丢。
- 节点上**没有「相机」输出口**：上游不出内参。worker 把那套光线拟合成的**最小二乘针孔近似**（给了 Focal Length 时是精确的）
  只在家族内部用来反投影和摆位；上游自己那套球谐相机模型另外存在 `raw/camera.json` 里——Lab2Shot 的相机数据类型是针孔加畸变，
  装不下球谐表示（`adapters/unik3d/worker.py`，`adapters/unik3d/nodes.py` 的 `solves_camera = False`）。
- 节点上没有镜头模型、畸变系数、主点这类口：把 `rays` 拟合成鱼眼参数不是上游的结果。

**出处**：简介来自论文摘要（arXiv 2503.16591：「Monocular 3D estimation is crucial for visual perception. However, current methods fall short by relying on oversimplified assumptions, such as pinhole camera models or rectified images.」
以及它介绍方法的那两句：球面三维表示、与相机模型无关的光线表示由学出来的球谐叠加给出、角度损失）；
输入输出依据 `third_party/unik3d/repo/scripts/demo.py`、`repo/unik3d/models/unik3d.py`
和 `adapters/unik3d/nodes.py` 的 `official`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → UniK3D 深度图 → 深度图 / 距离图 / 射线场 / 点云；鱼眼、运动相机、GoPro 这类没去畸变的素材也能拿来就用（点云是按真实视线方向算的，不受镜头模型限制）。
- 「深度图」「距离图」「射线场」是同一次推理的三种说法：深度图是相机坐标系里到成像面的距离，距离图是沿这条视线离镜头多远（越靠画面边缘两者差得越多，鱼眼和广角上差别很大），射线场是这条视线本身（单位向量，相机空间）。距离乘射线场就是「点云」那片三维点。
- 节点不输出镜头模型和畸变系数：把射线场拟合成 OpenCV 鱼眼（Kannala-Brandt）参数不是 **上游 UniK3D 的输出**。要镜头畸变参数请用显式的标定节点（「AnyCalib 镜头标定」）。
- 镜头参数：raw/camera.json 里有 UniK3D 自己的镜头模型（球谐函数表示的视线场）和它的针孔最小二乘拟合（fx、fy、cx、cy）及拟合误差（像素）。针孔误差小于约 2 像素说明是普通镜头；家族内部反投影用的内参就是这个针孔近似。
- 知道镜头就填「已知 Focal Length」「Filmback」：只适合已经去畸变的普通镜头素材，会把这个针孔镜头作为条件输入网络，尺度更可信。
- "精度等级" 0–9（默认 9）：网络内部计算的分辨率，降低更快、细节更少；输出始终是原图大小。模型：ViT-L（默认）、ViT-B、ViT-S。
- 每帧单独计算，没有时序平滑；没有天空检测，也不输出法线。

## 效果和局限

RTX 4090 上（ViT-L，精度等级 9，fp16）：

- 速度和显存：864×480 约 0.055 秒/帧，1080×1920 约 0.08 秒/帧；显存峰值约 3.2–3.4 GB，内存约 3.6 GB。
- 864×480 固定机位镜头（参考 Focal Length 约 560–590px）：针孔近似 Focal Length 717px（704–730），偏长约 25%；针孔拟合误差约 1 像素（判断为普通镜头，正确）；舞者 3.05 米，后墙 6.5 米（预期约 5 米）。静止后墙逐帧闪动：标准差约 9 厘米（1.4%），逐帧跳动 0.45%。
- 1080×1920 iPhone 长焦镜头（参考 Focal Length 约 5484px）：针孔近似 Focal Length 4531px（逐帧 3973–5060，波动大），偏短 17%；它认为镜头有一点畸变（针孔拟合误差约 9 像素，实际素材是无畸变的），人物约 8.9 米。填入真实视场角后人物约 9.6 米（针孔几何核算约 10–11 米）。
- camera.json 里的镜头公式可靠：按公式还原的视线和输出点云方向相差不到 0.03°。
- 鱼眼效果没有在实拍鱼眼素材上验证过。
- 已知问题：普通焦段 Focal Length 偏长、长焦偏短且逐帧波动；远处尺度偏大。

## 团队

苏黎世联邦理工学院（ETH Zürich）Luc Van Gool 教授的计算机视觉实验室，一作 Luigi Piccinelli（也是 UniDepth 的作者），与保加利亚 INSAIT 研究所和丰田欧洲合作，论文发表于 CVPR 2025。UniK3D 是 UniDepth 的"任意镜头"版本，是第一个能在鱼眼和全景图上直接估计米制三维的单图模型。

## 模型下载和安装

- 运行 `lab2shot ext install unik3d`：下载 UniK3D 代码（锁定版本）、建独立 Python 环境（PyTorch 2.9，依赖清单和 UniDepth 相同，和 UniDepth、Depth Anything 3 共用下载缓存，基本不额外占空间），再下载三个模型：ViT-L 1.4 GB、ViT-B 0.47 GB、ViT-S 0.14 GB（Hugging Face 固定版本，安装程序校验 sha256）。网速正常时十分钟左右。
- 不需要申请权限，不需要自行下载任何文件。

## 许可证说明

- **非商用**。代码仓库的 LICENSE 文件和源码文件头写的是 CC BY-NC-SA 4.0（README 里写的是 CC BY-NC 4.0，按更严格的 LICENSE 文件对待）：可以研究、修改、分享，要署名，不能用于商业目的；改编后再发布必须用同样的许可证。
- 权重（Hugging Face 上的 unik3d-vitl / vitb / vits）的模型卡没有单独写许可证，按仓库许可证对待，结果不能用于商业项目。
- 主干网络 DINOv2 的结构来自 Meta（Apache-2.0），权重已经包含在 UniK3D 的检查点里。

## 参考

- 论文：https://arxiv.org/abs/2503.16591
- 项目主页：https://lpiccinelli-eth.github.io/pub/unik3d/
- 代码：https://github.com/lpiccinelli-eth/UniK3D
- 许可证原文：https://github.com/lpiccinelli-eth/UniK3D/blob/main/LICENSE
- 模型卡（ViT-L）：https://huggingface.co/lpiccinelli/unik3d-vitl
- 在线演示：https://huggingface.co/spaces/lpiccinelli/UniK3D-demo
