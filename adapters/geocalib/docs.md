+++
team = "苏黎世联邦理工学院计算机视觉与几何组（ETH Zurich CVG）+ 微软（Microsoft）"
people = "Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys"
paper = "GeoCalib: Learning Single-image Calibration with Geometric Optimization（ECCV 2024）"
paper_url = "https://arxiv.org/abs/2409.06704"
website = "https://veichta-geocalib.hf.space"
repo = "https://github.com/cvg/GeoCalib"
year = 2024
+++

## 这是什么

上游自己的话：GeoCalib 是做单图标定的算法——仅凭一张画面就估计出相机内参和重力方向。把几何优化与深度学习结合之后，它的标定比以往的做法更灵活、也更准确。论文《GeoCalib: Learning Single-image Calibration with Geometric Optimization》（ECCV 2024）。

在 Lab2Shot 里，它是「GeoCalib 重力方向」，主要用它给出的重力方向（相机的倾斜 roll 和俯仰 pitch）：把相机解算出来的世界转正、落地——这是 AnyCalib 没有的一项，也是接它的原因。上游给的是 Focal Length（px），「Focal Length」口交出去之前按节点上的「Filmback」换算成毫米（毫米是 Lab2Shot 的内部标准单位）。

## 输入输出

**官方要什么、给什么**

- 吃：**一张**画面，`(C, H, W)` 或 `(B, C, H, W)`、RGB、值在 0–1（`geocalib/extractor.py`）。
  可选的先验 `priors`：已知 Focal Length `focal`、已知重力 `gravity`；
  还可以选镜头模型 `camera_model`（默认 `pinhole`）。
- 给：一个字典——`camera`（相机对象，里面有 Focal Length（px） `f` 和畸变 `dist`）、
  `gravity`（重力方向）、`covariance`，以及所有带 `field`、`confidence`、`uncertainty` 的项
  （逐像素的视场场、置信度，和重力、Focal Length 各自的不确定度）。

**我们怎么接的**

- 「图像」口就是 `img`，节点上的**「拟合模型」**（参数 `fit_model`）就是 `camera_model`，「已知 Focal Length」走的是官方的 `priors["focal"]`
  那一路（填了就固定它、只估重力方向）。
- 「重力方向」= `gravity`，「重力误差」= 上游的 `gravity_uncertainty`（单位度），
  「Focal Length」= `camera.f`（**像素**，按节点上的「Filmback」换成毫米再交），「畸变系数」= `camera.dist`，
  「镜头模型」口（`lens_model`）= 请求的那个模型名。
- **「拟合模型」是要求，「镜头模型」是结果**：前者是节点上的参数（送进上游当 `camera_model`），后者是输出口。
- **「镜头模型」「畸变系数」两个口只在带畸变的那几档下用得上**：「拟合模型」选到「无畸变」或「鱼眼」（鱼眼是除法模型，核心镜头表里没有对应公式，按无畸变交，估出来的那一份留在镜头的来源里）时，上游根本没解畸变，那两个口没有值——它们在节点上**变灰、鼠标停上去写清「「拟合模型」选…时才用」、线也接不出去**（口不消失，位置不跳）。GeoCalib 没有主点输出口。
- **不一样的三处**：①**单位换过**——上游的 `camera.f` 是**像素**，「Focal Length」口交的是**毫米**，
  按节点上的「Filmback」换算（毫米 = 像素 ÷ 画面宽度 × Filmback）。毫米是 Lab2Shot 的内部标准单位，
  下游的「LensDistortion」「相机属性」吃的也是毫米；②「Filmback」输出口给的是**节点上那个参数的原值**——
  上游没有 Filmback 这一项；节点上的「Filmback」参数本身还是**输入侧**要的：用户填的 Focal Length（mm）要靠它换成像素
  才能当先验送进模型。③ 上游还有逐像素的视场场、置信度和 Focal Length 不确定度，没有开口。
- 上游一张画面一个答案；节点按「隔帧」抽帧各算一遍，逐帧交出来（手持镜头每一帧的倾斜都看得见）。

出处：简介来自 `third_party/geocalib/repo/README.md`；输入输出依据 `third_party/geocalib/repo/geocalib/extractor.py`、
`siclib/models/optimization/lm_optimizer.py` 和 `adapters/geocalib/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- **把解算放平、落地**：任何相机解算（ViPE、COLMAP、VGGT、Pi3、CUT3R……）出来的世界都是"跟着第一帧相机"的：第一帧相机低头 10°，整个世界的地面就斜 10°，进 Maya / Houdini 还要手动转。接法（模板「重力估算 · GeoCalib」）：读取序列 → GeoCalib 重力方向，它的「重力方向」「重力误差」接到「自动落地」的同名参数上（点参数旁边的接入按钮）；解算出的相机（或和点云打包后的场景）接「自动落地」的「场景」。它把每一帧的朝上方向用那一帧的相机转进世界，按重力误差加权平均（误差大的帧少算），把世界绕原点转到 Y 轴真正朝上（只转倾斜和俯仰，朝向不变），再按人物脚底或点云（取相机下方点云里最密的那一层）把地面放到 y = 0。
- **知道镜头就填 Focal Length**：填了「Focal Length」（和「Filmback」），或者接一个浮点（比如「拆分相机」拆出的 ViPE Focal Length），就固定用它，只估重力方向，更准；留空时它自己估 Focal Length（长焦会估短）。
- **隔帧**：默认每帧都估计，能看出手持镜头每一帧的倾斜；只为放平整个场景，长镜头设 5–10 就够。
- **拟合模型**：普通镜头、手机、长焦用「无畸变」；广角选「径向 k1」；GoPro、鱼眼选「鱼眼」。

## 效果和局限

RTX 4090 上：

| 素材 | 重力方向 | Focal Length | 用时 / 显存 |
|---|---|---|---|
| 固定机位跳舞，864×480，124 帧，室内有地面和墙 | 倾斜中位 −0.1°、俯仰 −1.3°，误差估计 ±1.4°，逐帧之间相差不到 0.2° | 612 px（35mm 等效约 27 mm，和预计的 560–590 px 相近） | 每帧 0.06 秒，显存 6.5 GB |
| iPhone 长焦跟拍，1080×1920，300 帧，走廊和花园 | 俯仰中位 −6.4°，误差估计 ±4.5°；少数帧（被人挡住、全是树叶）偏到 20–30°，「自动落地」放平时按离群帧去掉 | 3955 px，比 ViPE 的 5479 px 短 28%（长焦超出它的训练范围） | 每帧 0.16–0.25 秒 |
| 人脸特写，772×855 | 俯仰 −20°、误差 ±5.2°：画面里几乎没有直线，节点提示不可靠 | 1690 px | 每帧 0.06 秒 |

放平一个真实解算（Pi3 的相机和深度转成的点云，「合成场景」后接「自动落地」并接上重力方向和重力误差，隔 5 帧估计重力；用 RANSAC 在点云里找相机下方最大的平面，量它和水平面的夹角）：

| 素材 | 放平前地面倾斜 | 放平后 | 填上已知 Focal Length（ViPE 解出的）后 | 「自动落地」后地面高度 |
|---|---|---|---|---|
| 固定机位跳舞 | 2.1° | 0.7° | 0.2°（但 ViPE 在固定机位上解的 1054 px Focal Length 本身不可靠，只能当参考） | 0.05 cm |
| 长焦跟拍 | 5.0° | 3.7° | 3.6° | 2.5 cm（只有 4% 的点在这一层，节点提示可能不准） |

- 有地面、有墙的室内镜头放平得很好。长焦跟拍只改善了一点：长焦镜头里地面是斜着远远看过去的，Pi3 解出的地面本身就有几度的形变，这张表量的"地面倾斜"里有一部分是解算的误差，不全是没放平。
- 靠的是画面里的直线和地平线：城市、室内、有地面的外景最准；特写（整个画面是一张脸）、天空、树林里线索少，误差会大，节点会提示。
- 它的 Focal Length 不如 AnyCalib 准（论文和 InFlux 评测都是这样），要 Focal Length 优先用 AnyCalib；重力方向是 AnyCalib 没有的，这才是用它的原因。
- 放平的前提是相机解算本身转得对：各帧的朝上方向放进世界以后应该指向同一个方向，「自动落地」会报告它们相差多少，相差大说明相机和画面对不上。

## 团队

苏黎世联邦理工学院计算机视觉与几何组（Marc Pollefeys 教授）和微软混合现实与 AI 实验室合作，一作 Alexander Veicht。Paul-Edouard Sarlin 是 SuperGlue、LightGlue、hloc 的作者，Philipp Lindenberger 是 LightGlue 和 Pixel-Perfect SfM 的作者。这个组也是 COLMAP 现在的主要维护者之一。

## 模型下载和安装

- 运行 `lab2shot ext install geocalib`：下载锁定版本的 GeoCalib 代码、建独立环境（PyTorch 2.8，和 AnyCalib 同一版本，共用下载缓存），再从作者的 GitHub 发布页下载两个权重（普通镜头、畸变镜头，各 116 MB，校验 sha256）。
- 不需要申请权限，也不需要手动下载；运行时完全离线（原程序会在运行时从网上取权重，这里改成安装时下载好、从本地文件加载）。

## 许可证说明

- 代码 Apache-2.0，可以商用。
- 两个权重是同一个仓库的发布文件，没有另写许可证，按仓库的 Apache-2.0 理解。训练数据 OpenPano 来自 PolyHaven（CC0）、HDRMAPS 和 Laval 室内 HDR 数据集；Laval 数据集本身只许非商用，作者没有说明这是否限制权重的商用。商用项目请自己确认。

## 参考

- 论文：https://arxiv.org/abs/2409.06704
- 代码：https://github.com/cvg/GeoCalib （本扩展锁定 97b8968）
- 在线演示：https://veichta-geocalib.hf.space
- 许可证原文：https://github.com/cvg/GeoCalib/blob/main/LICENSE

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
