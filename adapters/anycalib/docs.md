+++
team = "西班牙萨拉戈萨大学 I3A 研究所（University of Zaragoza, I3A）"
people = "Javier Tirado-Garín, Javier Civera"
paper = "AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera Calibration（ICCV 2025）"
paper_url = "https://arxiv.org/abs/2503.12701"
website = ""
repo = "https://github.com/javrtg/AnyCalib"
year = 2025
+++

## 这是什么

上游项目页给自己写的一句话是：从一张透视、编辑过或带畸变的画面做相机标定，镜头模型由使用者
自由选定。论文题目是《AnyCalib: On-Manifold Learning for Model-Agnostic Single-View Camera
Calibration》。按上游 README 的最小示例，网络回归的是视场角场（FoV field）以及与之对应的、
相机坐标系下的射线方向，再按选定的镜头模型解出内参。

在 Lab2Shot 里，它是「AnyCalib 镜头标定」：不用拍棋盘格，从画面估出 Focal Length、主点和镜头畸变，
可选的镜头模型有 6 类（无畸变、Brown 径向、除法模型、Kannala-Brandt 鱼眼、UCM、EUCM），
每类系数个数不同，一共 30 种。结果落成一台固定不动的相机（USD，带 Focal Length，畸变模型和
系数也记在相机上），另外给两张 ST-map（「去畸变」和「加畸变」），Nuke 的 STMap 节点拿来就能用，
Houdini 里也能用。整段镜头只算一个镜头，适合定焦镜头的素材。

## 输入输出

**官方要什么、给什么**

- 吃：**一张**画面，`(3, H, W)` 的张量，RGB，值在 0–1；调用时还要选一种镜头模型
  `model.predict(image, cam_id="pinhole")`。
- 给：一个字典——`intrinsics`（`(D,)`，按所选镜头模型的参数顺序排的内参，
  Focal Length、主点、畸变系数都在这一串里）、`fov_field` / `tangent_coords`（网络回归的视场场）、
  `rays`（每个像素的射线方向，x 右、y 下、z 朝前）、`pred_size`（网络实际用的画面尺寸）。

**我们怎么接的**

- 「图像」口就是上游的 `im`；节点上的**「拟合模型」**参数（`fit_model`）就是官方的 `cam_id`，直接用官方名字
  （simple_radial:1、simple_kb:4……）。提供哪几个、各对应核心公式表的哪个模型，在 `adapters/anycalib/lens.py` 的
  GROUP 里；没提供的官方模型也在那里留了档。
- 「Focal Length」、「主点 X / Y」、「畸变系数」四个口都是从那一串 `intrinsics` 里按镜头模型的顺序取出来的，
  「镜头模型」口（`lens_model`）就是请求的 `cam_id` 本身。
- **「拟合模型」是要求，「镜头模型」是结果**：前者是节点上的参数（送进上游当 `cam_id`），后者是输出口。
- **「镜头模型」「畸变系数」两个口只在带畸变的那几档下用得上**：「拟合模型」选到「无畸变」时，上游根本没解畸变，
  那两个口没有值——它们在节点上**变灰、鼠标停上去写清「「拟合模型」选…时才用」、线也接不出去**（口不消失，位置不跳）。
  **「主点 X / Y」不变灰**：那一串 `intrinsics` 里每种镜头模型都有主点。
- 上游只给一张画面的结果；节点按「采样帧数」抽几帧各算一遍再取中位数，整段一个答案。
- **不一样的两处**：①**单位换过**——上游那一串 `intrinsics` 里的 Focal Length 和主点都是**像素**，
  「Focal Length」「主点 X」「主点 Y」三个口交的是**毫米**，用节点上的「Filmback」参数换算
  （毫米 = 像素 ÷ 画面宽度 × Filmback）。毫米是 Lab2Shot 的内部标准单位，下游的「LensDistortion」
  「相机属性」吃的也是毫米，接上去两边对得上；②「Filmback」输出口给的是**节点上那个参数的原值**，
  不是上游算的——上游没有 Filmback 这一项，那个口只是把这个值带给下游，省得在两个节点上各填一遍。
- 上游还有、节点没有开口的：`fov_field`、`rays`、`pred_size`。

出处：简介来自 `third_party/anycalib/repo/README.md`；输入输出依据同一份 README 和
`third_party/anycalib/repo/anycalib/model/anycalib_pretrained.py`，以及 `adapters/anycalib/nodes.py`、`worker.py`、`lens.py`。

## 在 Lab2Shot 里怎么用

- **典型接法**：读取序列 → AnyCalib 镜头标定 → 两张 ST-map 各接一个「序列图输出设置」，相机接「USD 输出设置」。内置模板「镜头标定 · AnyCalib」就是这样接的。
- **给其他节点当 Focal Length 来源**：相机输出可以接到「SAM 3D Body 全身动作」的相机输入上，这时 SAM 3D Body 用 AnyCalib 的 Focal Length，不再自己估。
- **在 Nuke 里用**：「去畸变」ST-map 把原素材拉直，得到理想针孔画面（和 USD 相机对得上）；CG 渲染完再用「加畸变」ST-map 把畸变加回去，和实拍对上。两张图的 A 通道标出了哪里有有效映射。ViPE 等相机解算不处理镜头畸变，畸变明显的素材可以先在 Nuke 里去畸变，再拿去解算。
- **什么素材效果好**：画面里有直线（建筑、室内、地砖）的最准。镜头不能变焦。整段在几帧上分别估计后取中位数，所以画面内容变化大（一会儿空镜、一会儿特写）时，多取几帧。
- **关键参数**：
  - **拟合模型**：普通镜头用默认的 simple_radial:1；广角 simple_radial:2；手机和长焦镜头畸变很小，用 simple_pinhole 更稳；GoPro、鱼眼用 simple_kb:1 到 :4（Kannala-Brandt）。交出去的「镜头内参」带着组名 AnyCalib 和这个模型名：「LensDistortion」上「镜头内参组」选 AnyCalib、「镜头模型」选同一个，两边才对得上（不一样提交前拦下）。
  - **网络权重**：「通用」什么镜头都能用（推荐）；「无畸变画面」只在普通 / 长焦镜头上训练过，Focal Length 略准；「鱼眼 / 强畸变」给 GoPro、鱼眼用。
  - **主点**：实拍素材基本在画面中心，选「画面中心」更稳；只有裁切过的素材才选「估计」。
  - **Filmback (mm)**：上游给的 Focal Length 和主点都是像素，「Focal Length」「主点 X / Y」三个口都按这个 Filmback 换算成毫米交出去（毫米 = 像素 ÷ 画面宽度 × Filmback）。全画幅填 36；手机按等效 Focal Length 填 36。「Filmback」输出口交的就是这里填的值。

## 效果和局限

RTX 4090 上：

- **速度**：一段镜头（默认取 8 帧）4–10 秒，显存 1.3–1.8 GB。
- **OpenCV 自带的棋盘格样片**（640×480，普通带桶形畸变的镜头）：用棋盘格标定出来的真值是 Focal Length 约 536 像素、k1 约 −0.26。AnyCalib 估出 Focal Length 512 像素（偏小约 5%），k1 −0.18（方向对，程度偏弱）。
- **iPhone 长焦**（等效 105 mm，1080×1920）：真实 Focal Length 约 5350–5600 像素，AnyCalib 估出 4100–4230 像素，**偏小 20–25%**。换网络权重或镜头模型，结果都在 4130–4300 之间。同一个镜头 ViPE 相机解算得到 5484 像素，是对的。
- **鱼眼（TUM-VI room1，512×512，真值 Kannala-Brandt 4 项，fx 191.0，视场 195°，40 帧）**，`simple_kb:4`、通用权重、主点画面中心：
  Focal Length 195 px（真值 191，偏 2%）；k1 −0.0082 k2 0.0050 k3 0.0007 k4 −0.0006（真值 0.0035 0.0007 −0.0021 0.0002）。
  按像素半径比（同一入射角下两条 KB 曲线的差）：80° 以内最大 6.4 px、中位 2.5 px；到 95° 最大 10.9 px。
  同一段素材 COLMAP 两段式解出来的是 0.2 px 以内——AnyCalib 是单张图估的，只能当起点：
  接「LensDistortion」后节点会说「畸变是估算的」（P-LENS-ESTIMATED），交付前换 COLMAP 解的或剧组的镜头数据。
  「镜头内参」→「LensDistortion」（组 AnyCalib，模型 simple_kb:4）→ 两张 ST-map 整条链可用；组选错（如 COLMAP）时提交前拦下（B-LENS-MODELMISMATCH）。

所以：

- 长焦镜头上它会把 Focal Length 估短。知道实拍 Focal Length 时，以实拍记录为准。
- 它只看单帧画面，不看运动，整段镜头按一个镜头处理，变焦镜头不行。
- 相机是固定不动的，只提供镜头参数。相机运动要用 ViPE、COLMAP 解算，或者导入跟好的相机。
- 除法模型的 4 系数版本在轻微畸变的画面上可能拟合失败，节点会报出来。

## 团队

西班牙萨拉戈萨大学（University of Zaragoza）I3A 研究所的 Javier Tirado-Garín 和 Javier Civera。Civera 是该校机器人、感知与实时组（Robotics, Perception and Real-Time Group）的 SLAM 实验室的副教授，研究方向是运动恢复结构（SfM）和 SLAM。他博士期间提出的「逆深度参数化」是单目 SLAM 里的经典方法。萨拉戈萨大学的 SLAM 实验室还做过开源视觉 SLAM 系统 ORB-SLAM3，相机跟踪领域很有名。

## 模型下载和安装

- **自动安装**：运行 `uv run lab2shot ext install anycalib`，会下载：
  - 3 个 AnyCalib 权重（通用 gen、无畸变 pinhole、鱼眼 / 强畸变 dist），每个约 1.28 GB，一共约 3.6 GB。从 Hugging Face 的 javrtg/AnyCalib 按固定版本下载，下完会做校验，文件不对会删掉让你重下。官方另外还有一个给「拉伸 / 裁切过的图片」用的 edit 权重，没有接。
  - 一个独立的 Python 环境（含 PyTorch，约 6.5 GB），不用编译。
  - 网络主干 DINOv2 ViT-L 的权重已经包含在 AnyCalib 权重里，不另外下载。
- 不需要申请权限。

## 许可证说明

- **代码和权重都是 Apache-2.0，可以商用**，可以修改和再分发，再分发时保留许可证和版权声明。
- 训练数据里有 Laval 室内 HDR 数据集，作者说明已经得到对方许可，所以权重可以用宽松许可证发布。其他训练全景图来自 PolyHaven、HDRMaps、AmbientCG、BlenderKit 等免费资源。

## 参考

- 论文：https://arxiv.org/abs/2503.12701
- 代码仓库：https://github.com/javrtg/AnyCalib
- 许可证原文：https://github.com/javrtg/AnyCalib/blob/main/LICENSE
- 模型权重：https://huggingface.co/javrtg/AnyCalib
- 在线演示：https://huggingface.co/spaces/javrtg/AnyCalib
- Javier Civera 主页：https://webdiis.unizar.es/~jcivera/
- ORB-SLAM3（萨拉戈萨大学 SLAM 实验室）：https://github.com/UZ-SLAMLab/ORB_SLAM3
