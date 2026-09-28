+++
team = "Meta 超级智能实验室（Meta Superintelligence Labs）"
people = "Xitong Yang, Devansh Kukreja, Don Pinkus, …, Jitendra Malik, Piotr Dollár, Kris Kitani"
paper = "SAM 3D Body: Robust Full-Body Human Mesh Recovery（2025）"
paper_url = "https://arxiv.org/abs/2602.15989"
website = "https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/"
repo = "https://github.com/facebookresearch/sam-3d-body"
year = 2025
+++

## 这是什么

SAM 3D Body（3DB）是可提示的单图全身三维人体网格恢复模型，在多种实拍条件下保持稳定精度。它以 Momentum Human Rig（MHR）参数化网格表示估计身体、脚与手的姿态，MHR 将骨架结构与表面形状解耦。模型采用编码器—解码器架构，支持 2D 关键点与遮罩等辅助提示。

在 Lab2Shot 中，该模型用于从实拍序列解算人物的全身动作（含手指）。结果输出为带蒙皮的角色——MHR 人体模型，127 个关节、18439 个顶点、带 UV，附逐帧骨骼动画——写成 USD，可用于 Houdini 与 Maya 的对位、参考动画或动捕起点。结果位于相机坐标系。

## 输入输出

### 官方实现

官方入口 `demo.py` 接收一个画面文件夹，逐张处理。人物框由内置的 ViTDet 检测器给出（`--detector_name` 默认 `vitdet`，阈值 `--bbox_thresh` 默认 0.8）；视场角由 MoGe-2 估计（`--fov_name` 默认 `moge2`）；`--use_mask` 启用时，以人物框生成的分割遮罩作为附加条件。

输出为每个人的 MHR 参数与网格，附内参 `focal_length` 与人在相机坐标系中的位移 `pred_cam_t`。官方不输出相机。

出处：`third_party/sam_3d_body/repo/demo.py`、`third_party/sam_3d_body/repo/sam_3d_body/sam_3d_body_estimator.py`。

节点的输入与输出端口、数据类型及其与官方符号的对应，见下方由节点声明生成的表；本节只说明官方实现与本节点的差异。

节点不设「相机」端口。官方既不接收相机，也不输出相机；要得到世界坐标系中的结果，在节点图上接「ViPE 相机解算」或「导入 USD」提供相机，再接核心节点「相机空间转换」。

### 与官方实现的差异

| 差异 | 说明 |
|---|---|
| 跨帧关联 | 官方逐张独立处理。本节点在其之上做同一人的身份跟随、整段体型锁定与姿态平滑，因此输出为连续动画。 |
| 未开放的提示 | 官方支持的 2D 关键点与遮罩提示尚未开放为端口。 |
| 未开放的输出 | 官方产出的 2D 关键点已随结果保存，尚未开放为输出端口。 |

## 在 Lab2Shot 里怎么用

- **典型接法**：读取序列（或 读取视频 → 视频转序列）→ SAM 3D Body 全身动作 → 合成场景 → 3D 变换 / 自动落地 → USD 输出设置（或 Alembic 输出设置）。拿不准的时候在视图里透过相机看，画面叠在结果后面，看解算出的人和原画面对不对得上。
- **节点上没有「相机」的进出口**：官方只吃 `cam_int` 这个内参、只给 `focal_length` 和 `pred_cam_t`，从来不吃也不给一台相机。**要世界空间的动作，把摆位做成图上看得见的一步**：先用「ViPE 相机解算」解出相机（它会自己找出画面里运动的人和物体，解算时避开），或者用「导入 USD」导入 3DE 等软件里跟好的相机，再接核心节点「相机空间转换」，把这个节点的「蒙皮角色」摆进那台相机的世界。内置模板「全身动作 · SAM 3D Body」就是第一种接法（ViPE 解相机 →「相机空间转换」）；要用自己跟好的相机，把 ViPE 换成「导入 USD」，那时不做去畸变——相机是从别处解算好导进来的，画面也用同一次解算的去畸变版本。
- **不接相机时**结果就留在相机空间：相机在原点不动，人在镜头前动。
- **「人物框」是可选输入口**：官方函数 `process_one_image(img, bboxes=None, …)` 本来就收框，官方脚本 `demo.py` 只是自己用 ViTDet 检出框再传给它。接了就按框解、不再自己检人；不接时画面里有几个人它自己检，检出几个就解几个。
- **只要画面里的某一个人**时：「ViTDet 人物框」→「选人」→ 这个节点的「人物框」口。另一条路是把别人从画面里去掉——「ViTDet 人物框」→「选人」→「人物框转遮罩」→「图像合成」（留下）→「RGB」口，相乘之后画面上只剩那个人，它自己的检测器自然只会找到他。
- **什么素材效果好**：人尽量全身入画、别太小；被遮挡或出画的部位只能靠模型猜。
- **关键参数**：
  - **已知 Focal Length + Filmback**：知道实拍镜头就填，人物的距离和大小会准很多。不填时会用 MoGe-2 在 12 帧上估计 Focal Length 再取中位数，但长焦镜头会估短（见下面的数字），人就会被放得太近太小。接一条 Focal Length 进来也行（一帧一个值，跟变焦）。
  - **平滑强度**：默认 0.5。0 保留全部细节但会抖；调到 1 很稳，但快速动作会变软、脚可能滑。
  - **手部精修**：默认打开，手指更准，每帧慢约 30%；手不重要或看不清时可以关掉。
- **导进 DCC**：USD 输出设置的单位选项里，给 Houdini 选「米」，给 Maya 选「厘米」。

## 效果和局限

RTX 4090 上：

- iPhone 长焦（等效 105 mm）手持跟拍走路，序列 1080×1920、300 帧，接 ViPE 解出的相机，解算 1 个人（开手部精修）：一共约 3 分钟（含加载模型，约 0.6 秒/帧）。在 Houdini 里检查过：人和画面对位准确，单位正确。
- 固定机位，864×480、124 帧，Focal Length 由 MoGe-2 自动估计：一共 84 秒（含估计 Focal Length 和加载模型）。
- 同一个长焦镜头，真实 Focal Length 约 5350–5600 像素，MoGe-2 只估到约 2700 像素，差了一半；ViPE 解出 5484 像素是对的。所以长焦镜头请手填 Focal Length，或者从解相机的节点把 Focal Length 接进来。

已知问题：

- 模型本身是单帧估计，快动作靠平滑压抖动。平滑开得大，脚会在地面上滑。
- 脚可能穿到地面以下或者悬空，后面接「自动落地」（把地面移到 y=0），或者用「3D 变换」手动调。
- 某些帧没检测到人时，这些帧不补（不插值），角色在这些帧上隐藏，节点会提示有多少帧没有结果；整段几乎没解出几帧的人不输出（「最少解出帧数」）。
- USD 里的蒙皮是 MHR 自己的线性蒙皮，没带姿态修正形变（pose correctives），关节弯得很厉害的地方和原版有一点点差别。
- 不输出面部表情；脸的细节要用面部类节点。**官方自己就把表情和下巴的预测置零**（`sam_3d_body/models/heads/mhr_head.py`：`pred_pose_euler[:, -3:] = 0`、`pred_face = pred[...] * 0`），所以 MHR 模型有表情通道，官方权重却从来不填。

## 团队

Meta 超级智能实验室（Meta Superintelligence Labs）出品，作者里有 Jitendra Malik、Piotr Dollár、Kris Kitani 等计算机视觉领域的资深研究者。Meta 这条线做过 Segment Anything 系列（SAM、SAM 2、SAM 3），是现在很多 AI 抠像、Roto 工具的底子。人物检测用的 ViTDet 来自 Meta 的 Detectron2。和 SAM 3D Body 一起开源的还有 MHR（Momentum Human Rig）人体模型，骨骼和体表是分开建模的。

## 模型下载和安装

- **自动安装**：运行 `uv run lab2shot ext install sam_3d_body`，会下载：
  - SAM 3D Body 权重（facebook/sam-3d-body-dinov3，约 2.7 GB，里面带 MHR 人体模型）；
  - ViTDet-H 人物检测权重（约 2.6 GB，Meta 服务器）；
  - MoGe-2 Focal Length 估计权重（约 1.3 GB）；
  - 一个独立的 Python 环境（含 PyTorch，约 7 GB），并编译 detectron2。只编译 C++ 部分，不用装 CUDA 工具包，但本机要有 C++ 编译器（gcc）。
  - 权重合计约 6.5 GB，加上环境一共约 14 GB 硬盘。
- **需要申请权限（Hugging Face gated）**：SAM 3D Body 权重要先申请。
  1. 登录 Hugging Face，打开 https://huggingface.co/facebook/sam-3d-body-dinov3 ；
  2. 按页面提示填写表单（同意共享联系信息）并提交，等 Meta 审批（通常很快）；
  3. 在这台机器上登录 Hugging Face：`uv run hf auth login`，粘贴一个有 Read 权限的 Access Token；
  4. 再运行一次 `uv run lab2shot ext install sam_3d_body`，下到一半的会接着下。
- **解算不联网**：原版代码运行时会从 GitHub 拉取 DINOv3 网络结构的代码；Lab2Shot 安装时就按固定版本把这份代码下载好，解算时用本地的这份。

## 许可证说明

- **SAM 3D Body（代码和权重）：SAM License，可以商用**，可以修改和再分发，再分发时要附上许可证原文；用它做研究发论文，要注明用了 SAM 3D Body。
- **禁止用途**：军事和战争、核工业、间谍活动、枪支和非法武器，以及受美国《国际武器贸易条例》（ITAR）管制的用途；使用时要遵守美国等国家的出口管制和制裁规定。
- 如果你对 Meta 就这些材料提起专利诉讼，Meta 给你的专利许可会终止。软件不提供任何担保。
- 一起用到的其他模型：ViTDet（Detectron2，Apache-2.0）、MoGe-2（MIT）、MHR 人体模型（Apache-2.0），都可以商用。

## 参考

- 论文：https://arxiv.org/abs/2602.15989
- Meta 论文页：https://ai.meta.com/research/publications/sam-3d-body-robust-full-body-human-mesh-recovery/
- 发布博客（SAM 3D Objects + SAM 3D Body）：https://ai.meta.com/blog/sam-3d/
- 代码仓库：https://github.com/facebookresearch/sam-3d-body
- 许可证原文：https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE
- 模型卡（本扩展用的 DINOv3 版）：https://huggingface.co/facebook/sam-3d-body-dinov3
- 模型卡（ViT-H 版，本扩展没用）：https://huggingface.co/facebook/sam-3d-body-vith
- 在线演示：https://www.aidemos.meta.com/segment-anything/editor/convert-body-to-3d
- MHR 人体模型：https://github.com/facebookresearch/MHR
- Detectron2（ViTDet 检测器）：https://github.com/facebookresearch/detectron2
- MoGe（Focal Length 估计）：https://github.com/microsoft/MoGe
