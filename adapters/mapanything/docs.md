+++
team = "Meta + 卡内基梅隆大学（CMU）"
people = "Nikhil Keetha, Norman Müller, Johannes Schönberger, Lorenzo Porzi, …, Sebastian Scherer, Peter Kontschieder"
paper = "MapAnything: Universal Feed-Forward Metric 3D Reconstruction（3DV 2026）"
paper_url = "https://arxiv.org/abs/2509.13414"
website = "https://map-anything.github.io/"
repo = "https://github.com/facebookresearch/map-anything"
year = 2025
+++

## 这是什么

上游自己的话（Overview）：MapAnything 是一个**开源研究框架**，做通用的真实尺度三维解算。它的核心是一个简单的、端到端训练的 transformer 模型：给定各种输入（画面、标定、位姿或深度图），一步回归出场景分解之后的真实尺度三维几何。单个前馈模型支持 12 种以上的三维解算任务，包括多图 SfM、多视角立体、单目真实尺度深度估计、配准、深度图补全等。

在 Lab2Shot 里，它是「MapAnything 深度与相机」：交出每一帧的相机（位置、朝向、**每帧各自的 Focal Length**）和深度图，上游给的单位是米，换算成厘米交出。因为不假设整段只有一个 Focal Length，它能处理**变焦镜头**；知道实拍 Focal Length 时填进去，它按这个 Focal Length 算。

## 输入输出

**官方要什么、给什么**

- 吃：一组画面。`model.infer` 的必需输入只有两样：`img`（按 `data_norm_type` 归一化的 RGB）和 `data_norm_type`。
  可选的几何输入：`intrinsics` 或 `ray_directions`（二选一，不能都给）、
  `depth_z`（要配着标定给）、`camera_poses`（OpenCV 约定的 cam2world 4×4）、`is_metric_scale`。
  官方说单个前馈模型支持 12 种以上的三维解算任务。
- 给：每一视图一份结果（`mapanything/models/mapanything/model.py`）——
  `pts3d`（世界点图）、`pts3d_cam`（同一份点图的相机空间形式）、`intrinsics`、`depth_z`、`camera_poses`、置信度。

**我们怎么接的**

- 「RGB」口就是 `img`，「隔帧」「每段最多帧数」是节点分段计算的设置，不是模型的输入。
- 「深度图」= `depth_z`、「相机」= `camera_poses` 加 `intrinsics`、「点云」= 官方的 `pts3d`
  （worker 交的是它的相机空间形式 `pts3d_cam`，家族再按拼好的相机摆回世界——同一份数据换坐标系）。
- **上游可选、节点只用一条**：`intrinsics` 这一路对应节点上的「已知 Focal Length」「Filmback」两个参数
  （填了就把内参送进去）。`ray_directions`、`depth_z`、`camera_poses`、`is_metric_scale`
  这几样没有接口——要接就得是显式的一条线，现在不接。
- **上游没有、节点上也没有**：遮罩和人物框输入口。官方 view 字典里可选的只有内参 / 射线 / 深度图 / 位姿，没有任何遮罩。

出处：简介来自 `third_party/mapanything/repo/README.md`（Overview）；输入输出依据同一份 README、
`mapanything/models/mapanything/model.py` 和 `adapters/mapanything/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → MapAnything 深度与相机 → 输出相机、深度图和点云；和 ViPE 相机解算 / COLMAP 的结果对比，互相校验。相机和点云可以导出 USD 到 Houdini、Maya 里当参考。「点云」口交的是官方自己的点图（`pts3d_cam`，官方世界点图 `pts3d` 的相机空间形式），不是从深度图反投影的。
- **权重选 main**（非商用）效果明显好得多；apache（可商用）在长焦镜头上基本不能用，普通镜头也把 Focal Length 估得偏短（见下面的数字）。
- 知道 Focal Length 时一定填 `focal_px`（输入分辨率下的 Focal Length（px）= Focal Length mm ÷ 传感器宽mm × 画面宽像素）：main 权重的相机轨迹误差能再减一半。
- 适合：普通到中长焦、有一定视差的镜头、固定机位（相机基本不漂）。长焦要格外小心：不填 Focal Length 时 main 把 iPhone 5 倍长焦估短了约 25%。镜头畸变不处理，大畸变素材先去畸变。
- `max_frames`：一次送进网络的帧数，默认按 24 GB 显存自动定（16:9 或 9:16 画面 150 帧，4:3 约 112 帧）。镜头更长时自动分段，相邻段重叠 4–16 帧，用重叠帧的稠密点云做相似变换（平移、旋转、缩放）对齐，整段共用一个世界坐标。`step` 隔帧取样（每 N 帧算一帧，中间帧的相机由前后插值），长镜头可以用 2–3 省时间。
- **手填 `max_frames` 的上限是 200 帧**：24 GB 显卡上 150 帧 18.5 GB、200 帧 23.5 GB，已经贴着显存上限、没有余量；服务器按同样的上限拒绝更大的数字。填到上限时同一张卡上不要再跑别的重任务。
- 模型本身不接受遮罩，节点上也没有遮罩 / 人物框输入口：画面里的人照样参与计算，结果里也不会把它们抹掉。想只算画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口。

## 效果和局限

RTX 4090 上（main / apache 两套权重对比）：

- **iPhone 5 倍长焦跟拍**（1080×1920，150 帧，真实 Focal Length 约 5484 px；参考：ViPE 相机解算 Focal Length 5133 px，走了约 5 m）：
  - main：Focal Length 4141 px（3946–4198，偏短 24%），相机轨迹和 ViPE 对齐后误差 RMS 18 cm（占路径 3.7%），朝向差中位数 1.3°。
  - main + 填入 Focal Length 5484：误差 RMS 10 cm（2.1%），朝向差 0.55°。
  - main 分两段（每段 83 帧）：误差 RMS 26 cm（5.2%），两段拼接残差 1.1%，但两段的尺度差了 14%（前馈模型每段自己估真实尺度）。
  - apache：Focal Length 1691 px（偏短 69%，逐帧乱跳 1285–1776），轨迹误差 68 cm（13.7%），朝向差 7.6°；填了 Focal Length 它也几乎不理会（自己仍估 1749 px）。**长焦镜头不要用 apache。**
  - 前面那位女士的距离：main 7.2–10.1 m，main + Focal Length 9.1–11.7 m，apache 约 3.4 m。按真实 Focal Length 和她在画面里的高度（按 1.6 m 身高）推算约 9–12 m，main + Focal Length 最接近。
- **固定机位跳舞**（864×480，124 帧）：main Focal Length 648 px（638–664），相机最大漂移 6.7 cm / 0.36°；apache Focal Length 315 px（明显偏短），漂移 10.8 cm / 1.3°。ViPE 在这种不动的镜头上 Focal Length 解成 1053 px（没有视差，解不准），MapAnything 反而更合理。
- 速度：150 帧 1080×1920 全程约 35 秒（载入模型 3–5 秒、网络推理约 10 秒，其余是读图和把深度图放大回原分辨率写盘）；300 帧自动分 3 段约 70 秒。
- 显存：一次 150 帧峰值 18.5 GB，200 帧 23.5 GB（24 GB 卡的上限），300 帧放不下。内存约 5.5 GB。
- 已知问题：网络在 518 像素左右的小图上计算，深度图放大回原分辨率后边缘偏软；帧与帧之间没有平滑，轨迹有明显的逐帧抖动（main 150 帧的路径长度是平滑轨迹的两倍多），做最终相机前需要平滑；分段处可能有尺度跳变：长焦跟拍整段 300 帧分 3 段时，镜头转了 46° 的最后一段和前一段尺度差了 0.72 倍（节点会报警告），结果里会报告每段的对齐误差和尺度。

## 团队

Meta 和卡内基梅隆大学（CMU）机器人研究所 AirLab 合作完成，一作 Nikhil Keetha（CMU 博士生，也做过 SplaTAM、AnyLoc）。作者里的 Johannes Schönberger 是传统反求软件 COLMAP 的作者；Peter Kontschieder、Lorenzo Porzi、Samuel Rota Bulò 来自原 Mapillary 研究团队（Mapillary Vistas 街景数据集）；Sebastian Scherer 的实验室做过 TartanAir 仿真数据集，模型的积木库 UniCeption 也出自这个实验室。

## 模型下载和安装

- 运行 `lab2shot ext install mapanything`：下载 MapAnything 代码（锁定版本）、建独立 Python 环境（PyTorch 2.9，约 5.4 GB，和其他扩展共用下载缓存），再下载两套权重各 4.9 GB（apache 和 main，都不需要申请）以及 DINOv2 的网络结构代码（3 MB）。网速正常时十来分钟；下载完会自动校验文件完整性。
- 不需要自行下载任何东西。

## 许可证说明

- 代码：Apache-2.0，可以商用、修改、再分发（附带许可证原文）。
- apache 权重（facebook/map-anything-apache）：Apache-2.0，**可以商用**。按官方训练脚本，它只用许可证宽松的 6 个数据集训练（BlendedMVS、Mapillary 深度、ScanNet++、Spring、TartanAirV2、UnrealStereo4K）。
- main 权重（facebook/map-anything）：CC BY-NC 4.0，**非商用**，只能用于研究等非商业用途，使用时要注明出处。按官方训练脚本，它在上面 6 个之外又多用了 7 个数据集（共 13 个，如 MegaDepth、DL3DV、Aria 合成场景、Dynamic Replica 等，其中有只许非商用的），所以效果更好。两套权重结构完全一样，只是训练数据不同。
- 依赖：UniCeption 为 BSD-3-Clause；DINOv2 结构代码为 Apache-2.0（DINOv2 仓库里另有 Cell-DINO / XRay-DINO 非商用代码和权重，本扩展不用也不下载）。

## 参考

- 论文：https://arxiv.org/abs/2509.13414
- 项目主页：https://map-anything.github.io/
- 代码：https://github.com/facebookresearch/map-anything
- 许可证原文：https://github.com/facebookresearch/map-anything/blob/main/LICENSE
- main 权重模型卡（CC BY-NC 4.0）：https://huggingface.co/facebook/map-anything
- apache 权重模型卡（Apache-2.0）：https://huggingface.co/facebook/map-anything-apache
- CC BY-NC 4.0 许可证：https://creativecommons.org/licenses/by-nc/4.0/
- 在线演示：https://huggingface.co/spaces/facebook/map-anything
- UniCeption：https://github.com/castacks/UniCeption
- DINOv2：https://github.com/facebookresearch/dinov2
