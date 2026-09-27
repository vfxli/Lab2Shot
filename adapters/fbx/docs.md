+++
team = "Autodesk"
people = ""
paper = ""
paper_url = ""
website = "https://aps.autodesk.com/developer/overview/fbx-sdk"
repo = "https://github.com/pybind/pybind11"  # FBX SDK has no public repository: the code this extension pulls is pybind11
year = 2020
+++

## 这是什么

Autodesk 自己的说法：FBX 是免费、跨平台的三维创作与交换格式，能取用绝大多数三维厂商和平台的三维内容；FBX 文件格式支持全部主要的三维数据，也支持 2D、音频和视频媒体。Maya、MotionBuilder、3ds Max、Unreal、Unity 都用它交换带骨骼的角色和动画。它不是机器学习项目，没有论文，也没有模型。

在 Lab2Shot 里，它是一个三维文件格式模块，用 Autodesk 官方的 FBX SDK 读写，给两个节点：**导入 FBX**（按层级列出 FBX 文件里的相机、模型、骨架动画和蒙皮角色，选哪些就输出哪些）和 **FBX 输出设置**（把相机、静止的模型、骨架动画和蒙皮角色——蒙皮、骨骼动画、blend shape 和它的权重曲线——写成 FBX）。

FBX 没有点云、也没有逐帧的顶点缓存：导入节点里没有点云；输出设置连上点云或逐帧变形的模型时是红线，提示改接「USD 输出设置」或「Alembic 输出设置」。

## 输入输出

**官方要什么、给什么**

- Autodesk 对 FBX SDK 的说明：「Use this free C++ SDK to build plug-ins, converters, and applications
  that translate and exchange 3D assets using FBX technology」（官方开发者页）。
- SDK 这一层：`.fbx` 文件是一棵节点树（`FbxNode`），每个节点挂一样数据——
  `FbxMesh`（网格，可带 `FbxSkin` 蒙皮和 `FbxBlendShape` 形变）、`FbxSkeleton`（骨骼，分
  root / limb / limbNode / effector）、`FbxCamera`（相机，Focal Length 等属性可以有动画曲线）、
  `FbxLight`（灯光）、NURBS 和细分面等；动画按 `FbxAnimStack`（动画段）分层存。
  单位和上轴写在文件的 `FbxGlobalSettings` 里。

**我们怎么接的**

- 「导入 FBX」读 `FbxCamera` → 「相机」、`FbxMesh` → 「模型」、`FbxSkeleton` → 「骨架动画」、
  带 `FbxSkin` / `FbxBlendShape` 的网格 → 「蒙皮角色」，「动画段」参数就是 `FbxAnimStack`（`adapters/fbx/fbxio.cpp`）。
- 「FBX 输出设置」把场景写成 `.fbx`，按 Lab2Shot 的内部标准写：厘米、Y 轴向上。
- **不一样的地方**：① `FbxLight`、NURBS、细分面不读——Lab2Shot 的数据种类里没有它们；
  ② 文件里的单位和上轴按 `FbxGlobalSettings` 换算到内部标准，原文件不改；
  ③ SDK 本身不在任何包索引上：由用户本人在「手动下载」里同意 Autodesk 的协议后装，
  安装这个扩展时编译一个小模块去调它。

出处：简介来自 Autodesk 开发者页 https://aps.autodesk.com/developer/overview/fbx-sdk ；
输入输出依据 Autodesk FBX SDK 的类型（`FbxNode` / `FbxMesh` / `FbxSkin` / `FbxBlendShape` / `FbxSkeleton` /
`FbxCamera` / `FbxAnimStack`）和 `adapters/fbx/fbxio.cpp`、`worker.py`。

## 在 Lab2Shot 里怎么用

- **导入 FBX**：
  - 用「文件」的按钮选 .fbx。「相机」「模型」「骨架动画」「蒙皮角色」里列出文件里的每一项：它在大纲里的完整路径（如 `/World/set/table`，角色写它的根关节），后面是分得开它们的说明：帧范围、Focal Length、点数、关节数、网格数和 blend shape 数。
  - 相机选一台；别的选任意几个（勾选，另有「全部」）。某一种文件里只有一个时自动选上；选了哪种才有哪种的输出口。换了文件、原来选的那项没有了，节点标红写明原因。
  - 骨架动画是只有关节的（动捕、MotionBuilder 的骨架）；蒙皮角色是蒙皮到骨骼上的网格，blend shape 和它的动画一起带进来。补帧节点（Kimodo、Two-stage Transformer）两种都收。
  - **单位和上轴**：FBX 文件自己记着（Maya 厘米 Y 轴向上、3ds Max 英寸 Z 轴向上……），自动换算成厘米、Y 轴向上，不用设。帧率也按文件的。
  - 所有父级的变换（包括动画）逐帧乘进去；相机去掉父级的缩放；有目标点（look-at）的相机按目标算朝向。
- **FBX 输出设置**：
  - 写厘米、Maya 的 Y 轴向上坐标系（在文件里明确写出），Maya、MotionBuilder、Unreal 打开时不用再转。文件是二进制 FBX 7.x。
  - 蒙皮角色：骨骼、每帧的骨骼动画、蒙皮权重、blend shape 和每帧的权重曲线都保留，在 DCC 里还能改动画。
  - 模型只写静止的。逐帧变形的模型（人体解算的「网格」、「烘焙成模型」的结果）FBX 装不下，请接蒙皮角色，或者改用 USD / Alembic。

## 效果和局限

- 只用 CPU，很快。
- 测试：`tests/integration/test_scene_formats.py` 用 FBX SDK 写一个像 DCC 里做出来的镜头（层级里各处的相机、静止的模型、两个带 blend shape 的蒙皮角色、一个只有骨骼的动捕骨架，都挂在有动画的父级下；厘米 Y 轴向上和米 Z 轴向上两种），导入后和独立算出的真值逐帧比对；还有和 USD、Alembic 之间互相转。`tests/integration/test_fbx_worker.py` 检查 worker 自己：写出的文件用 FBX SDK 读回来，检查单位、坐标系、骨骼、蒙皮、blend shape 的权重曲线和相机。
- **没有在 Maya / Unreal 里核对过**：FBX 相机默认沿自己的 +X 轴看，Lab2Shot 的相机沿 -Z 看，读写时各转 90 度（按 FBX SDK 自己的约定）。测试都是用 FBX SDK 读回来比对的，没有在 Maya、Unreal 里打开核对过。
- 蒙皮按 FBX SDK 的线性蒙皮（clusters）读写；双四元数蒙皮、约束、IK 不带。

## 团队

FBX 最早是 Kaydara 为 MotionBuilder 设计的格式，2006 年 Autodesk 收购 Kaydara 之后成了 Autodesk 的格式，由 Autodesk 维护 FBX SDK。Lab2Shot 用的是 FBX SDK 2020.3。

## 模型下载和安装

没有模型要下载，但要先装 FBX SDK：

- FBX SDK 不在任何软件源里，要你自己下载，并本人同意 Autodesk 的许可协议。到 Autodesk 的 FBX SDK 页面下载 Linux 版（`fbx2020310_fbxsdk_gcc_linux.tar.gz`），原样放进项目的 `downloads/` 收件文件夹；在「帮助与扩展包」首页的「手动下载」里看许可协议，点「同意并安装」，它装到 `third_party/_fbx_sdk/2020.3.10/`。
- 然后运行 `uv run lab2shot ext install fbx`：建一个 conda-forge 环境（Python 3.12、numpy、FBX SDK 要的 libxml2 2.x），用 pybind11 编译一个小模块（`adapters/fbx/fbxio.cpp`），把 FBX SDK 静态链接进去。编译约 7 秒，环境约 370 MB。
- **需要本机有 C++ 编译器**，在 `config/local.toml` 的 `[build]` 段里设置 `cxx`（例如 `g++-14`）。

## 许可证说明

- **FBX SDK 属于 Autodesk**，按 Autodesk FBX SDK 许可协议使用：由你本人在页面上看过并同意（谁、什么时候、哪份原文都记在数据库里）。协议允许用于开发、研究、内部、教育或商业用途；SDK 本身不能再分发，所以仓库和安装包里没有任何 Autodesk 的文件。
- 编译用的 pybind11 是 BSD-3-Clause。

## 参考

- FBX SDK：https://aps.autodesk.com/developer/overview/fbx-sdk
- 许可协议：https://www.autodesk.com/developer-network/platform-technologies/fbx-sdk-license
- pybind11：https://github.com/pybind/pybind11
