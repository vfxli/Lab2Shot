+++
team = "索尼图形图像运作（Sony Pictures Imageworks）+ 工业光魔（Industrial Light & Magic / Lucasfilm）"
people = ""
paper = ""
paper_url = ""
website = "http://www.alembic.io/"
repo = "https://github.com/alembic/alembic"
year = 2011
+++

## 这是什么

官方项目页 alembic.io 这样介绍它：Alembic 是一个开放的图形交换框架，把复杂的动画场景蒸馏成
一份非过程化、与软件无关的烘焙几何结果。它只高效存放复杂过程化几何运算算出来的结果——譬如
动画和解算之后的顶点位置与变换——不存产生这些结果的那张运算网络，也就是 rig。

在 Lab2Shot 里，Alembic 不是机器学习项目，没有论文也没有模型，而是一个 3D 文件格式模块，
给出两个节点：「导入 Alembic」按层级列出 .abc 里的相机、模型、点云和三维曲线，选哪些就输出哪些；
「Alembic 输出设置」把相机、模型（静止的或逐帧变形的点缓存）、点云和三维曲线写成 .abc，给 Maya、Houdini
等软件用。因为 Alembic 不存 rig，骨架动画和蒙皮角色装不下：导入节点里没有这两种，
输出设置连上它们时是红线。

## 输入输出

**官方要什么、给什么**

- Alembic 官网对它自己的定位：存的是「复杂动画和仿真算完之后的顶点位置和变换动画」，
  **明确不存**产生这些结果的那张依赖网络（也就是绑定）。
- 库这一层：`.abc` 文件是一棵对象树，每个对象有自己的 schema——
  `IXform`（变换）、`ICamera`（相机）、`IPolyMesh`（多边形网格）、`IPoints`（点）、`ICurves`（曲线）、
  `IFaceSet`（面集）、`ISubD`（细分面）、`INuPatch`（NURBS 面片）、`ILight`（灯光），
  见仓库 `lib/Alembic/AbcGeom/`。采样按时间采样（TimeSampling）记，和帧率分开。

**我们怎么接的**

- 「导入 Alembic」读 `ICamera` → 「相机」、`IPolyMesh`（含 `IFaceSet` 的面集 → 分区）→ 「模型」、
  `IPoints` → 「点云」、`ICurves` → 「三维曲线」，层级由 `IXform` 链算出。
- 「Alembic 输出设置」把场景写成 `.abc`：网格写 `OPolyMesh`（分区写成 `OFaceSet`）、相机写 `OCamera`。
- **不一样的地方**：① `ISubD`、`INuPatch`、`ILight` 不读——Lab2Shot 的数据种类里没有细分面、NURBS 和灯光；
  ② 文件里的单位和上轴由节点上的「单位」「上轴」参数换算成内部标准（厘米、Y 轴向上），原文件一个字节不改。

出处：简介来自 Alembic 官网首页 https://www.alembic.io ；输入输出依据 Alembic 官网首页、
仓库 `lib/Alembic/AbcGeom/` 的 schema，以及 `adapters/alembic/worker.py`。

## 在 Lab2Shot 里怎么用

- **导入 Alembic**：
  - 用「文件」的按钮选 .abc。选好后「相机」「模型」「点云」里列出文件里的每一项：它在层级里的完整路径（和 Maya 大纲、Houdini 里看到的一样，写的是变换节点，如 `/World/shot_rig/cam_main`，不是它下面的 `cam_mainShape`），后面是分得开它们的说明：帧范围、Focal Length、点数、「变形」。
  - 相机选一台；模型、点云选任意几个（勾选，另有「全部」）。某一种文件里只有一个时自动选上。选了哪种才有哪种的输出口。换了文件、原来选的那项没有了，节点标红并写明「文件里没有模型 …：在「模型」里重新选」，不会悄悄换一个。
  - 所有父级的变换（包括动画）逐帧乘进去；相机去掉父级的缩放，只取位置和朝向（相机看到的画面和缩放无关）。
  - **单位**：Alembic 本身不记录单位。Maya 导出的一般是厘米，Houdini 导出的一般是米，选错了会差 100 倍。
  - **上轴**：Alembic 也不记录哪个轴朝上。Maya、Houdini 默认 Y 轴向上；Maya 场景设成 Z 轴向上时导出的选 Z 轴，读进来转成 Y 轴向上。
  - **帧率**：Alembic 里存的是秒，按文件里记录的 DCC 帧率换算成帧号；文件没记帧率时按 24 算。
  - **画面宽度**：填和素材一样的宽度，高度按相机 Filmback 的比例自动算。
  - 输出的相机可以接到「SAM 3D Body 全身动作」的相机输入上，把人匹配到你自己跟好的相机里。
- **Alembic 输出设置**：
  - 输入接相机、模型、点云、三维曲线，或者「合成场景」打在一起的场景。
  - **蒙皮角色**（人体解算、补帧的结果）Alembic 装不下：连上时是红线，参数面板里有「插入「烘焙成模型」」，点它就在中间插上，角色烘焙成逐帧变形的模型（点缓存，和 Maya 导出 Alembic 时一样）。
  - **单位**：给 Maya 用厘米，给 Houdini 用米。
  - 写出的内容：模型（逐帧变形的每帧写顶点，带 UV），每一台动画相机，点云（颜色 Cd、点大小；3D 跟踪点另有每点的编号 id、速度 v 和可见 visible，Houdini 读进来就是这几个点属性，导入时也读回来），三维曲线（线性曲线，颜色 Cd、宽度）。文件格式是 Ogawa。
- **读的时候会自动处理**：有运动模糊子帧采样时，每帧只取离整帧最近的那个；静止的相机、模型也能读。

## 效果和局限

- 只用 CPU，很快：300 帧、18439 个顶点的网格加相机整段写出大约 1 秒；读一个 300 帧的相机不到 1 秒。
- 验证：用 PyAlembic 写一个像 DCC 里做出来的镜头（层级里各处的相机、静止和逐帧变形的模型、点云，都挂在有动画、有缩放的父级下），按厘米 / 米、Y 轴 / Z 轴向上逐一导入，和独立算出的每帧矩阵、Focal Length、Filmback、顶点比对；还有 USD、FBX 和 Alembic 之间互相转、蒙皮角色「烘焙成模型」以后写 Alembic 再读回来；另外用 Alembic 自带的 abcls、abctree、abcecho 检查文件结构、面的朝向和 UV。
- **读相机取 Focal Length、Filmback、胶片偏移（film offset）、挤压比（lens squeeze）、overscan 和每帧位置**：相机上有 Filmback 变换（film back ops）时，节点会提示，但它不会带进 Lab2Shot 的相机。
- 面的顶点顺序按 Alembic 的约定（顺时针）写，和 Maya、Houdini 自己导出的 .abc 一致，导进去法线朝外。

## 团队

Alembic 由索尼图形图像运作（Sony Pictures Imageworks）和卢卡斯影业旗下的工业光魔（Industrial Light & Magic）联合开发，2011 年 8 月 9 日在温哥华的 SIGGRAPH 上发布 1.0 开源版本。发布时 Autodesk、Side Effects（Houdini）、The Foundry 都表态支持，现在它已经是 DCC 之间交换动画缓存的标准格式。用到的数学库 Imath 由学院软件基金会（Academy Software Foundation）维护。

## 模型下载和安装

没有模型要下载。运行 `uv run lab2shot ext install alembic` 会从源码编译 Alembic：

- 下载 micromamba（约 18 MB，只用来建这个扩展包自己的 conda 环境），从 conda-forge 装 Python 3.12、Boost.Python、numpy、cmake、ninja；
- 拉取固定版本的 Alembic 1.8.12 和 Imath 3.2.3 源码，连同 Python 接口（PyAlembic、PyImath）一起编译；
- 顺带编译 Alembic 的命令行工具 abcls、abctree、abcecho、abcechobounds、abcdiff、abcstitcher，放在 `third_party/alembic/.venv/bin/` 里，可以用来检查 .abc 文件。abcconvert 需要 HDF5，没有编译。
- 编译大约 2–3 分钟，环境约 750 MB。
- **需要本机有 C++ 编译器**。要指定用哪个编译器时，在 `config/local.toml` 的 `[build]` 段里设置 `cc` / `cxx`，例如 `gcc-14` / `g++-14`。

**为什么要从源码编译**：官方的 Alembic Python 接口（PyAlembic）没有发布在 PyPI 上。PyPI 上叫「alembic」的包是另一个跟 CG 无关的数据库工具。conda-forge 上的 Imath 也不带 Python 接口。所以只能用固定版本的源码自己编。

## 许可证说明

- **Alembic 是 BSD-3-Clause，可以商用**，可以修改和再分发，要保留版权声明（版权方是 Lucasfilm 和 Sony Pictures Imageworks）。
- 依赖的 Imath 也是 BSD-3-Clause，Boost 是 Boost 许可证，都可以商用。

## 参考

- 官网：http://www.alembic.io/
- 代码仓库：https://github.com/alembic/alembic
- 许可证原文：https://github.com/alembic/alembic/blob/master/LICENSE.txt
- Imath（数学库）：https://github.com/AcademySoftwareFoundation/Imath
- Alembic 1.0 发布新闻稿（2011）：https://sony.mediaroom.com/2011-08-09-Lucasfilm-and-Sony-Pictures-Imageworks-Release-Alembic-1.0
- Houdini 的 Alembic 文档：https://www.sidefx.com/docs/houdini/io/alembic.html
