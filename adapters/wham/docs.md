+++
team = "卡内基梅隆大学（CMU）+ 马普智能系统研究所（MPI-IS）"
people = "Soyong Shin, Juyong Kim, Eni Halilaj, Michael J. Black"
paper = "WHAM: Reconstructing World-grounded Humans with Accurate 3D Motion（CVPR 2024）"
paper_url = "https://arxiv.org/abs/2312.07531"
website = "https://wham.is.tue.mpg.de/"
repo = "https://github.com/yohanshin/WHAM"
year = 2024
+++

## 这是什么

WHAM（World-grounded Humans with Accurate Motion）从视频里准确而高效地解出全局坐标系下的三维人体动作。它用动捕数据学会把二维关键点序列抬到三维，再与画面特征融合，把动作的上下文和画面所见结合起来；并用 SLAM 估出的相机角速度配合人的动作，估计身体的全局轨迹。在此之上还有一套考虑接触的轨迹细化，使它在上楼梯这类各式条件下都捕得到人的动作。

在 Lab2Shot 里，这个节点叫「WHAM 全身动作」，交出世界空间里的蒙皮角色动画、逐帧网格和二维关键点。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`demo.py --video`，一段视频。`--calib` 给一个内参文件 `[fx fy cx cy]`，
  不给时按 CLIFF 的约定拿画面对角线当 Focal Length；`--estimate_local_only` 跳过 SLAM 只出相机空间的动作；
  `--run_smplify` 打开 Temporal SMPLify 精修。
  人的框和二维关键点由它自己的 `DetectionModel`（YOLOv8x + ViTPose-H）检出并跟住。
- **相机那一路它只用旋转**：DPVO 交出的是 7 个数（位置 3 + 四元数 4），而 `CustomDataset` 只取四元数那四个
  （`lib/data/datasets/dataset_custom.py` 里的 `quat = traj[:, 3:]`），换算成相邻帧之间的相机角速度 `cam_angvel` 喂进网络。
  **位置那三个数上游没有用。**
- **给**：每个人的 SMPL 动作，**同一个人同时给两份**——相机空间那一份
  （`results[_id]['pose'] / ['trans']`）和世界那一份
  （`['pose_world'] / ['trans_world']`，重力对齐、Y 轴向上、米），外加 betas 和顶点。
- **它自己解相机**：`demo.py` 里 `slam = SLAMModel(video, output_pth, width, height, calib)`、
  `slam_results = slam.process()`，结果存成 `slam_results.pth`。
  那台相机（DPVO）**只有旋转被喂进网络**（上一条），位移上游自己不看。

**我们怎么接的**

- 「图像」= 上游那段素材；「Focal Length」「Filmback」= 上游的 `--calib`；「固定机位」对应上游不执行 DPVO 那条路（相机转动为零）；
  「SMPLify 细化」= 上游的 `--run_smplify`。上游跟不到 30 帧的人会丢掉，这里照做。
- 「2D 关键点」= 上游 `DetectionModel` 的关键点；「蒙皮角色」= 上游世界那一份 SMPL 结果，
  按 CG 的形态交出来（骨架 + 逐帧动画 + 蒙在骨架上的网格，DCC 里能二次修正）。
- **「相机旋转」是一个参数，不是一个相机输入口**：上游只吃 `cam_angvel`，
  所以线接在参数上——从「拆分相机」的「旋转」口拉过来，worker 把它摆成 DPVO 那个 7 数布局
  （`adapters/wham/worker.py` slam_from_rotations），上游照原样只取旋转。Focal Length 同理，走「Focal Length」参数。
  节点不会接收一台完整的相机再暗中只取其中一部分。
- **「参照相机」输出口 = 上游那两份 SMPL 结果之间的关系**：同一个人既在相机空间又在世界里，
  两者之间的刚性关系**就是**这一段结果在它自己那个世界里配着的那台相机
  （`worker_sdk/lab2shot_worker/world_humans.py` `camera_from_body`，旋转取自上游自己的 DPVO）。
  这样上游交出的相机空间那一份也保留了下来。
  **它不是成品相机**，唯一的用处是接进核心节点「相机空间转换」当参照。
- **不一样的两点**：
  ① **「把人放到你那台相机的世界里」是额外的一步，做成了一个看得见的节点**：
     核心节点「相机空间转换」`core.camera_space`（接到「来源相机」）收两台相机——
     这个节点的「参照相机」和你自己那台——算出一个修正挂到人身上
     （`worker_sdk/lab2shot_shared/poses.py` `rigid_align`，核心和 worker 共用同一份）。
     上游本身没有「输入一台相机」这件事。
  ② 上游每个人各自一个世界，这里把跟得最久那个人的世界当作场景的世界，其余的人整体并进来（`world_humans.merge_worlds`）。

**出处**：简介来自论文摘要里介绍方法的三句（arXiv 2312.07531：「WHAM learns to lift 2D keypoint sequences to 3D using motion capture data and fuses this with video features…」
「WHAM exploits camera angular velocity estimated from a SLAM method together with human motion to estimate the body's global trajectory.」
「We combine this with a contact-aware trajectory refinement method…」）；
输入输出依据 `third_party/wham/repo/demo.py`、`repo/README.md`、
`repo/lib/data/datasets/dataset_custom.py` 和 `adapters/wham/worker.py`、`nodes.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → ViPE 解相机 →「拆分相机」（把 Focal Length 和每帧旋转显式接给解算器）→ WHAM 解算 →「相机空间转换」（「参照相机」接它的「来源相机」，ViPE 相机接「目标相机」）→ USD 输出设置 → 输出，导进 Houdini、Maya 做动作参考或替身。
- **不用接人物框**，节点上没有这个输入口：画面里的人由 WHAM 自己找（官方 demo.py 用的就是它自带的 YOLOv8 检测加 ViTPose 关键点），连续出现 30 帧以上的人都会解，编号按它自己认出来的人排。
- 只想解画面里的某一个人：在上游把别人从画面里去掉，再接进「图像」口——「ViTDet 人物框」→「选人」→「人物框转遮罩」→「图像合成」（框变成黑白图，和原图相乘，画面上只剩他）。这条链在节点图上看得见，比藏在解算器里的一个口清楚。
- **要让解出来的人在你那台相机下和实拍对上，接「相机空间转换」**（模板里已经接好了）：
  WHAM 交出来的人在**它自己的世界**里（原点在这个人起点的髋部），所以
  「WHAM 全身动作」的「参照相机」→「相机空间转换」的「来源相机」，你那台相机 →「相机空间转换」的「目标相机」，
  节点默认「逐帧贴合」：每帧按 目标相机 × 来源相机⁻¹ 把人搬过去，画面上严格对上；「整段平滑」则把两条相机轨迹
  拟合成一个带尺度的常量变换，动作更连贯但每帧不严格贴。**「来源相机」不要留空**——留空表示结果在相机前面、
  没有世界位置，那会给这种自带世界的结果每帧再乘一次相机，相当于把相机的运动加了两遍，人会跑出画面、位置完全不对。
- **相机的转动和 Focal Length 走参数，不走相机口**：从「拆分相机」的「旋转」「Focal Length」「Filmback」三个口拉线到这个节点的同名参数上
  （WHAM 只吃这几样）；Focal Length 取整段中值。**「参照相机」不是成品相机，别交付**：图上要一台能用的相机，
  从它自己的来源接（ViPE 或「导入 USD」）。
- 「逐帧贴合画面」：逐帧透过相机放人，和画面严格贴合（离相机的远近会抖，脚可能滑）。
- 「固定机位」：不跑 DPVO，按相机不动处理。
- 「2D 精修（SMPLify）」：用 2D 关键点再优化一遍，画面贴合更好，但更慢。
- 「2D 关键点」输出：WHAM 解算前 ViTPose 在画面上找的全身 17 个点（带模型自己的把握值），画面上的点（不是三维结果的投影），接「2D 跟踪点输出设置」交给 3DEqualizer / Nuke，也可以叠在画面上看解出来的人贴不贴。按人分组，一个人一组。
- 知道 Focal Length 必须填（「Focal Length (mm)」配合「Filmback」），长焦镜头尤其重要；或者接上 ViPE / 3DE 相机。

## 效果和局限

- RTX 4090 上（模型加载后，不含 YOLOv8 逐帧检测的时间）：手机长焦跟拍 1080×1920、120 帧（含 DPVO 相机估计）约 45 秒，显存峰值 2.9 GB；固定机位 124 帧约 31 秒。人物检测另外按帧计时。
- 效果：主要人物投回画面贴合；同一镜头里的第二个人按各自的世界动作对齐后，长焦镜头上会偏离画面最多约 1.4 米（结果里有 residual 数字），固定机位的旋转舞蹈上主要人物也会偏开几十厘米。要严格贴合画面请打开「逐帧贴合画面」（代价是脚可能滑）。
- 世界原点在人物起点的髋部，地面不在 y=0，需要「自动落地」。
- WHAM（DPVO）只估相机的转动；接进来的相机也只用到转动。**节点不输出成品相机**：上游只算了转动，从人的位置反推出的相机不是上游的结果。
- 只有身体（SMPL 24 个关节），没有手指和表情。
- 人要连续出现 30 帧以上；遮挡严重或只露半身时效果下降。

## 团队

卡内基梅隆大学的 Soyong Shin 在马普智能系统所实习期间完成，合作者包括人体模型 SMPL 的发明人 Michael J. Black。马普所感知系统部是 SMPL、SMPL-X、SMPLify、VIBE、BEDLAM 等人体重建基础工作的出处。

## 模型下载和安装

- 自动安装：`lab2shot ext install wham`，下载原始仓库（含 DPVO、ViTPose 子模块）、独立 Python 环境（torch 2.9 + CUDA 13，现场编译 DPVO，需要几分钟）和权重：WHAM、HMR2.0a、DPVO、ViTPose-H，以及 SMPL 关节回归矩阵等辅助文件（共约 6 GB）。
- 需要手动下载：SMPL 人体模型。到 https://smpl.is.tue.mpg.de 注册登录，在 Download 页面下载「SMPL for Python users」的 1.1.0 版（SMPL_python_v.1.1.0.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，「帮助与扩展包」页面的「手动下载」写着这个文件夹在哪）。用到的是里面的 `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`；已经有改名为 `SMPL_NEUTRAL.pkl` 的文件或 SMPLify 的 `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` 也可以。

## 许可证说明

- 不能商用。WHAM 代码是 MIT，但必须配合 SMPL 人体模型使用，SMPL 仅限非商用科研、禁止再分发；随 WHAM 下载的关节回归矩阵由 SMPL 派生，同样按 SMPL 许可。
- WHAM 权重作者没有单独写许可，训练数据（AMASS、BEDLAM、3DPW 等）都只许研究用，按研究用途对待。
- 其他组件：DPVO（MIT）、HMR2.0a（MIT）、ViTPose-H 和 mmcv / mmpose（Apache-2.0）、smplx 代码（马普所非商用许可）。

## 参考

- 论文：https://arxiv.org/abs/2312.07531
- 项目主页：https://wham.is.tue.mpg.de/
- 代码：https://github.com/yohanshin/WHAM
- DPVO：https://github.com/princeton-vl/DPVO
- SMPL：https://smpl.is.tue.mpg.de
