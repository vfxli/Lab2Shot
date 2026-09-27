+++
team = "浙江大学 CAD&CG 国家重点实验室 三维视觉组（ZJU3DV）"
people = "Zehong Shen, Huaijin Pi, Yan Xia, …, Ruizhen Hu, Xiaowei Zhou"
paper = "World-Grounded Human Motion Recovery via Gravity-View Coordinates（SIGGRAPH Asia 2024）"
paper_url = "https://arxiv.org/abs/2409.06662"
website = "https://zju3dv.github.io/gvhmr/"
repo = "https://github.com/zju3dv/GVHMR"
year = 2024
+++

## 这是什么

上游自己的话（论文摘要）：GVHMR 从单目视频里恢复**世界空间**的人体动作。难点在于世界坐标系怎么定，每段镜头都不一样；以往的方法用自回归的方式预测相对动作，误差会越积越多。GVHMR 改为在新的**重力—视角（Gravity-View）坐标系**里估计人体姿态，这个坐标系由世界重力和相机的观看方向共同定义，天然与重力对齐、每一帧唯一，大幅降低了从画面学姿态的歧义；估出的姿态再用相机旋转转回世界坐标系，拼成一整段全局动作。论文 SIGGRAPH Asia 2024。

在 Lab2Shot 里，它是「GVHMR 全身动作」：交出的 SMPL-X 骨骼参数和逐帧网格落进「蒙皮角色」类型，单位厘米、Y 轴向上：人往哪走、走了多远、脚踩在地上不打滑都保留下来。节点上**没有「相机」输出口**：官方只给两套 SMPL 参数和内参，不算相机。

## 输入输出

**官方要什么、给什么**

- 吃：**一段视频**。官方脚本（`tools/demo/demo.py`）收 `--video`、`-s/--static_cam`（固定机位就跳过 SLAM）、`--use_dpvo`、
  `--f_mm`（全画幅等效 Focal Length，毫米，留空用默认值）。
  **人是它自己检的**：用它自带的 `Tracker().get_one_track(video_path)` 出框。
  真正喂进网络的是这几样：`bbx_xys`（它自己检的框）、`kp2d`（它自己跑的 ViTPose）、
  `K_fullimg`（按 `--f_mm` 或估出来的内参）、`cam_angvel`、`f_imgseq`。
  其中 `cam_angvel = compute_cam_angvel(R_w2c)`——**只有每帧旋转** `R_w2c`：
  固定机位时是单位阵，否则由它自带的 SimpleVO 或 DPVO 解出，**位移不参与**。
- 给：**同一副身体两份**——`smpl_params_incam`（相机空间的 SMPL 参数）和
  `smpl_params_global`（世界空间、重力对齐的那一份），两份都存进 `paths.hmr4d_results`。
- **它自己解相机的转动**：默认 `SimpleVO`，`--use_dpvo` 换成 DPVO；喂进网络时只取旋转。
  **相机的位移上游自己不解、也不吃。**

**我们怎么接的**

- 「图像」口就是 `--video` 那段画面；「已知 Focal Length」「Filmback」对应 `--f_mm`；「固定机位」就是 `--static_cam`。
- 「蒙皮角色」口是 `smpl_params_global`，按 CG 的形态交出来（骨架 + 逐帧动画 + 蒙在骨架上的网格，
  DCC 里能二次修正）；「2D 关键点」口是 `kp2d`。没有单独的点缓存「网格」口：它和「蒙皮角色」逐点平均只差 0.2 毫米，是同一个结果的烘焙版。
- **「相机旋转」是一个参数，不是一个相机输入口**：上游只吃 `cam_angvel`，所以线接在参数上——从「拆分相机」的「旋转」口拉过来。
  Focal Length 同理，走「Focal Length」参数。不接整台相机：接整台相机而只用它的转动，等于把位移扔掉了，而图上看不出来；
  一个节点只接它真正会用到的那一样。
- **「参照相机」输出口 = 上游那两份 SMPL 参数之间的关系**：同一副身体既在相机空间又在世界里，
  两者之间的刚性关系**就是**这一段结果在它自己那个重力世界里配着的那台相机
  （`adapters/gvhmr/worker.py` 的 `camera_from_body`，旋转取自上游自己的 VO）。
  **它不是成品相机**，唯一的用处是接进核心节点「相机空间转换」当参照。
- **不一样的一点**：**「把人放到你那台相机的世界里」是 Lab2Shot 加的一步，做成了一个看得见的节点**——
  核心节点「相机空间转换」（`core.camera_space`）收两台相机：这个节点的「参照相机」和你自己那台，
  算出一个常量修正挂到人身上（`worker_sdk/lab2shot_shared/poses.py` 的 `rigid_align`，核心和 worker 共用同一份）。
  这是 Lab2Shot 的用法，不是上游的要求；上游根本没有「输入一台相机」这件事。
- **「人物框」是可选输入口**：官方包的 `VitPoseExtractor.extract(video_path, bbx_xys)`、
  `Extractor.extract_video_features(video_path, bbx_xys)` 和 `DemoPL.predict(data)` 的 `data["bbx_xys"]` 收的就是逐帧的框，
  官方 demo 只是先用自带的 YOLOv8 出框再喂给它们。接了就按框解、不再跑跟踪器；不接就照官方 demo 自己跟踪。

出处：简介来自论文摘要（arXiv 2409.06662）和仓库 `README.md`；输入输出依据
`third_party/gvhmr/repo/tools/demo/demo.py` 和 `adapters/gvhmr/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → ViPE 解相机 →「拆分相机」（把 Focal Length 和每帧旋转显式接给解算器）→ GVHMR 解算 →「相机空间转换」（「参照相机」接它的「来源相机」，ViPE 相机接「目标相机」）→ USD 输出设置 → 输出，导进 Houdini 或 Maya 做动画参考、替身或碰撞体。
- **「人物框」可选**：不接时画面里的人由 GVHMR 自己找（官方脚本用的就是它自带的 YOLOv8 检测器），连续出现 16 帧以上的人都会解，编号按它自己认出来的人排。
- 只想解画面里的某一个人：接「ViTDet 人物框」→「选人」→ 这个节点的「人物框」口，就只解选中的那几个（编号跟着框走）。另一条路是在上游把别人从画面里去掉再接进「图像」口——「ViTDet 人物框」→「选人」→「人物框转遮罩」→「图像合成」（框变成黑白图，和原图相乘，画面上只剩他）。
- **要让解出来的人在你那台相机下和实拍对上，接「相机空间转换」**（模板里已经接好了）：
  GVHMR 交出来的人在**它自己的重力世界**里（原点在这个人起点的髋部），所以
  「GVHMR 全身动作」的「参照相机」→「相机空间转换」的「来源相机」，你那台相机 →「相机空间转换」的「目标相机」，
  节点默认「逐帧贴合」：每帧按 目标相机 × 来源相机⁻¹ 把人搬过去，画面上严格对上；「整段平滑」则把两条相机轨迹
  拟合成一个带尺度的常量变换，动作更连贯但每帧不严格贴。**「来源相机」不要留空**——留空表示结果在相机前面、
  没有世界位置，那会给这种自带世界的结果每帧再乘一次相机，相当于把相机的运动加了两遍，人会飞到镜头里面去、不在画面里。
- **相机的转动和 Focal Length 走参数，不走相机口**：从「拆分相机」的「旋转」「Focal Length」「Filmback」三个口拉线到这个节点的同名参数上
  （GVHMR 只吃这几样）。**「参照相机」不是成品相机，别交付**：图上要一台能用的相机，
  从它自己的来源接（ViPE 或「导入 USD」），一眼看得出交付里那台是谁的。
- 「逐帧贴合画面」：逐帧透过相机放人，和画面严格贴合（代价是深度方向可能有抖动）。
- 固定机位的镜头请打开「固定机位」：不做相机估计，并用固定机位的后处理，脚更稳。
- 知道 Focal Length 必须填（「Focal Length (mm)」配合「Filmback」）：长焦镜头如果不填，默认按约 53° 对角视场猜，人会被放得太近、动作比例失真。
- 「2D 关键点」输出：GVHMR 解算前 ViTPose 在画面上找的全身 17 个点（带模型自己的把握值），画面上的点（不是三维结果的投影），接「2D 跟踪点输出设置」交给 3DEqualizer / Nuke，也可以叠在画面上看解出来的人贴不贴。按人分组，一个人一组。
- 输出「蒙皮角色」是骨骼带动的网格（能在 DCC 里改动画，和模型原始结果差 1 厘米左右）。模板「全身动作 · GVHMR」：ViPE 解相机 →（拆分相机给 Focal Length 和旋转）→ GVHMR →「相机空间转换」（自带世界）→ USD 输出；对位在视图里透过相机看。
- 适合：人物全身或大半身入画、持续出现至少 16 帧的镜头。不适合：人只露出上半身特写、严重遮挡、多人紧贴。

## 效果和局限

- RTX 4090 上（模型加载后，人物框由上游节点给、不含节点自己找人那一遍）：手机长焦跟拍 1080×1920、120 帧、Focal Length 5484 像素，单人约 30 秒、两人 26 秒，显存峰值 5.5 GB；固定机位舞蹈 864×480、124 帧约 21 秒。不接「人物框」时多一遍 YOLOv8 逐帧找人，时间和显存会更高，没有单独的数字。
- 效果：解算出的人物网格通过解算用的那台相机投回画面和人物贴合；长焦跟拍里人走了约 4.6 米，脚底贴在 y=0 附近（起伏 7 厘米内），身高 1.59 米；接 ViPE 相机时刚性对齐误差约 0.24 米，打开「逐帧贴合画面」则逐帧贴合。
- GVHMR 内部只估相机的转动，不估平移；接进来的相机也只用到转动。**节点不输出相机**：官方没算相机。
- 手指和表情不估计（SMPL-X 手是放松的默认手型）。
- 多人是逐个解算再按相机对齐到同一个世界，人与人之间的相对位置精度取决于每个人的深度估计。

## 团队

浙江大学 CAD&CG 国家重点实验室周晓巍老师的三维视觉组（ZJU3DV），和深圳大学胡瑞珍老师合作。这个组在人体和场景三维重建上很有名，做过 EasyMocap、NeuralBody、LoFTR 特征匹配等工作。

## 模型下载和安装

- 自动安装：`lab2shot ext install gvhmr`，下载原始仓库、独立 Python 环境（torch 2.3 + CUDA 12.1）和三个权重：GVHMR 主网络、HMR2.0a 图像特征、ViTPose-H 关键点（共约 5.5 GB）。
- 需要手动下载：SMPL-X 人体模型。到 https://smpl-x.is.tue.mpg.de 注册登录，在 Download 页面下载「SMPL-X v1.1 (NPZ+PKL, 830 MB)」（models_smplx_v1_1.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，「帮助与扩展包」页面的「手动下载」写着这个文件夹在哪），Lab2Shot 会自动解压并找到里面的 `SMPLX_NEUTRAL.npz`。

## 许可证说明

- 不能商用。GVHMR 代码和权重是浙江大学的许可：只允许教育、科研和非营利用途；基于它的修改必须开源，并且同样禁止商用。商用要联系 xwzhou@zju.edu.cn。
- SMPL-X 人体模型：仅限非商用科研，禁止再分发。
- 其他组件：HMR2.0a（4D-Humans，MIT）、ViTPose-H（Apache-2.0）、pytorch3d / pycolmap（BSD）、smplx 代码（马普所非商用许可）。

## 参考

- 论文：https://arxiv.org/abs/2409.06662
- 项目主页（有演示视频）：https://zju3dv.github.io/gvhmr/
- 代码：https://github.com/zju3dv/GVHMR
- SMPL-X：https://smpl-x.is.tue.mpg.de
