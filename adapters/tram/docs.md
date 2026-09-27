+++
team = "宾夕法尼亚大学 GRASP 实验室（University of Pennsylvania）"
people = "Yufu Wang, Ziyun Wang, Lingjie Liu, Kostas Daniilidis"
paper = "TRAM: Global Trajectory and Motion of 3D Humans from in-the-wild Videos（ECCV 2024）"
paper_url = "https://arxiv.org/abs/2403.17346"
website = "https://yufu-wang.github.io/tram4d/"
repo = "https://github.com/yufu-wang/tram"
year = 2024
+++

## 这是什么

TRAM 是一种两阶段方法，从实拍视频解出人的全局轨迹和动作。它让 SLAM 在画面里有走动的人时依然稳定，求出相机运动，并用场景背景推出动作的尺度；以求得的这台相机作为真实尺度的参考系，再用一个视频 transformer 模型（VIMO）回归人的身体动作。两段运动合起来，得到世界空间里的三维人物，全局动作误差比以往方法大幅降低。

在 Lab2Shot 里，这个节点叫「TRAM 全身动作」。上游按顺序执行的三个脚本（先解相机、再估每个人、最后合到世界空间）在节点内部一次算完，交出相机、蒙皮角色和逐帧网格。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`scripts/estimate_camera.py --video`，一段视频。三个脚本**依次**执行：
  `estimate_camera.py`（ViTDet + SAM + DEVA 检出并跟住人、存成 `boxes.npy` / `tracks.npy` → 按 SLAM 重投影误差搜索 Focal Length
  → 遮掉人的 DROID-SLAM + ZoeDepth 定真实尺度 → SPEC 估重力）、`estimate_humans.py`（VIMO 逐个人解相机空间的 SMPL，
  框从 `tracks.npy` 读回来）、`visualize_tram.py`（相机 × 相机空间的身体 = 世界里的身体）。
- **上游不吃相机，但官方函数收人物框**：相机那一路 VIMO 只收 `img_focal` 和 `img_center`（内参），没有任何一帧外参
  （`lib/models/hmr_vimo.py` 的 `inference(...)` 签名），相机是 `estimate_camera.py` 自己解出来的**产物**；
  框那一路：官方 demo 里框是 `estimate_camera.py` 内部 ViTDet + SAM + DEVA 检出、存进 `tracks.npy` 的，
  但官方包的 `HMR_VIMO.inference(imgfiles, boxes, …)` 收的就是一个人逐帧的框，官方自己的 EMDB 评测脚本
  （`scripts/emdb/run_smpl.py`）就是拿外部真值框调它的。
- **给**：一台重力对齐、真实尺度的相机（`estimate_camera.py` 的 `world_cam_R` / `world_cam_T` / `img_focal`），
  每个人的 SMPL 参数（`hmr_vimo.py` 的 `pred_rotmat` / `pred_shape` / `pred_trans`），
  和它自己拿这些参数算出的顶点（`lib/vis/traj.py` 的 `pred_vert`）。

**我们怎么接的**

- 「图像」口 = 上游那段素材（worker 把 PNG 按 `*.jpg` 链过去，不重编码）。
- 「相机」「蒙皮角色」「网格」= 上游那三步的结果，一样不少：「蒙皮角色」是官方 SMPL 参数装成的，「网格」是官方 `pred_vert`。
- 「Focal Length」「Filmback」参数 = 上游的 `img_focal`：不填就走它自己的 Focal Length 搜索（500–1500 px，长焦镜头搜不到）；
  「固定机位」= 上游的 `--static_camera`；「最多人数」= 上游的 `--max_humans`。
- **节点上没有「相机」输入口**——上游没有；要按自己解好的相机摆人，走显式的小工具节点「相机空间转换」。
  **「人物框」是可选输入口**（官方函数本来就收框）：接了就按框解这几个人；不接就照官方 demo 自己检人跟踪。
- **不一样的一点**：每个人整段用一个体型（中位数），上游做可视化时也是这么平均的；VIMO 要至少 16 帧连续，更短的片段丢掉（上游一样）。

**出处**：简介来自论文摘要前三句（arXiv 2403.17346：「We propose TRAM, a two-stage method to reconstruct a human's global trajectory and motion from in-the-wild videos…」）；
输入输出依据 `third_party/tram/repo/README.md`、`repo/scripts/estimate_camera.py`、`repo/scripts/estimate_humans.py`、
`repo/lib/models/hmr_vimo.py`、`repo/lib/vis/traj.py` 和 `adapters/tram/nodes.py` 的 `official`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → TRAM 解算 → USD 输出设置 → 输出。
- **「人物框」可选**：接「ViTDet 人物框」→「选人」→ 这个口，VIMO 就按框解这几个人（上游的检测 + 分割照旧跑一遍，SLAM 要的人物遮罩来自它）；「最多人数」按框的顺序（最显眼的在前）取前几个。不接：人由 TRAM 自己的 ViTDet + SAM + DEVA 认出来，「最多人数」按出现时间长短决定解几个。
- **没有「相机」输入口**：上游的 VIMO 只吃 Focal Length 和主点，相机是 TRAM 自己用遮人的 DROID-SLAM 解出来的，是这条流程的产物，不是它的输入。要把人按你自己的相机摆进世界，接显式的小工具节点「相机空间转换」，节点图上看得见。
- 知道 Focal Length 必须填（「Focal Length (mm)」配合「Filmback」）。不填时 TRAM 会在 500–1500 像素之间按 SLAM 误差搜索 Focal Length，长焦镜头会猜错。
- 「固定机位」：不跑 SLAM，按相机不动处理（TRAM 自己也会检测）。
- 人要连续出现 16 帧以上；更短的片段会被丢掉。

## 效果和局限

- RTX 4090 上（模型加载后）：手机长焦跟拍 1080×1920、120 帧（检测、遮罩、DROID-SLAM、ZoeDepth、VIMO 全流程）约 2 分钟，显存峰值 8.5 GB，内存约 9 GB；固定机位 124 帧约 70 秒。
- 效果：所有人都通过相机放进世界，投回画面严格贴合（包括第二个人）；但在长焦镜头上重力方向和尺度不稳，上面那段长焦跟拍里脚底高度前后起伏约 0.9 米（SPEC 按第一帧估的视场角 2003 像素，远小于真实的 5484 像素）。长焦镜头要在「Focal Length」里填实拍 Focal Length。
- 相机解算需要画面里有足够的背景纹理和视差；纯摇镜头、长焦压缩的镜头上尺度可能不准。
- 重力方向只用第一帧估计（SPEC），第一帧看不到地平线或透视线索时可能歪。
- 只有身体（SMPL 24 个关节），没有手指和表情。

## 团队

宾夕法尼亚大学 GRASP 实验室（Kostas Daniilidis 组）和刘玲杰老师合作完成。GRASP 是机器人和计算机视觉的老牌实验室；作者后来把 TRAM 并入了 PromptHMR。

## 模型下载和安装

- 自动安装：`lab2shot ext install tram`，下载原始仓库（含 DROID-SLAM、DEVA 子模块）、独立 Python 环境（torch 2.9 + CUDA 13，现场编译 DROID-SLAM，需要几分钟）和权重：VIMO、ViTDet-H、Segment Anything ViT-H、DEVA、DROID-SLAM、ZoeDepth、SPEC（共约 10 GB）。
- 需要手动下载：SMPL 人体模型。到 https://smpl.is.tue.mpg.de 注册登录，在 Download 页面下载「SMPL for Python users」的 1.1.0 版（SMPL_python_v.1.1.0.zip），压缩包原样放进 Lab2Shot 的 `downloads/` 文件夹（不用解压、不用改名，「帮助与扩展包」页面的「手动下载」写着这个文件夹在哪）。用到的是里面的 `basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl`；已有的 `SMPL_NEUTRAL.pkl` 或 SMPLify 的 `basicModel_neutral_lbs_10_207_0_v1.0.0.pkl` 也可以。

## 许可证说明

- 不能商用。TRAM 代码是 MIT，但：仓库里的 SPEC 相机标定代码和权重是马普所的非商用许可；DEVA 视频跟踪（代码和权重）是 CC BY-NC-SA 4.0 非商用；必须配合 SMPL 人体模型使用，SMPL 仅限非商用科研、禁止再分发。
- VIMO 权重作者没有单独写许可，训练数据（BEDLAM、3DPW、Human3.6M 等）都只许研究用。
- 其他组件：DROID-SLAM 和 lietorch（BSD-3）、ViTDet-H（detectron2）和 Segment Anything（Apache-2.0）、ZoeDepth / MiDaS（MIT）、smplx 代码（马普所非商用许可）。

## 参考

- 论文：https://arxiv.org/abs/2403.17346
- 项目主页：https://yufu-wang.github.io/tram4d/
- 代码：https://github.com/yufu-wang/tram
- DROID-SLAM：https://github.com/princeton-vl/DROID-SLAM
- SMPL：https://smpl.is.tue.mpg.de
