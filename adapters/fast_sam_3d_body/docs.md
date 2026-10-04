+++
team = "南加州大学 物理超级智能实验室（USC Physical Superintelligence Lab）"
people = "Timing Yang, Sicheng He, Hongyi Jing, Jiawei Yang, Zhijian Liu, Chuhang Zou, Yue Wang"
paper = "Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery（ECCV 2026）"
paper_url = "https://arxiv.org/abs/2603.15603"
website = "https://yangtiming.github.io/Fast-SAM-3D-Body-Page/"
repo = "https://github.com/yangtiming/Fast-SAM-3D-Body"
year = 2026
+++

## 这是什么

上游自己的话（论文摘要）：SAM 3D Body 在单目三维人体网格解算上达到了当时最好的精度，但每张画面几秒的推理延迟使它无法用于实时场合。Fast SAM 3D Body 是一套**免训练**的加速框架，把 SAM 3D Body 的推理路径重新组织：解开串行的空间依赖、按网络结构剪枝，从而并行提取多张裁切图的特征、精简 transformer 解码；另外把迭代式的 MHR 转 SMPL 拟合换成一次前馈映射，这一步加速一万倍以上。整体端到端最快 10.9 倍，网格精度与 SAM 3D Body 持平，在 LSPET 这类基准上还更好。

在 Lab2Shot 里，它是「SAM 3D Body 全身动作」的提速版本：用的是同一套 Meta 权重，进出和那个节点一样——序列图进，出人物（MHR 骨骼动画加蒙皮网格，含手指），单位厘米，结果在相机空间里，可以接 USD 输出设置 → 输出。节点上**没有「相机」的进出口**（和「SAM 3D Body 全身动作」一样：上游吃的是一个 Focal Length 数值，不是一台相机）。

## 输入输出

**官方要什么、给什么**

- 吃：一个画面文件夹。官方 `demo.py` 收 `--image_folder`（必填）、`--checkpoint_path`、`--detector_name`（默认 `vitdet`）。
  **人是它自己检的**：`demo.py` 的 `main()` 自己建一个 `HumanDetector`，调用处只传画面路径；
  它自带的另一个 demo（`demo_human.py`）也是 `--detector` 默认 yolo。
- 给：每个人一份估计结果（`sam_3d_body/sam_3d_body_estimator.py`）——
  `body_pose_params`（MHR / SMPL 的姿态参数）和 `pred_vertices`（网格顶点），以及相机那一路的中间量。

**我们怎么接的**

- 「RGB」口就是 `--image_folder`：和上游一样，**只有画面**。
- 「蒙皮角色」口是官方整套 MHR 参数（`body_pose_params` 等）装成的蒙皮角色：骨骼动画加静止网格和蒙皮权重；没有单独的「网格」口。
- **「人物框」是可选输入口**：这个仓库自己的 `process_one_image(img, bboxes=None, …)`
  （`sam_3d_body/sam_3d_body_estimator.py`）收框。接了就按框解；不接时 worker 用它自己的 YOLO11-Pose 检人
  （一趟同时给出框和手腕）。
- 节点上的「已知 Focal Length」「Filmback」对应上游用的那一个 Focal Length 数值（上游自己吃的就是一个 Focal Length，不是一台相机），
  所以这里没有「相机」输入口。

出处：简介来自论文摘要和 `third_party/fast_sam_3d_body/repo/README.md`；
输入输出依据 `third_party/fast_sam_3d_body/repo/demo.py`、`demo_human.py`、`sam_3d_body/sam_3d_body_estimator.py`
和 `adapters/fast_sam_3d_body/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → Fast SAM 3D Body 全身动作 → USD 输出设置，和 SAM 3D Body 全身动作的接法一模一样，可以替换节点。要世界空间的动作，后面接核心节点「相机空间转换」，把人摆进 ViPE 解出的（或 3DE 导入的）那台相机的世界。
- 和 SAM 3D Body 全身动作一样，**「人物框」可选**：接「ViTDet 人物框」→「选人」→ 这个口就只解这些人；不接时画面里有几个人由它自己检（YOLO11-Pose，和它找手腕是同一趟，所以不多花时间），检出几个就解几个。接了框而且开着「手部精修」时，YOLO-Pose 仍跑一遍只为拿手腕。
- 适合：镜头多、帧数多、需要快速出一版动作的时候；多人镜头一帧里的人一起算。最终精修的镜头如果手指动作很重要，建议和 SAM 3D Body 全身动作对比一次再用。
- 参数和 SAM 3D Body 全身动作相同：Focal Length / Filmback、手部精修、锁定体型、平滑强度。关掉「手部精修」只算身体，最快（864×480 单人素材上约 0.08 秒一帧）。

## 效果和局限

RTX 4090 上（显卡和其他任务共用，数字有 ±15% 的波动），同一段画面、同样的参数（手部精修开、锁定体型开、平滑 0.5）：

| 素材 | SAM 3D Body 全身动作 | Fast SAM 3D Body | 每帧提速 |
|---|---|---|---|
| 跳舞，864×480，124 帧，1 人 | 0.49 秒/帧，整段 90 秒 | 0.17 秒/帧，整段 48 秒 | 约 3.0 倍 |
| 手机长焦跟拍，1080×1920，60 帧，1 人 | 0.55 秒/帧，整段 52 秒 | 0.20 秒/帧，整段 39 秒 | 约 2.7 倍 |
| 同一跟拍的另 60 帧，两个人交错走过 | 0.72 秒/帧，整段 68 秒 | 0.30 秒/帧，整段 47 秒 | 约 2.4 倍 |

- 整段时间里还有加载模型（约 10 秒）、估计 Focal Length（约 6 秒）这些固定开销，所以镜头越长，提速越接近每帧的倍数。显存两者差不多（峰值约 4.0–4.2 GB），内存峰值约 6 GB。
- 和 SAM 3D Body 全身动作的结果比：身体骨骼平均差 0.5–2.4 厘米（大部分是整个人前后差 1–2 厘米的前后差），只看姿态（以骨盆为准）平均差 0.2–0.5 厘米；差别主要在手指，平均 1.7–3 厘米，个别帧指尖差到 10 厘米左右（手部裁图的位置来源不同）。体型（身高）一致。
- 论文里的「快 10 倍」是和原版演示程序比的（原版每张图都要跑一次 ViTDet 找人和 MoGe 估 Focal Length，并且用了 TensorRT 和 RTX 5090）。Lab2Shot 的 SAM 3D Body 全身动作本来就只估一次 Focal Length，找人也放在单独节点里，所以在这里实际能拿到的是每帧 2.4–3 倍。
- **TensorRT（默认开）**：按上游官方路径，把 DINOv3 骨干网络换成 TensorRT FP16 引擎。每种显卡第一次用时自动生成一份、存在扩展自己的 cache/tensorrt 文件夹里（RTX 4090 约 75 秒，RTX 5090 约 2.5 分钟，之后直接用）；生成或加载不了就自动改用 PyTorch，并提示原因。8 帧 3 人实测：估计姿态每帧 RTX 4090 0.57→0.49 秒、RTX 5090 0.24→0.19 秒；和 PyTorch 结果比，身体骨骼平均差不到 1 毫米，手指个别关节最多差约 2 厘米（比和 SAM 3D Body 全身动作之间的差还小）。上游另外两个引擎没用：YOLO11-Pose 引擎在 RTX 5090 上检测一点不快、生成要 6 分钟，还让手部裁图位置变了（手指差到 4 厘米）；MoGe 引擎是给小号 MoGe 模型做的，换了会改变 Focal Length 结果。启动服务前设环境变量 `LAB2SHOT_FAST_SAM_3D_BODY_TRT=0` 可以关掉 TensorRT。
- 默认没有打开 torch.compile：在 RTX 4090 上 torch.compile 每帧只省 0.02 秒，但每次计算要多花 60–90 秒编译，几千帧以上的镜头才划算。需要时，启动 Lab2Shot 服务前设环境变量 `LAB2SHOT_FAST_SAM_3D_BODY_COMPILE=1` 即可打开 torch.compile（编译结果缓存在扩展自己的 cache 文件夹里）。
- 手腕被挡住、YOLO 看不清手腕的帧，会自动改用原版的方式算手（稍慢），日志里会提示有几帧。上面两人交错的片段里有 7 帧是这样。

## 团队

南加州大学（USC）Yue Wang 老师的物理超级智能实验室（PSI Lab），和 UC San Diego / NVIDIA 的 Zhijian Liu、Meta Reality Labs 的 Chuhang Zou 合作。实验室主要做人形机器人和三维视觉，做过 Psi-0 人形机器人基础模型、Denoising Vision Transformers 等工作；这篇论文的初衷是让机器人能用一个普通摄像头实时模仿人的动作。

## 模型下载和安装

- 自动安装：`lab2shot ext install fast_sam_3d_body`，下载提速版代码仓库、独立 Python 环境（Python 3.11，torch 2.8 + CUDA 12.8，约 7 GB）、DINOv3 骨干网络代码、YOLO11-Pose 权重（42 MB）和 NVIDIA TensorRT 运行库（3.1 GB，可断点续传）。依赖包已在缓存里时不到 1 分钟。
- SAM 3D Body 主权重（2.1 GB）和 MoGe-2 Focal Length 模型（1.3 GB）和「SAM 3D Body」扩展包是同一份：已经装了 SAM 3D Body 就直接链接过来，不再下载；没装的话会自动下载。
- 需要申请权限：SAM 3D Body 权重在 Hugging Face 上需要申请（https://huggingface.co/facebook/sam-3d-body-dinov3 页面上填表申请），批准后再运行一次安装。已经装好 SAM 3D Body 扩展包的话说明已经批准过了。

## 许可证说明

- 可以商用，但找手腕用的 YOLO11-Pose 是 AGPL（见最后一条）。
- 提速部分的代码是 MIT，但它是在 Meta 的 SAM 3D Body 代码上改的，原有部分和权重仍按 SAM License：可以商用、修改、再分发（要附上许可证），发表论文要注明使用了 SAM 3D Body，禁止军事等出口管制用途。
- 骨干网络代码 DINOv3 License，条款同类（可商用，需附许可证、论文注明，禁止军事用途）。MoGe-2 Focal Length 模型 MIT。
- 找手腕用的 Ultralytics YOLO11-Pose（代码和权重）是 AGPL-3.0：自己内部用没问题；如果修改后拿去对外提供服务或分发，需要按 AGPL 开源；闭源商用要向 Ultralytics 购买企业许可。

## 参考

- 论文：https://arxiv.org/abs/2603.15603
- 项目主页：https://yangtiming.github.io/Fast-SAM-3D-Body-Page/
- 代码：https://github.com/yangtiming/Fast-SAM-3D-Body
- 原版 SAM 3D Body：https://github.com/facebookresearch/sam-3d-body
- SAM 3D Body 权重（需申请）：https://huggingface.co/facebook/sam-3d-body-dinov3
- DINOv3：https://github.com/facebookresearch/dinov3
- YOLO11-Pose：https://docs.ultralytics.com/tasks/pose/ ，许可说明 https://www.ultralytics.com/license
- 实验室主页：https://psi-lab.ai
