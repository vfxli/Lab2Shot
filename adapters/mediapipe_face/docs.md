+++
team = "Google（MediaPipe 团队）"
people = "Yury Kartynnik, Artsiom Ablavatski, Ivan Grishchenko, Matthias Grundmann"
paper = "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs（CVPR 2019 Workshop）"
paper_url = "https://arxiv.org/abs/1907.06724"
website = "https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker"
repo = "https://github.com/google-ai-edge/mediapipe"
year = 2019
+++

## 这是什么

Google 官方的说法：MediaPipe Face Landmarker 这个任务能在画面和视频里检测面部关键点和面部表情。它输出三维面部关键点、blendshape 分数（表示面部表情的系数）以便实时推断细致的面部表面，以及做特效渲染所需的变换矩阵；官方给的用途是识别面部表情、做面部滤镜、虚拟形象和特效渲染。它属于端上机器学习，只用 CPU，不占显卡。

在 Lab2Shot 里，它逐帧交出三样物体：**52 条表情曲线（blendshape）**，名字和苹果 ARKit 的面部表情表一样（jawOpen 张嘴、eyeBlinkLeft 左眼闭、mouthSmileRight 右嘴角笑……），每条是 0–1 的数值，可以当权重去驱动角色的 blendShape；**478 个面部关键点**（含左右眼虹膜各 5 个点），可以写成 3DEqualizer 的 2D 点或 CSV；**头部运动**，一张跟着头动的面具网格（468 个点，带 UV）和它每帧的变换（相对 MediaPipe 官方那台虚拟相机，垂直视场角 63°，节点原样交出）。

## 输入输出

**官方要什么、给什么**

- 吃：一张画面。官方文档写的输入是三选一：静止画面、解码后的视频帧、实时视频流；
  API 这一层是 `FaceLandmarker.detect(image)`（`third_party/mediapipe_face/repo/mediapipe/tasks/python/vision/face_landmarker.py`）。
  配置项有最多几张脸、检测阈值、跟住门槛、存在阈值，以及要不要出表情和变换矩阵。
- 给：`FaceLandmarkerResult` 三样——
  `face_landmarks`（每张脸的三维面部关键点，归一化画面坐标）、
  `face_blendshapes`（可选的表情分数）、
  `facial_transformation_matrixes`（可选的面部变换矩阵）。

**我们怎么接的**

- 「RGB」口就是 `image`，「最多几张脸」「检测阈值」「跟住门槛」「存在阈值」就是官方的那几个配置项。
- 「面部关键点」= `face_landmarks`，「表情曲线」= `face_blendshapes`（52 条，ARKit 的命名），
  「头部」= 官方自带的那张标准脸网格按 `facial_transformation_matrixes` 逐帧摆位——网格和矩阵都是官方的。
- **上游没有、节点也不给**：「相机」输出口。官方的矩阵是相对它自己假设的一台虚拟相机（63° 垂直视场）说的，
  `FaceLandmarkerResult` 里根本没有相机这一项。要把头放进某台相机的世界，接核心节点「相机空间转换」。

出处：简介来自 MediaPipe 官方文档 Face Landmarker 页的第一句
（https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker ：
「The MediaPipe Face Landmarker task lets you detect face landmarks and facial expressions in images and videos」）；
输入输出依据同一页的任务输入 / 输出两节，以及
`third_party/mediapipe_face/repo/mediapipe/tasks/python/vision/face_landmarker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「面部动作 · MediaPipe」）：读取序列 → **MediaPipe 面部动作** →「表情曲线」接「曲线输出设置」（CSV / .chan / JSON / USD）；「头部」接 合成场景 → USD 输出设置。「面部关键点」可以接「2D 跟踪点输出设置」写 3DEqualizer 点文件。
- **节点上没有「相机」的进出口**：官方的结果里根本没有相机这一项，头部变换是相对它假设的 63° 虚拟镜头说的。头的远近是那台虚拟镜头下的相对值。要把头摆进某台相机（ViPE 解算或读取的 3DE 相机）的世界，接核心节点「相机空间转换」（`camera_space`）——这一步在节点图上看得见。
- 什么素材好：**正脸、脸够大**（特写、近景）、光线正常。不行：侧脸超过约 80°、脸被挡住一半以上、远景小脸（官方检测模型针对 2 米以内的自拍距离）、背影（整段找不到脸时节点会提示，三个口都是空的）。
- 关键参数：
  - 最多几张脸：主角特写保持 1 最稳；大于 1 时 MediaPipe 会关掉帧间平滑。
  - 方式：默认「视频」，利用上一帧的位置并做平滑，抖动少约 40%；「逐张」适合不相关的照片。
  - 检测 / 跟住 / 存在阈值：脸找不到或经常跟丢就调低；把别的数据当成脸就调高。

### 用 52 条表情曲线驱动 Maya / Houdini 里的角色

1. **角色要有 ARKit 命名的表情目标。** 角色头部需要一套按 ARKit 52 个表情名做好的 blendShape 目标（browInnerUp、jawOpen、eyeBlinkLeft……名字和大小写要一致）。很多面部绑定和角色生成工具可以导出这一套，也可以按苹果文档里每个表情的说明自己雕。
2. **左右是演员自己的左右**（和 ARKit 一样）：eyeBlinkLeft 是演员的左眼，正对镜头时在画面右边。
3. **导出曲线**：「曲线输出设置」选 CSV，第一列是帧号（原始帧号），后面每列一条曲线，列名就是表情名。52 条里有一条叫 `_neutral`（中性脸），不用接，其余 51 条和 ARKit 同名；ARKit 里的 tongueOut（吐舌）MediaPipe 不预测，角色上这个目标不会被驱动。
4. **Maya**：在 Script Editor（Python）里执行下面这段，把 CSV 逐帧打成 blendShape 权重的关键帧（只处理角色上有同名目标的曲线）：

   ```python
   import csv
   from maya import cmds

   BS = "blendShape1"                                    # 角色头上的 blendShape 节点名
   targets = set(cmds.aliasAttr(BS, query=True)[::2])    # 这个节点上所有目标的名字
   with open("D:/shots/shot010/shot010_face.csv", newline="") as fh:
       for row in csv.DictReader(fh):
           frame = float(row.pop("frame"))
           for name, value in row.items():
               if name in targets:
                   cmds.setKeyframe(BS, attribute=name, time=frame, value=float(value))
   ```

5. **Houdini**：「曲线输出设置」选 .chan，会同时写出 `<文件名>_names.txt`（按顺序列出每条曲线的名字）。用 File CHOP 读 .chan（每行一帧），按 names 文件的顺序给通道改名（Rename CHOP），再把通道 Export 到角色 Blend Shapes SOP 对应的权重参数上，或在权重参数里用 `chop()` 表达式引用。也可以像 Maya 一样用 Python 读 CSV 打关键帧。选 USD 格式时，每条曲线是 `/shot/curves` 上一个带动画的属性（`lab2shot:curve:<表情名>`），适合用脚本接到 Solaris / 管线里。
6. **必须先"修"曲线**：这些数值是模型的估计，不是按你的角色标定的。比如一段面部特写上眨眼最大只到约 0.75–0.8，左右眼还不一样；有些曲线在无表情时也不是 0。常见做法是每条曲线乘一个增益、减掉无表情时的底数、截到 0–1，再轻微平滑（Maya Graph Editor 的 Butterworth 滤波、Houdini 的 Filter CHOP）。
7. **头部动画**：USD 里 `/shot/face_01` 带每帧变换，下面挂着面具网格。可以把角色头骨约束到这个变换上，得到头部转动；位移的远近是 MediaPipe 虚拟镜头（63°）下的相对值，要按真实镜头摆位就先接「相机空间转换」。

## 效果和局限

只用 CPU：一段面部特写（772×855，113 帧）整段约 2.9 秒，约 0.025 秒/帧（网络本身约 6 毫秒，其余时间在读图），113 帧全部找到脸。背影行走（看不到脸）的镜头会提示"整段都没找到脸"，三个口都是空的。

局限：

- 官方定位是手机 AR 娱乐（自拍视频），结果适合做表情动画的初版，精度和细微表情（嘴唇、眼皮）比不上专业面捕头盔 + 标记点。
- 曲线有抖动，光线差、有运动模糊、脸被挡时更明显（模型卡也这么写）；「视频」方式平滑的是关键点，表情曲线导出后最好再滤一遍。
- 没找到脸的帧：关键点记为"被挡住"，表情曲线和头部动画沿用前后帧的值。
- 头部的远近按"平均成年面部的大小"推算，不同演员的头部距离会有误差；关键点的深度图（z）只是相对值。
- 不做身份识别：模型卡明确说明它不提供面部识别，不存储能识别个人的面部特征。

## 团队

Google 的 MediaPipe 团队（代码仓库现在在 Google AI Edge 名下）。MediaPipe 是 Google 开源的端侧实时感知框架（论文 2019）。同一批研究者（Matthias Grundmann 等）还发表了 BlazeFace（手机 GPU 上亚毫秒级面部检测）、MediaPipe Hands（实时手部跟踪）和 BlazePose（实时人体姿态）。本扩展用到的三个模型分别由 Valentin Bazarevsky（BlazeFace 面部检测）、Geng Yan 和 Ivan Grishchenko（Face Mesh V2 关键点、Blendshape V2 表情）等人开发。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install mediapipe_face`。会下载：
  - 锁定版本（v1.0.0）的官方代码（约 130 MB，用来对照许可证、标准脸网格和坐标约定）；
  - 独立 Python 环境（官方 pip 包 mediapipe 1.0.0 + OpenCV，不需要 PyTorch，约 470 MB）；
  - 模型包 `face_landmarker.task`（Google 官方地址，float16 v1 版，**只有 3.6 MB**，下载后核对 SHA-256），里面是面部检测、478 点关键点和 52 条表情三个模型。
  - 文件都很小，很快就能装完。
- 不需要申请权限，不需要额外手动下载，也不需要显卡；装好后离线运行。

## 许可证说明

**可以商用。** MediaPipe 代码（仓库和 pip 包）是 Apache-2.0；`face_landmarker.task` 里的三个模型——BlazeFace 面部检测（近距离版）、Face Mesh V2（478 点）、Blendshape V2（52 条表情）——按各自模型卡也都是 Apache-2.0；标准脸网格随仓库以 Apache-2.0 发布。Apache-2.0 要求再分发时保留许可证和版权声明、注明改动。

模型卡写明的用途边界：主要用于 AR 娱乐；不用于攸关人身安全的决策；不提供面部识别或身份辨认，任何形式的监控和身份识别都不在适用范围内。本扩展不依赖 FLAME / SMPL 这类需要注册的面部、人体模型。

## 参考

- 官方文档（Face Landmarker）：https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- 代码：https://github.com/google-ai-edge/mediapipe ，许可证 https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE
- 标准脸网格（带 UV 的 OBJ）：https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
- 模型卡：Blendshape V2 https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf ；Face Mesh V2 https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf ；BlazeFace（近距离版）https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20(Short%20Range).pdf
- 苹果 ARKit 52 个表情的定义（每个名字对应什么动作）：https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
- 论文：Face Mesh https://arxiv.org/abs/1907.06724 ；BlazeFace https://arxiv.org/abs/1907.05047 ；MediaPipe 框架 https://arxiv.org/abs/1906.08172

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
