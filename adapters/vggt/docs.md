+++
team = "牛津大学视觉几何组（VGG）+ Meta AI"
people = "Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny"
paper = "VGGT: Visual Geometry Grounded Transformer（CVPR 2025 最佳论文）"
paper_url = "https://arxiv.org/abs/2503.11651"
website = "https://vgg-t.github.io/"
repo = "https://github.com/facebookresearch/vggt"
year = 2025
+++

## 这是什么

Visual Geometry Grounded Transformer（VGGT，CVPR 2025）是一个前馈神经网络，从一张、几张乃至几百张画面里，几秒钟内推断出场景的全部关键三维属性：相机外参和内参、点图、深度图，以及 3D 跟踪点。

在 Lab2Shot 里，这个节点叫「VGGT 深度与相机」：整段画面切成有重叠的段送进网络、段之间对齐拼接，交出每帧相机、深度图、点云和置信度。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`load_and_preprocess_images([...])` 出来的画面张量，送进 `model(images)`。
  `forward` 的入参只有 `images` 和 `query_points` 两个（`vggt/models/vggt.py`）。
- **给**（README 里列的那几个头，对应 `vggt.py`）：`camera_head` → 按 OpenCV 约定的外参和内参；
  `depth_head` → 深度图和它的置信度；`point_head` → 世界点图和它的置信度；
  `track_head` → 给定查询点的轨迹、可见性和把握程度。
  README 还注明：由深度图加相机反投影出的点通常比点图那一路更准。
- README 说：画面里不想要的像素（反光、天空、水面）把像素值设成 0 或 1 就能遮掉，简单的框遮罩就够用——
  **那是把像素涂掉，不是模型的一个输入。**

**我们怎么接的**

- 「RGB」= 上游那个画面张量；整段按「每段最多帧数」切成有重叠的段，段之间用相似变换拼接。
- 「深度图」= `depth_head` 的深度图，「置信度」= 它的置信度，「相机」= `camera_head` 的 `pose_enc`，
  「点云」= `point_head` 的世界点图 `world_points`（worker 把它放回每一帧的相机空间再交出来，无损，只是换坐标系）。
- **对不上的一点**：上游的 `track_head`（点轨迹、可见性、把握程度）**没有做成输出口**——它要 `query_points`，
  这个节点没有那个输入（`adapters/vggt/nodes.py` 的 `official` 末句）。要点轨迹，用 TAPNext++ 或 TAPIP3D。
- 节点上**没有遮罩 / 人物框输入口**：上游 `forward` 里根本没有这一路。
  只算画面的一部分，照 README 的办法在送进去之前把别处涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→「RGB」口。

**出处**：简介来自 `third_party/vggt/repo/README.md`（「Visual Geometry Grounded Transformer (VGGT, CVPR 2025) is a feed-forward neural network that directly infers all key 3D attributes of a scene…」）；
输入输出依据同一份 README、`repo/vggt/models/vggt.py` 和 `adapters/vggt/nodes.py` 的 `official`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 → VGGT 深度与相机 → 和 ViPE 相机解算 / COLMAP 的结果对比，互相校验。「点云」口交的是官方点头（`world_points`）算出来的每像素三维点，不是从深度图反投影的。
- 用途定位是"交叉检验"：一次前馈、没有平差，Focal Length 和轨迹是网络估的，不如 ViPE 精确，但几乎不会"解崩"，拿来发现另一套解算的大错很合适。
- 适合：有一定视差的镜头、静止机位，长焦也能用（Focal Length 会偏小一点，见下）。不适合：变焦、镜头畸变大的素材。
- 尺度不是米：VGGT 的单位是它自己归一化的要手动缩放，整段镜头内一致；需要真实尺度时用 ViPE 或 Pi3X。
- 关键参数：`max_frames` 一次送进网络的帧数（可选 32 / 64 / 130，默认 130，控制在 20 GB 以内），更长的镜头会自动分段、用重叠帧对齐拼接；`step` 隔帧取样，镜头很长时用 2–4 可以省时间。
- **手填每段最多帧数的上限**：`max_frames` 一次性看完整段（不是流式模型），填的数字越大显存越高。24 GB 显卡上的安全上限统一按竖画面的更紧数字封顶为 **130 帧**（横画面约 230 帧还在 20 GB 内，但填参数时还不知道画幅，只能按更保守的数字统一封顶）；服务器按同样的上限拒绝更大的数字。
- 模型本身不接受遮罩，节点上也没有遮罩 / 人物框输入口：画面里的人照样参与计算，结果里也不会把它们抹掉。想只算画面的一部分，在送进去之前把其余部分涂黑：「ViTDet 人物框」→「人物框转遮罩」→「图像合成」（留下）→ 这个节点的「RGB」口，图上一眼看得见。

## 效果和局限

RTX 4090 上，和 ViPE 相机解算对比（相机轨迹先做相似变换对齐再算误差）：

| 素材 | 结果 |
|---|---|
| iPhone 长焦跟拍，1080×1920，150 帧（真实 Focal Length 约 5484 px，ViPE 5133 px） | Focal Length 5061 px（偏小 8%）；轨迹和 ViPE 的差 0.18 m / 4.8 m 行程（3.7%），朝向差最大 1.0°；分两段计算，共 45 秒，显存峰值 14.4 GB |
| 固定机位跳舞，864×480，124 帧（真实 Focal Length 约 560–590 px） | Focal Length 664 px（偏大 13–18%，ViPE 在这个镜头上是 1053 px，更离谱）；相机基本不动：位移只有场景深度的 0.3%，转动最大 0.1°；18 秒，显存 12.5 GB |

- 一次最多送进网络的帧数有限，更长的镜头自动分段：相邻两段共用 1/4 的帧，用这些帧的相机朝向和点云把后一段对齐到前一段（接缝残差约为深度的 1–1.5%），接缝处相机、Focal Length、深度图做渐变。分得越碎误差越大：同一镜头故意切成 5 段时，轨迹误差从 3.7% 增大到 5%，朝向差最大 1.6°。
- 「回环闭合」（默认关，和 VGGT、Pi3、Depth Anything 3 共用一套）：分段拼接时一段接一段误差会积累；打开后找出隔得远却拍到同一处的两段，把这两处的画面放在一起再算一次，用它把各段拉回一致（和 VGGT-Long 的做法相同，但不需要另外的检索模型：按缩略图像不像、按目前的拼接结果互相看不看得见来找；放在一起算对不上的、优化后仍和别的对不上的会被丢掉）。临时文件放在缓存里、算完就删（792 帧约 1.3–2.2 GB）。效果（没有在真正绕一圈回到原处的实拍素材上验证过）：固定机位跳舞镜头切成 4 段：找到 3 处都用上，但相机几乎没变（最大移动 5.7 → 5.3 cm，转动 0.08° → 0.09°）；iPhone 长焦整段 792 帧往前走的跟拍（8 段）：5 处候选用上 2 处，和 ViPE 的位置差 4.7% → 3.8%，朝向差中位 9.1° → 9.9°，多花 8% 时间。固定机位的漂移主要在段里面，回环管不到；没有回到原处的镜头上作用有限，所以默认关。
- 尺度是 VGGT 自己的任意单位，不是米。
- Focal Length 是逐帧估的，同一镜头里会有 ±3% 左右的抖动（真实镜头没变焦）；要准 Focal Length 请用 ViPE / COLMAP 或实拍参数。
- 竖画面会被补白边成正方形再算（VGGT 训练时没见过竖画面），显存比横画面多用约 75%。
- 对画面里大面积没纹理或很远的区域（地面、远处楼面）VGGT 的置信度很低，这些像素不会进输出点云（遮罩）。
- 模型不接受遮罩输入，节点上也没有这个口。

## 团队

牛津大学视觉几何组（Visual Geometry Group，VGG）和 Meta AI 联合完成，一作 Jianyuan Wang。VGG 是三维视觉领域的老牌实验室（VGGNet、经典的《计算机视觉中的多视几何》作者之一 Andrew Zisserman 所在的组）；同一批作者还做过 CoTracker（长距离点跟踪）、PoseDiffusion、VGGSfM。VGGT 拿了 CVPR 2025 最佳论文，2026 年又发布了后续版本 VGGT-Ω。

## 模型下载和安装

- 运行 `lab2shot ext install vggt`：下载 VGGT 代码（锁定版本）、建独立 Python 环境（PyTorch 2.10，约 7 GB，和其他扩展共用下载缓存），再下载 VGGT-1B 原版权重（5.0 GB）。网速正常时十来分钟。
- 可商用的 VGGT-1B-Commercial 权重（5.0 GB）需要先申请：登录 Hugging Face，打开 https://huggingface.co/facebook/VGGT-1B-Commercial ，填写页面上的申请表并同意许可证，审批通过后再运行一次 `lab2shot ext install vggt` 即可（没批下来之前安装会提示"需要申请权限"，原版权重照常可用）。

## 许可证说明

- 代码：Meta 的 VGGT License（2025-07-29 版），允许商用、修改和再分发（要附带许可证原文）；发论文要注明用了 VGGT；禁止军事、战争、核工业、间谍、武器、关键基础设施操作、欺诈冒充等用途；起诉 Meta 侵权则许可终止。
- 权重：VGGT-1B 原版是 CC BY-NC 4.0，**非商用**，只能用于研究（本扩展默认用它）；VGGT-1B-Commercial 按 VGGT License 发布，**可以商用**（同样禁止军事等用途），需要先申请。官方说两份权重效果接近。

## 参考

- 论文：https://arxiv.org/abs/2503.11651
- 项目主页：https://vgg-t.github.io/
- 代码：https://github.com/facebookresearch/vggt
- 许可证原文：https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt
- 原版权重模型卡：https://huggingface.co/facebook/VGGT-1B
- 可商用权重模型卡：https://huggingface.co/facebook/VGGT-1B-Commercial
- 在线演示：https://huggingface.co/spaces/facebook/vggt
- 后续版本 VGGT-Ω：https://vggt-omega.github.io/
