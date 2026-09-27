+++
team = "Meta（Reality Labs）"
people = "Rawal Khirodkar, He Wen, Julieta Martinez, Yuan Dong, Su Zhaoen, Shunsuke Saito"
paper = "Sapiens2（ICLR 2026）"
paper_url = "https://arxiv.org/abs/2604.21681"
website = "https://rawalkhirodkar.github.io/sapiens2/"
repo = "https://github.com/facebookresearch/sapiens2"
year = 2026
+++

## 这是什么

Sapiens2 是一组高分辨率 transformer，在 10 亿张人像上预训练，在多种以人为中心的任务上取得了当时最好的成绩：姿态估计、身体部位分割、表面法线、点图和人体抠像。

在 Lab2Shot 里接了其中三项，都是逐帧算：

- **身体部位分割**：每个像素标上是头发、脸和脖子、上衣、左小臂、右鞋……共 29 类（含背景），相当于一张按身体部位分好的 ID 遮罩。
- **人体抠像**：0–1 的 alpha，头发丝、包带这类细边也留得住；同一次前向还给出**前景色**（解混后的颜色，预乘 alpha），细边上原来混进来的背景色已经去掉。
- **人体法线**：人身上每个像素的表面朝向，衣褶、肌肉起伏看得很清楚，可以拿来做重打光、加轮廓光或者合成时的辅助 pass。

官方还列了姿态估计和点图两项，没有接。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一张画面张量 `1 × 3 × 1024 × 768`（RGB，建议按 ImageNet 归一化），模型按 `img_size=(1024, 768)`、patch 16 建起来（README）；
  除 Sapiens2-1B (4K) 外都在 1024 × 768（高 × 宽）上训练。
  两个演示脚本的命令行只有 config / checkpoint / `--input` / `--output` / `--save_pred` / `--device`
  （`sapiens/dense/tools/vis/vis_matting.py`、`vis_normal.py`）——**整幅画面一起算，不按框裁切**。
- **给**：抠像那一路一次前向出 4 个通道（alpha 加前景颜色，`vis_matting.py`）；
  分割那一路 `pred_labels = seg_logits.argmax(dim=1)`，`--save_pred` 存成 `_seg.npy`（`vis_seg.py`）；
  法线那一路 `np.save` 存的是**没有遮过的**整幅预测（`vis_normal.py`，它只在给人看的那张对比图上才遮）。

**我们怎么接的**

- 「图像」口 = 上游那张画面张量，整幅送进去，和它的演示脚本一样。分割和抠像按官方 test_pipeline 拉伸到 1024 × 768，
  法线保长宽比补边，结果再缩回画面本来的尺寸（`adapters/sapiens2/worker.py`）。
- 「部位分割」= 官方分割头的类别，「Alpha」「前景」= 官方抠像那一路的 alpha 和它自己的前景颜色（已按 alpha 预乘），
  「法线图」= 官方法线头那一份（OpenCV 相机约定，整幅画面都有值，和 `np.save` 存的一样，不加工）。
- **和官方不一样的一点**：官方 README 还列了姿态估计和点图两项，节点上没有这两样。节点也没有「人物框」输入或法线「遮罩」输出——官方脚本没有这两样。
- 只算画面里的一个人，在送进去之前把别处挡掉（人物框转遮罩 → 图像相乘），不在节点内部裁切。

**出处**：简介来自 `third_party/sapiens2/repo/README.md`（「A family of high-resolution transformers pretrained on 1 billion human images…」整句）；
输入输出依据同一份 README，`repo/sapiens/dense/tools/vis/vis_matting.py`、`vis_seg.py`、`vis_normal.py`
和 `adapters/sapiens2/nodes.py` 的两段 `official`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「人体分割 · Sapiens2」）：读取序列 → **Sapiens2 人体分割** 和 **Sapiens2 人体法线** → 部位分割、Alpha、法线图分别接「序列图输出设置」写 EXR，进 Nuke 合成。
- **整幅画面一起算，节点上没有「人物框」输入。** 上游两个演示脚本收的就是一张图（argparse 只有 config / checkpoint /
  --input / --output / --save_pred / --device）。
  模型的输入是 1024×768：分割和抠像把整幅画面拉伸进去（官方 `keep_ratio=False`），法线保长宽比补边（官方 `NormalResizePadImage`）。
  **只想算画面里的某一个人**：在节点图上接「ViTDet 人物框 →「人物框转遮罩」→「图像相乘」（原图 × 遮罩）→ 这里的「图像」」，
  遮挡发生在送进模型之前，图上一眼看得见。
  在 CRGNN 实拍 + VideoMatte 绿幕共 8 个人像镜头上（人在画面里很大），按人裁切和整幅一起算没区别（J&F 0.992 → 0.987）；人很小的镜头没有验证过。
- 部位分割 EXR 的 R 通道存的是类别编号（半精度浮点），附带类别表。常用编号：0 背景、3 脸和脖子、4 头发、13 下装、22 躯干、23 上装。在 Nuke 里用 Expression 节点写 `r==4` 就得到头发的遮罩。左右是**人物自己的左右**（面对镜头的人，他的左手在画面右边）。
- 法线图在相机空间：+X 画面右、+Y 画面上、+Z 轴朝向镜头（和 Maya / Houdini / Nuke 相机空间法线的习惯一致）。
  交出的是**官方原值**：每个像素都有法线，人以外的地方也有数、只是没有意义（官方 `vis_normal.py` 存下来的 .npy 就是没遮过的，
  它那张遮罩是从 `--seg_dir` 读进来的、只用在给人看的那张对比图上）。要只留人，自己接一张遮罩挡（比如接分割节点的 Alpha）。
- 关键参数：
  - 模型大小：默认 1B，边缘和细节更好；0.4B 快约 2 倍、显存约 1.5 GB，适合快速视图。
  - 精细抠像（只在分割节点上）：默认开，用单独的 1B 抠像模型出 alpha 和前景，头发边缘最好，但慢一倍；关掉则由分割结果推 alpha，有空洞、边缘硬，也就没有「前景」输出口。
  - 半精度：默认开（bf16），更快更省显存，和全精度相差不到 0.1%。
- 「前景」输出（打开精细抠像时才有）：预乘 alpha 的 RGBA，写成场景线性 EXR。在 Nuke 里拿它当前景合成，头发丝和半透明边缘上不会再带着旧背景的颜色；只用原图乘 alpha 做不到这一点。

## 效果和局限

RTX 4090 上（1B 模型、半精度，按人物框裁切后送入；整幅送入时模型输入同为 1024×768）：iPhone 长焦背影行走（1080×1920，30 帧，一个人）

- 人体分割 + 精细抠像：约 0.23 秒/帧，显存峰值约 6.5 GB；加载模型约 8.5 秒。
- 人体法线：约 0.26 秒/帧，显存峰值约 6.5 GB。

局限：

- 每帧单独算，没有前后帧约束，边缘和部位分界会有轻微闪动；要稳定的视频抠像，可以把这里的 alpha 当作 MatAnyone 2 的引导遮罩。
- 只认人。人拿着的道具、宠物、车不在结果里（抠任意物体用 BiRefNet 或 SAM 3）。
- 画面里几个人挨在一起时，部位分割不分人（它只说这个像素是哪个部位），alpha 也是所有人合在一起的一张。
- 法线是模型"看"出来的，不是几何计算，适合做合成辅助 pass，不适合当精确几何用。
- 官方还有 0.8B 和 5B 模型，以及姿态、点云等任务；本扩展只装了 0.4B 和 1B 的分割、法线和 1B 抠像（5B 每个任务约 20 GB，24 GB 显卡上太吃紧）。

## 团队

Meta 的人体视觉团队。第一版 Sapiens 由 Meta Reality Labs 发布，是 ECCV 2024 最佳论文候选；Sapiens2 发表在 ICLR 2026，预训练数据从 3 亿张人物图像扩大到 10 亿张，模型从 0.4B 到 5B。最后作者 Shunsuke Saito 在 Meta Codec Avatars 实验室（匹兹堡）负责下一代数字人，之前的代表作有 PIFu（ICCV 2019）和 PIFuHD（CVPR 2020）——从单张照片重建穿衣服的三维人体。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install sapiens2`。会下载：
  - 锁定版本的官方代码（约 200 MB）；
  - 独立 Python 环境（PyTorch 2.13 + CUDA 13，约 5 GB）；
  - 五个权重（Hugging Face，锁定版本，不需要申请）：1B 分割（约 5.9 GB）、1B 法线（约 6.2 GB）、1B 抠像（约 6.2 GB）、0.4B 分割（约 1.6 GB）、0.4B 法线（约 1.8 GB）。
  - **模型合计约 21.6 GB**，是这批扩展里最大的，要预留好硬盘和下载时间。
- 下载完会逐个核对文件大小和校验码（SHA-256）。网络中断导致文件不完整时会自动删掉坏文件，并提示你再运行一次安装，只重下坏掉的那几个。
- 不需要额外手动下载；装好后离线运行。

## 许可证说明

**可以商用，但有明确的禁止用途。** 代码和全部权重都用 Meta 自定义的 Sapiens2 License：免费、全球范围内可以使用、复制、修改、分发，没有"非商用"限制。

禁止用于（许可证原文第 1.b.vi 条）：

- 监控，包括和监控有关的研发；
- 生物特征处理；
- 识别或重新识别个人身份；
- **制作深度伪造（deepfake）或其他误导、欺骗性内容，包括冒充任何个人**——也就是说，不能拿它做未经授权的真人换脸、冒充真人的数字替身；
- 未经法律要求的授权和同意，收集或推断个人的健康、人口统计等敏感信息；
- 色情或诽谤内容；
- 侵犯第三方权利；无证从事金融、法律、医疗等专业工作；
- 军事、武器、核、间谍、毒品、关键基础设施操作等可能造成人身伤害的用途，以及美国出口管制禁止的用途。

其他条款：不能逆向工程；使用要遵守出口管制和隐私法规（如 GDPR）；对外分发 Sapiens 材料或其衍生作品时必须附上这份许可证；用它做研究发表论文时要注明使用了 Sapiens；Meta 认为你可能违规时可以审计并要求删除；Meta 可以修改条款，修改后继续使用即视为同意。

## 参考

- 论文：https://arxiv.org/abs/2604.21681
- 项目主页：https://rawalkhirodkar.github.io/sapiens2/
- 代码：https://github.com/facebookresearch/sapiens2
- 许可证原文：https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md
- 模型合集：https://huggingface.co/collections/facebook/sapiens2
- 模型卡：https://huggingface.co/facebook/sapiens2-seg-1b 、https://huggingface.co/facebook/sapiens2-normal-1b 、https://huggingface.co/facebook/sapiens2-matting-1b
- 在线演示（分割）：https://huggingface.co/spaces/facebook/sapiens2-seg
- 第一版 Sapiens（ECCV 2024）：https://github.com/facebookresearch/sapiens
- Shunsuke Saito 主页：https://shunsukesaito.github.io/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
