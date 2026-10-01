+++
team = "泰国 VISTEC 视觉与学习实验室（Vision & Learning Lab, VISTEC）+ Google Research、Stability AI、Pixiv 的合作者"
people = "Worameth Chinchuthakun, Pakkapon Phongthawee, Amit Raj, Varun Jampani, Pramook Khungurn, Supasorn Suwajanakorn"
paper = "DiffusionLight-Turbo: Accelerated Light Probes for Free via Single-Pass Chrome Ball Inpainting（2025）"
paper_url = "https://arxiv.org/abs/2507.01305"
website = "https://diffusionlight.github.io/turbo/"
repo = "https://github.com/DiffusionLight/DiffusionLight-Turbo"
year = 2025
+++

## 这是什么

上游 README 的摘要这样介绍它：把「从一张低动态范围（LDR）画面估计光照」这件事，改写成在画面里
补画一个镜面铬球的问题，借用预训练的扩散模型 Stable Diffusion XL，绕开已有方法受限于 HDR 全景
数据集、泛化不佳的毛病。补画对扩散的初始噪声很敏感，所以作者先用多次补画取中位数，得到一个稳定的
低频光照先验来引导最终结果；要出高动态范围（HDR）的光探针，则微调一个 Exposure LoRA 生成多档
曝光的 LDR 画面再合成。这一版是 DiffusionLight-Turbo：用一个 Turbo LoRA 预测那个平均铬球，把一次
估计从约 30 分钟压到约 30 秒，提速 60 倍，质量损失很小。

在 Lab2Shot 里，它是「DiffusionLight HDRI」：没拍 HDRI、没拍铬球的镜头，也能补出一份环境光。
铬球由模型画在画面正中，再展开成经纬图（lat-long）的 EXR，可以当穹顶灯（dome light）给 CG 物体
打光，用来做灯光参考、快速出初版灯光。装的是加速版 DiffusionLight-Turbo。

## 输入输出

**官方要什么、给什么**

- 吃：**一张**方形画面。官方要求先把画面缩放成 1024×1024，不是方的就用黑边补。三步命令：
  ① `python inpaint.py --dataset <输入目录> --output_dir <输出目录>`，
  ② `python ball2envmap.py --ball_dir <输出目录>/square --envmap_dir <输出目录>/envmap`，
  ③ `python exposure2hdr.py --input_dir <输出目录>/envmap --output_dir <输出目录>/hdr`。
  **不吃相机、不吃遮罩。**
- 给：第一步出 `raw`（画面中间补画出铬球的那张）和 `square`（方形裁切的铬球）；
  第二步把铬球展开成 LDR 经纬图；第三步按多档曝光合成 HDR，写进 `<输出目录>/hdr`。

**我们怎么接的**

- 「RGB」口就是 `--dataset` 那一张画面（节点内部按上游的要求补成方形）。
- 「HDRI」口就是第三步的 `hdr`，「铬球」口就是第一步的 `square`：两样都是上游自己的产物。
- 节点上的「已知 Focal Length」「Filmback」不进上游的模型（上游没有相机这一路），但进展开铬球那一步：
  由它们算出水平视场角，worker 按透视相机（相机在原点看球）展开；不填就按上游 `ball2envmap.py` 默认的正交展开
  （相机在无穷远，常见焦距下误差不大）。节点不把镜头数值传给下游。

出处：简介来自论文摘要和 `third_party/diffusionlight/repo/README.md`；输入输出依据同一份 README、`exposure2hdr.py`
和 `adapters/diffusionlight/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「HDRI · DiffusionLight」）：读取图片 → **DiffusionLight** →「HDRI」接序列图输出设置写 EXR，在 DCC 里挂成穹顶灯（Houdini Solaris、Maya）。「铬球」输出是画出来的球，用来一眼判断结果靠不靠谱。节点没有相机口、也不出灯光口。
- 挂灯时要注意方向：**经纬图正中间那一列就是镜头看的方向**，顶上是画面的上方；在 DCC 里需要自己旋转灯的 Y 轴对齐。
- 什么素材好：环境看得比较全的一帧（广一点的镜头、人挡得少）。只算一帧，所以选帧很重要。
- **只吃一帧**：接进来的是一段序列会被拒绝（`E-LIGHT-ONEFRAME`），前面接「FrameHold」选一帧——挑环境最完整、前景遮挡最少的那一帧。
- 关键参数：
  - 随机种子：**结果对它很敏感**。铬球里的反射看着不像这个环境、或主光方向和画面里的影子对不上时，换个数字重算，多试几个挑最像的。
  - 已知 Focal Length / Filmback：算出水平视场角，用来按透视正确展开铬球；不填按正交展开，常见焦距下误差不大。
  - 环境图宽度：默认 1024（高 512）。铬球本身只有 256 像素，调到 2048 不会更清楚，只是文件更大。

## 效果和局限

RTX 4090 上：iPhone 长焦素材（1080×1920，取一帧）算一个环境光约 51 秒（三档曝光各画一次铬球约 49 秒），显存峰值约 12.4 GB。论文里说 Turbo 版比原版 DiffusionLight 快约 60 倍（原版一张约 30 分钟）。

局限（用之前必须知道）：

- **这是"画"出来的，不是测量的。** 画面里看不到的方向（尤其是镜头背后）是模型根据画面猜的，大方向（天空在上、主光大致方位、冷暖）通常合理，细节不可信。它适合做灯光参考和快速初版，不能替代现场 HDRI。
- **亮度是相对的**：线性值 1.0 大约等于画面本身的白；三档曝光（0、-2.5、-5 EV）合成后最亮只能到画面白的约 32 倍。太阳这类极亮的点光源会被明显压低（户外长焦素材上最亮处约 12，中间值约 0.67），实际使用时一般要在 DCC 里调强度，或补一盏平行光当太阳。
- 色彩空间：EXR 是线性 Rec.709（sRGB 原色），但原色只是名义上的，没有经过色彩校准。
- 铬球放在画面正中间，得到的是"画面中心那个位置"看到的环境；画面会先缩放放进 1024×1024 的方形画布（多出来的地方补黑），竖幅素材浪费的画布更多。
- 铬球边缘对应镜头正前方的方向，那部分最不可靠；镜头背后的方向反而由球的中心看到，最清楚。
- 只用一帧，不随时间变化；换镜头、换场景要各算一次。

## 团队

主要由泰国 VISTEC（Vidyasirimedhi Institute of Science and Technology）的视觉与学习实验室完成，实验室负责人是 Supasorn Suwajanakorn；合作者来自 Google Research（Amit Raj）、Stability AI（Varun Jampani）和 Pixiv（Pramook Khungurn）。第一版 DiffusionLight 是 CVPR 2024 的口头报告论文；这个实验室之前还做过 NeX（CVPR 2021 口头报告、最佳论文候选，实时新视角合成）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install diffusionlight`。会下载：
  - 锁定版本的官方代码（约 190 MB，里面自带 Turbo LoRA 和曝光 LoRA 两个小模型，约 90 MB）；
  - 独立 Python 环境（PyTorch 2.8.0 + CUDA 12.8；官方测的是 2.0.1，管线代码是纯 Python，在新 torch 上原样可用）；
  - Hugging Face 上的模型（都锁定版本，只下半精度文件）：SDXL 1.0 基础模型（UNet 约 5.1 GB + 两个文本编码器约 1.6 GB）、SDXL 深度 ControlNet（约 2.5 GB）、SDXL-VAE-FP16-Fix（约 0.33 GB）、给 ControlNet 算深度的 Intel DPT-Hybrid MiDaS（约 0.49 GB）。
  - 模型合计约 10 GB，加上环境约 15 GB，时间主要看网速。
- 不需要申请权限，不需要额外手动下载。装好后离线运行。
- 显卡：需要 NVIDIA 显卡，显存建议 16 GB 以上（峰值约 12.4 GB）。

## 许可证说明

**可以商用，但 SDXL 部分有用途限制。** 逐项：

- DiffusionLight-Turbo 代码、仓库自带的 Turbo LoRA 和曝光 LoRA：MIT（Hugging Face 上同名的 TurboLoRA 模型卡也标 MIT）。
- SDXL 1.0 基础模型、SDXL 深度 ControlNet：CreativeML Open RAIL++-M。允许免费使用、修改、商用和分发，生成的结果版权不归授权方；但**不能用于许可证附件 A 列出的用途**：违法用途；伤害或利用未成年人；制作或传播以伤害他人为目的的虚假信息；传播可用来伤害个人的身份信息；诽谤、骚扰他人；对个人合法权益有不利影响的全自动决策；歧视；医疗建议；执法、司法、移民审查等。再分发模型时要把这些限制原样写进你的授权条款。
- SDXL-VAE-FP16-Fix：MIT。Intel DPT-Hybrid MiDaS：Apache-2.0（不是 Depth Anything，没有非商用的深度模型）。

## 参考

- DiffusionLight-Turbo 论文：https://arxiv.org/abs/2507.01305
- DiffusionLight-Turbo 项目主页：https://diffusionlight.github.io/turbo/
- 代码：https://github.com/DiffusionLight/DiffusionLight-Turbo ，许可证 https://github.com/DiffusionLight/DiffusionLight-Turbo/blob/main/LICENSE
- Turbo LoRA 模型卡：https://huggingface.co/DiffusionLight/TurboLoRA
- 第一版 DiffusionLight（CVPR 2024）：论文 https://arxiv.org/abs/2312.09168 ，主页 https://diffusionlight.github.io/ ，代码 https://github.com/DiffusionLight/DiffusionLight
- SDXL 1.0 模型卡：https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 ，许可证原文 https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md
- SDXL 深度 ControlNet：https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0
- SDXL-VAE-FP16-Fix：https://huggingface.co/madebyollin/sdxl-vae-fp16-fix
- DPT-Hybrid MiDaS：https://huggingface.co/Intel/dpt-hybrid-midas
- 实验室主页：https://vistec.ist/vision ，NeX：https://nex-mpi.github.io/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
