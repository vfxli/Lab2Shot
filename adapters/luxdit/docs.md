+++
team = "NVIDIA 多伦多 AI 实验室（NVIDIA Toronto AI Lab / Spatial Intelligence Lab）+ 多伦多大学、Vector Institute"
people = "Ruofan Liang, Kai He, Zan Gojcic, Igor Gilitschenski, Sanja Fidler, Nandita Vijaykumar, Zian Wang"
paper = "LuxDiT: Lighting Estimation with Video Diffusion Transformer（NeurIPS 2025）"
paper_url = "https://arxiv.org/abs/2509.03680"
website = "https://research.nvidia.com/labs/toronto-ai/LuxDiT/"
repo = "https://github.com/nv-tlabs/LuxDiT"
year = 2025
+++

## 这是什么

上游自己的话（Overview）：LuxDiT 是一个生成式的光照估计模型，从视觉输入预测高质量的 HDR 环境贴图；它给出的光照准确，同时保留场景语义，能在各种条件下把虚拟物体真实地放进画面。论文《LuxDiT: Lighting Estimation with Video Diffusion Transformer》。

在 Lab2Shot 里，它是「LuxDiT HDRI」：吃一帧画面（也可以是一小段镜头），一次交出整张 360° 经纬图（lat-long）的线性 HDR EXR，可以当穹顶灯（dome light / skydome）给 CG 物体打光。**非商用。**

## 输入输出

**官方要什么、给什么**

- 吃：一个画面目录。两步命令（`third_party/luxdit/repo/README.md`）：
  ① `python inference_luxdit.py --config configs/luxdit_base.yaml --transformer_path $DIT_PATH
  --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR --resolution 480 720 --guidance_scale 2.5
  --num_inference_steps 50 --seed 33`；
  ② `python hdr_merger.py --model_path checkpoints/hdr_merge_mlp --input_dir $OUTPUT_DIR/ldr_log
  --output_dir $OUTPUT_DIR/hdr`。视频那一路要加 `--data_type video`。
  **不吃相机、不吃遮罩。**
- 给：第一步出两张色调映射过的环境图，也就是 `ldr_log` 目录里的 `*_ldr.png` / `*_log.png`；
  第二步把那两张喂给 HDR 合成网络（`hdr_merger.py`），写出 HDR 的 `.exr`。

**我们怎么接的**

- 「图像」口就是 `--input_dir` 那一段画面，「去噪步数」= `--num_inference_steps`、
  「引导强度」= `--guidance_scale`、「随机种子」= `--seed`、「贴近画面程度」= LoRA 的 `lora_scale`。
- 「HDRI」口就是第二步写出的 `.exr`，「显示图」口就是第一步的 LDR 经纬图：两样都是上游自己的产物，
  一个节点里跑完官方那两步。

出处：简介来自 `third_party/luxdit/repo/README.md`（Overview）；输入输出依据同一份 README、`hdr_merger.py`
和 `adapters/luxdit/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法和 DiffusionLight 一样：读取序列 → **LuxDiT HDRI** →「HDRI」接序列图输出设置写 EXR；「灯光」接 USD 输出设置，得到一盏挂好 HDRI 的 USD 穹顶灯，Houdini Solaris 里引用，Maya 用 mayaUsd 导入。接上相机（ViPE / COLMAP 解算的，或读取的 3DE 相机）时，穹顶灯按取样那一帧的相机朝向摆好。
- 两个环境光节点的输出方向完全一致：**经纬图正中间那一列就是镜头看的方向**，顶上是画面的上方，所以可以随时换着比。
- 什么素材好：普通焦段（竖直视角大约 50°–80°，也就是常见的 24–35 mm 全画幅等效）、环境看得比较全的一帧。它训练时画面都是这个视角范围，**长焦镜头（比如 iPhone 的 5 倍长焦）会被当成广角来理解**，猜出来的环境会跑偏。
- 画面会先缩放并从中间裁成模型的输入尺寸：横幅裁 720×480，竖幅裁 480×720，接近方形裁 512×512；裁掉的边缘模型看不到。
- 关键参数：
  - 随机种子：**结果对它很敏感**，尤其是画面外的主光方向。同一帧换种子，太阳可能从右前方跑到正上方。多试两三个，挑和画面阴影对得上的。
  - 取样帧：留空用镜头中间那一帧。挑环境最完整、人挡得最少的一帧。
  - LoRA 强度（默认 0.8）：越高越像真实照片、画面内容越多地"贴"进环境图；0 是官方只用合成数据训练的原始模型（亮度范围更夸张）。

## 效果和局限

RTX 4090 上（种子 0，50 步）：

- 长焦阴天街道一帧（1080×1920 竖幅）：约 29 秒（加载模型 12 秒 + 生成 16 秒），显存峰值 13.3 GB。
- 室内舞蹈一帧（864×480）：约 25 秒，显存峰值 13.3 GB。
- 同样两帧 DiffusionLight：约 82 秒和 49 秒，显存约 12.4 GB。

效果：

- **室内明显好于 DiffusionLight**：生成的是一个像样的房间，天花板上有一排排灯，动态范围（最亮 / 中间值）约 1400 倍；DiffusionLight 同一帧只有约 150 倍，而且铬球展开后变形严重。
- **方向是对的**：用官方示例 HDRI 做一个"太阳在右前方 55°、高 15°"的测试画面，LuxDiT 给出的最亮方向是右 53°、高 13°；画面内容贴进环境图时左右不会镜像。
- **长焦 + 阴天不可靠**：它把阴天街道猜成了晴天的公园（有太阳，最亮值 150 以上），换种子太阳位置也在变（右 97°、右 35°、右 29°）；DiffusionLight 反而给出更像阴天的灰色天空。这类镜头建议两个都算，对照画面挑。
- 亮度是相对的：线性 1.0 大约等于画面本身的白，不是绝对照度；太阳这类点光源的强度偏差会很大，DCC 里一般要调强度或补一盏平行光。
- 模型自己的环境图只有 256×128，默认输出 1024×512 是插值放大的，细节是糊的，适合打光，不适合当背景看。
- 结果是"猜"的，适合做灯光参考和快速初版，不能替代现场 HDRI。
- 节点只把取样那一帧发给它，走的是单帧模式。

## 团队

NVIDIA 多伦多 AI 实验室（负责人 Sanja Fidler，现在 GitHub 上叫 NVIDIA Spatial Intelligence Lab）和多伦多大学、Vector Institute 合作完成，第一作者 Ruofan Liang。同一团队还做过 DiffusionRenderer（用视频模型拆反照率/法线并重新打光，CVPR 2025）、GEN3C（可控相机的视频生成）、GET3D（生成带贴图的三维模型），以及 Lab2Shot 已经在用的相机解算 ViPE。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install luxdit`。会下载：
  - 锁定版本的官方代码（约 60 MB）；
  - 独立 Python 环境（PyTorch 2.8 + CUDA 12.8，约 6.7 GB）；
  - Hugging Face 上的模型（锁定版本，全部校验 sha256）：LuxDiT 单帧模型（11.1 GB，5B 参数）和它的真实场景 LoRA（0.6 GB）、HDR 合成小模型（53 KB）；CogVideoX-5B-I2V 的视频 VAE（0.86 GB）和采样器配置。不下载 CogVideoX 的文本编码器。
  - 模型合计约 24 GB，加上环境约 31 GB。网络顺畅时下载约 20 分钟，时间主要看网速。
- 不需要申请权限，不需要手动下载。装好后离线运行。
- 显卡：需要 NVIDIA 显卡，显存 16 GB 以上（峰值约 13.4 GB）。

## 许可证说明

**不能商用，只能做研究或评估。** 逐项：

- LuxDiT 代码和全部权重（单帧模型、视频模型、两个 LoRA、HDR 合成模型）：NVIDIA OneWay Noncommercial License。只能非商业使用（原文："research or evaluation purposes only"），生成的结果也不能用在商业项目里；再分发时要附带同一份许可证、保留版权声明。仓库里从 diffusers 改来的 CogVideoX 模型文件头写的是 Apache-2.0，但随整个仓库一起按 NVIDIA 非商用许可发布。
- CogVideoX-5B-I2V 的 VAE：CogVideoX License。学术研究免费；商用要先在智谱开放平台登记拿授权；禁止用于军事和违法用途。
- 不依赖 nvdiffrast（NVIDIA 另一个非商用的光栅化库）：代码里只是可选导入，训练时才用，推理不需要，不安装。

## 参考

- 论文：https://arxiv.org/abs/2509.03680
- 项目主页：https://research.nvidia.com/labs/toronto-ai/LuxDiT/
- 代码：https://github.com/nv-tlabs/LuxDiT ，许可证原文 https://github.com/nv-tlabs/LuxDiT/blob/main/LICENSE.md
- 模型卡：https://huggingface.co/nvidia/LuxDiT
- 基础模型 CogVideoX-5B-I2V：https://huggingface.co/zai-org/CogVideoX-5b-I2V ，许可证 https://huggingface.co/zai-org/CogVideoX-5b-I2V/blob/main/LICENSE ，代码 https://github.com/zai-org/CogVideo ，商用登记 https://open.bigmodel.cn/mla/form
- 同团队：DiffusionRenderer https://research.nvidia.com/labs/toronto-ai/DiffusionRenderer/ ，GEN3C https://research.nvidia.com/labs/toronto-ai/GEN3C/ ，GET3D https://nv-tlabs.github.io/GET3D/ ，ViPE https://github.com/nv-tlabs/vipe
- 实验室主页：https://research.nvidia.com/labs/toronto-ai/

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
