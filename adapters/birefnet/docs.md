+++
team = "南开大学、上海人工智能实验室等（Nankai University, Shanghai AI Laboratory, …）"
people = "Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang, Nicu Sebe"
paper = "Bilateral Reference for High-Resolution Dichotomous Image Segmentation（CAAI Artificial Intelligence Research 2024）"
paper_url = "https://arxiv.org/abs/2401.03407"
website = ""
repo = "https://github.com/ZhengPeng7/BiRefNet"
year = 2024
+++

## 这是什么

论文摘要开头这样介绍它：一个面向高分辨率二分图像分割（dichotomous image segmentation，DIS）的
全新双边参考框架，名为 BiRefNet。它由两部分组成：借全局语义定位物体的定位模块（LM），以及带
双边参考（BiRef）的还原模块（RM）——还原时以画面的分层图块作为源参考、以梯度图作为目标参考，
两者配合给出最终结果。作者另加了梯度监督，让网络更关注细节多的区域。

在 Lab2Shot 里，它是「BiRefNet 抠像」：不点选、不画 roto，模型自己判断画面里的主体（人、物体、
动物、产品），逐帧输出 0–1 的 alpha。高分辨率和细边是它的长处——头发、镂空、细杆这类细节比一般
分割模型保得住，在 Nuke 里当遮罩拿来就用，也可以作为精修的起点。

## 输入输出

**官方要什么、给什么**

- 吃：画面，**没有任何提示**——没有遮罩、没有框、没有点选。官方 `inference.py` 收的是几个测试集目录
  （`--testsets`、`--ckpt`、`--pred_root`），每张画面按模型的分辨率缩放后成批送进网络。
- 给：`scaled_preds`——每张画面一张单通道的分割结果，缩回原尺寸后存成 PNG。

**我们怎么接的**

- 「图像」口就是上游的 `inputs`，「Alpha」口就是 `scaled_preds`，「处理分辨率」就是上游的 `--resolution`：
  一一对上，没有多的口，也没有少的。
- 上游逐张算，节点按序列逐帧调用同一条路径，所以前后帧之间没有约束（要整段稳定接视频抠像那一类节点）。

出处：简介来自论文摘要（arXiv 2401.03407）和仓库 `README.md`；输入输出依据 `third_party/birefnet/repo/inference.py`
和 `adapters/birefnet/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「抠像 · BiRefNet」）：读取序列 → **BiRefNet 抠像** →「序列图输出设置」写 alpha EXR，Nuke 里当遮罩（Copy 到 alpha、做 holdout、给 CG 做遮挡）。
- 也常用来给别的节点"喂"遮罩：比如当作视频抠像模型的引导遮罩，或给相机解算排除运动物体。
- 什么素材好：主体明确、占画面比例不小的镜头（人物中近景、产品、动物）。不好：好几个同样显眼的物体（模型不知道你要哪个，可能都抠或抠错）、主体很小、主体和背景颜色几乎一样。要指定"抠哪个"，用 SAM 3 视频分割点选或文字提示。
- 关键参数：
  - 模型：默认「抠像」，头发、运动模糊处有半透明过渡，人和动物用它；「通用分割」边缘干净接近黑白，适合硬边物体；2K/4K 素材要头发细节选「高分辨率抠像」（慢约 3 倍）；「任意比例」按原画面比例算，但容易把杂物也当主体。
  - 处理分辨率：留空就用训练尺寸（1024，高分辨率模型 2048），一般不用改；调大不一定更好，模型没见过那么大的图。
  - 半精度：默认开，更快更省显存，结果几乎一样。

## 效果和局限

RTX 4090 上（默认「抠像」模型、半精度）：

- sh020 iPhone 长焦背影行走（1080×1920，20 帧）：约 0.17 秒/帧（网络本身约 0.12 秒，其余是读写文件），加载约 3 秒。
- sh030 面部特写（772×855，113 帧）：约 0.10 秒/帧，整段 14 秒。
- 纯网络速度：1024 模型约 86 毫秒/帧；2048 高分辨率模型约 324 毫秒/帧。
- 显存：模型本身不到 0.5 GB，每帧约 1.5 GB；按显卡剩余显存自动决定一次算几帧（最多 4 帧），峰值约 9 GB。显卡被别的程序占着时会自动少算几帧，给别人留出显存。
- **手填处理分辨率的上限是 2048**（约 9 GB，会按显存自动降批大小）；更大的没有验证过，服务器按同样的上限拒绝更大的数字。

局限：

- **每帧单独算**，没有前后帧约束，边缘会有轻微"呼吸"闪烁；对稳定性要求高的镜头，把它的结果当作视频抠像模型（如 MatAnyone）的引导遮罩，再出最终 alpha。
- 只输出 alpha，不输出去溢色后的前景颜色。
- "主体是谁"由模型决定，一段镜头里主体换了（比如另一个人走进来抢了画面），结果也会跟着变。

## 团队

主要作者 Peng Zheng（南开大学、上海人工智能实验室），合作者来自南开大学、西北工业大学、国防科技大学、芬兰阿尔托大学（Aalto）、上海人工智能实验室和意大利特伦托大学。论文发表在 CAAI Artificial Intelligence Research（2024）。合作者之一 Deng-Ping Fan（范登平，南开大学）以伪装物体检测（SINet / COD10K 数据集，CVPR 2020 口头报告）闻名。BiRefNet 发布后被广泛采用，例如 BRIA 的 RMBG-2.0 背景去除模型就建立在 BiRefNet 结构上（RMBG-2.0 的权重是非商用的，本扩展不用它）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install birefnet`。会下载：
  - 锁定版本的官方代码（约 1 MB）；
  - 独立 Python 环境（PyTorch 2.9 + CUDA 12.8，约 7 GB）；
  - 五个官方权重（Hugging Face，锁定版本，下载后核对 SHA-256）：BiRefNet-matting（抠像，约 885 MB）、BiRefNet（通用分割）、BiRefNet_HR（高分辨率分割）、BiRefNet_HR-matting（高分辨率抠像）、BiRefNet_dynamic（任意比例），后四个各约 444 MB。模型合计约 2.7 GB。
- 不需要申请权限，不需要额外手动下载；装好后离线运行。

## 许可证说明

**可以商用。** 代码是 MIT，五个权重的 Hugging Face 模型卡也都标 MIT；MIT 只要求再分发时保留版权声明。

需要知道的一点：训练数据里有 DIS5K、P3M-10k、AM-2k、Distinctions-646 等**仅限学术研究**的数据集，作者仍然用 MIT 发布了权重。个人和研究使用没有问题；严格的商业项目交付前，建议请法务确认一次。

另外注意别混淆：BRIA 的 RMBG-2.0 用的也是 BiRefNet 结构，但它的权重是 CC BY-NC 4.0（非商用），本扩展不下载它。

## 参考

- 论文（arXiv）：https://arxiv.org/abs/2401.03407
- 期刊版（CAAI Artificial Intelligence Research 2024）：https://www.sciopen.com/article/10.26599/AIR.2024.9150038
- 代码：https://github.com/ZhengPeng7/BiRefNet ，许可证 https://github.com/ZhengPeng7/BiRefNet/blob/main/LICENSE
- 模型卡：https://huggingface.co/ZhengPeng7/BiRefNet-matting 、https://huggingface.co/ZhengPeng7/BiRefNet 、https://huggingface.co/ZhengPeng7/BiRefNet_HR-matting
- 在线演示：https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo
- 范登平的 SINet（伪装物体检测，CVPR 2020）：https://github.com/DengPingFan/SINet
- RMBG-2.0（基于 BiRefNet 结构，非商用权重）：https://huggingface.co/briaai/RMBG-2.0

<!-- 不要手写：节点列表、用到它的模板、权重文件表、安装状态、许可证徽标 —— 帮助页面会自动生成。 -->
