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

在 Lab2Shot 里，它接成两个节点。「BiRefNet 抠像」：不点选、不画 roto，模型自己判断画面里的主体
（人、物体、动物、产品），逐帧输出 0–1 的 alpha。高分辨率和细边是它的长处——头发、镂空、细杆这类
细节比一般分割模型保得住，在 Nuke 里当遮罩拿来就用，也可以作为精修的起点。「BiRefNet 边缘解混合」：
官方 `refine_foreground` 后处理，把边缘半透明带里混进来的背景色反解掉，给出干净的前景图。

## 输入输出

**官方要什么、给什么**

- 吃：画面，**没有任何提示**——没有遮罩、没有框、没有点选。官方 `inference.py` 收的是几个测试集目录
  （`--testsets`、`--ckpt`、`--pred_root`），每张画面按模型的分辨率缩放后成批送进网络。
- 给：`scaled_preds`——每张画面一张单通道的分割结果，缩回原尺寸后存成 PNG。
- 另有一个官方后处理 `refine_foreground(image, mask, r=90)`（`image_proc.py`，官方单图 / 视频两个
  教程都在用）：吃画面 + 一张 alpha，给去掉了背景色污染的前景图（FB blur fusion，官方致谢
  PhotoRoom 的 fast-foreground-estimation）。

**我们怎么接的**

- 「RGB」口就是上游的 `inputs`，「Alpha」口就是 `scaled_preds`；处理尺寸固定为各权重的训练尺寸
  （相当于上游 `--resolution` 永远填训练尺寸，不给用户改）。一一对上，没有多的口，也没有少的。
- 上游逐张算，节点按序列逐帧调用同一条路径，所以前后帧之间没有约束（要整段稳定接视频抠像那一类节点）。
- 「BiRefNet 边缘解混合」的「RGB」+「Alpha」就是 `refine_foreground` 的 `image` + `mask`，「前景」口是
  它的 `estimated_foreground`（预乘 alpha 后按家族约定写成 RGBA）。官方函数不挑 alpha 来源，所以
  接 MatAnyone、SAM 3 甚至手工 roto 的 alpha 都可以。
- 「未修正前景」口不是官方的输出：它是官方函数收到的那张 `image` 原样乘 alpha（预乘 RGBA），给 Nuke 里
  算修正量用（见下面「在 Nuke 里当修正量用」）。和「前景」口走同一条写出路径、同一个色彩空间标记。
- 官方函数有 GPU（`FB_blur_fusion_foreground_estimator_gpu_2`）和 CPU（`..._cpu_2`，cv2.blur）两个版本，
  本节点用 CPU 版（实测够快，不占显卡，数字见「效果和局限」）。两者的差别只在偶数模糊核（90、6）的
  半像素对齐：官方 GPU 版的 `mean_blur` 补边方向和 cv2.blur 差一个像素，写出的预乘颜色最多差约 0.05
  （sRGB 编码值），只在边缘带；换成奇数核两者一致到 2×10⁻⁶。

出处：简介来自论文摘要（arXiv 2401.03407）和仓库 `README.md`；输入输出依据 `third_party/birefnet/repo/inference.py`
和 `adapters/birefnet/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法（模板「抠像 · BiRefNet」）：读取序列 → **BiRefNet 抠像** →「序列图输出设置」写 alpha EXR，Nuke 里当遮罩（Copy 到 alpha、做 holdout、给 CG 做遮挡）。
- 要直接可合成的前景：读取序列 → **BiRefNet 抠像** → **BiRefNet 边缘解混合**（同一读取序列再拉一条 RGB 线进来）→「序列图输出设置」写 RGBA EXR，Nuke 里当前景合成，边缘不带旧背景的颜色。
- 在 Nuke 里当修正量用（不直接用 AI 的颜色）：「前景」和「未修正前景」各接一个「序列图输出设置」写成 EXR（两路的色彩转换完全相同，都转到交付色彩空间，例如 ACEScg）。在 Nuke 里：
  1. 两路都读进来（ACEScg，线性）。
  2. `修正值 = 前景 − 未修正前景`（Merge，operation 选 minus，A 接「前景」、B 接「未修正前景」）。
  3. `结果 = 你自己的 plate × alpha + k × 修正值`：k = 0 完全不用 AI 的颜色，k = 1 复现官方结果，中间是线性插值，而且是在线性空间里做的。
  - 为什么可以这样用：alpha = 1 的实心区域，官方函数的输出就是原图，两路逐位相同，修正值为 0；alpha = 0 处两路都是 0。所以修正值只在 0 < alpha < 1 的边缘带不为 0，你自己的浮点 plate 在实心区域一个字节都不会被碰到；本节点输入只有 8 bit（平台约定），这点损失只落在边缘带的修正量上。已用 4K 实拍素材（24 帧）验证：写出的两路 EXR 在 alpha = 1 处逐位相同、alpha = 0 处都是 0、差值只出现在边缘带；两路按交付的同一个转换转到 ACEScg 后相减，实心区域和 alpha = 0 处的差值都是 0。
  - 不直接输出差值层的原因：交付时颜色层要从工作空间（sRGB 编码的 Rec.709）转到 ACEScg，转换里有非线性的传递函数，差值和负值过了它就不对了；两路都是合法的颜色层，各自转换后再在线性空间相减才对。
- 也常用来给别的节点"喂"遮罩：比如当作视频抠像模型的引导遮罩，或给相机解算排除运动物体。
- 什么素材好：主体明确、占画面比例不小的镜头（人物中近景、产品、动物）。不好：好几个同样显眼的物体（模型不知道你要哪个，可能都抠或抠错）、主体很小、主体和背景颜色几乎一样。要指定"抠哪个"，用 SAM 3 视频分割点选或文字提示。
- 关键参数：
  - 模型：五个都是官方原名，括号里是官方模型卡写的训练尺寸。默认「BiRefNet_HR-matting（2048×2048）」：抠像质量与头发细节最好，慢约 3 倍；要快选「BiRefNet-matting（1024×1024）」（头发、运动模糊处同样有半透明过渡）；「BiRefNet（1024×1024）」边缘干净接近黑白，适合硬边物体（2048 版是「BiRefNet_HR」）；「BiRefNet_dynamic（256–2304 任意尺寸）」按原画面分辨率算，但容易把杂物也当主体。
  - 处理尺寸由模型决定，没有参数：方形模型按训练尺寸压扁计算（结果再缩放回原画面大小）；BiRefNet_dynamic 按原画面分辨率算（对齐到 32 的倍数），超出训练范围 256–2304 才等比缩进范围。
  - 半精度：默认开，更快更省显存，结果几乎一样。

## 效果和局限

RTX 4090 上（半精度；前两条为 BiRefNet-matting 1024 的实测整段）：

- sh020 iPhone 长焦背影行走（1080×1920，20 帧）：约 0.17 秒/帧（网络本身约 0.12 秒，其余是读写文件），加载约 3 秒。
- sh030 面部特写（772×855，113 帧）：约 0.10 秒/帧，整段 14 秒。
- 纯网络速度：1024 模型约 86 毫秒/帧；2048 模型（BiRefNet_HR-matting）约 324 毫秒/帧。
- 实测（RTX 4090，4K 实拍 24 帧，半精度）：默认 BiRefNet_HR-matting 0.46 秒/帧（网络本身 0.44 秒）；BiRefNet_dynamic（3840×2160 按 2304×1280 算）0.36 秒/帧；BiRefNet-matting（1024）约 0.15 秒/帧（含把 4K 缩到 1024 的时间）。
- 显存按所选模型分档实测（整卡峰值，含 CUDA 本身；模型权重约 0.8 GB；半精度开和关取较大的那个），调度器按所选模型的这个数挑显卡。worker 先用 1 帧量出每帧要多少显存，再按显卡剩余显存决定一次算几帧（激活显存不超过 8 GB），所以 2048 和 2304 的模型永远一次一帧，1024 的模型最多一次 2 帧：
  - BiRefNet-matting、BiRefNet（1024×1024）：约 8.5 GB（一次 2 帧；半精度 7.6 GB，关掉半精度 8.4 GB）。
  - BiRefNet_HR-matting（默认）、BiRefNet_HR（2048×2048）：约 15.5 GB（半精度 15.0 GB，关掉半精度 15.4 GB）。
  - BiRefNet_dynamic：随画面变。16:9 的 4K（按 2304×1280 / 2304×1216 算）约 10.3–11.0 GB；方形画面（2304×2304，面积最大）约 18.3 GB，关掉半精度 18.6 GB——调度按这个最坏情况 18.7 GB 算。
- 「BiRefNet 边缘解混合」在 CPU 上算，不占显卡。默认半径 90，只算官方函数本身：1080p 约 0.095 秒/帧，4K 约 0.43–0.50 秒/帧（AMD Ryzen 9 9950X3D；开 1 个线程和 32 个线程几乎一样）；放在真实任务里（同时在读 PNG、写结果），1080p 实测 0.13 秒/帧、4K 0.68 秒/帧。GPU 版是 1080p 0.020 秒、4K 0.114 秒，但为两次均值模糊占一整张卡不值得。纯后处理，没有模型加载。

局限：

- **每帧单独算**，没有前后帧约束，边缘会有轻微"呼吸"闪烁；对稳定性要求高的镜头，把它的结果当作视频抠像模型（如 MatAnyone）的引导遮罩，再出最终 alpha。
- 抠像节点只输出 alpha；要直接可合成的前景（边缘去污染），接「BiRefNet 边缘解混合」。
- 12 GB 左右的显卡跑不了默认的 BiRefNet_HR-matting（约 15.5 GB）和 BiRefNet_dynamic（最坏 18.7 GB），换成 1024 的模型（约 8.5 GB）就可以。
- 「BiRefNet 边缘解混合」的「Alpha」缺帧时，缺的那几帧跳过、不输出（节点会提示缺了哪几帧），不是按空遮罩算。
- "主体是谁"由模型决定，一段镜头里主体换了（比如另一个人走进来抢了画面），结果也会跟着变。
- 「BiRefNet 边缘解混合」是统计方法（邻域模糊），不是模型：背景很乱或与前景颜色接近时，边缘估出的颜色会糊一些；官方半径 90 按常规尺寸调的，高分辨率素材可以调大。它只解 alpha 0–1 边缘带的颜色、**实心区域原样直通**（主体上的绿反光要去掉得用常规去溢色），也**不修正 alpha 本身**——alpha 不准的地方前景也不会准。

## 团队

主要作者 Peng Zheng（南开大学、上海人工智能实验室），合作者来自南开大学、西北工业大学、国防科技大学、芬兰阿尔托大学（Aalto）、上海人工智能实验室和意大利特伦托大学。论文发表在 CAAI Artificial Intelligence Research（2024）。合作者之一 Deng-Ping Fan（范登平，南开大学）以伪装物体检测（SINet / COD10K 数据集，CVPR 2020 口头报告）闻名。BiRefNet 发布后被广泛采用，例如 BRIA 的 RMBG-2.0 背景去除模型就建立在 BiRefNet 结构上（RMBG-2.0 的权重是非商用的，本扩展不用它）。

## 模型下载和安装

- 自动安装：`uv run lab2shot ext install birefnet`。会下载：
  - 锁定版本的官方代码（约 1 MB）；
  - 独立 Python 环境（PyTorch 2.9 + CUDA 12.8，约 7 GB）；
  - 五个官方权重（Hugging Face，锁定版本，下载后核对 SHA-256）：BiRefNet-matting（约 885 MB）、BiRefNet、BiRefNet_HR、BiRefNet_HR-matting、BiRefNet_dynamic，后四个各约 444 MB。模型合计约 2.7 GB。
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
