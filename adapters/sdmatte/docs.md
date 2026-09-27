+++
team = "vivo 影像研究院（vivo Camera Research）+ 上海大学"
people = "Longfei Huang, Yu Liang, Hao Zhang, Jinwei Chen, Wei Dong, Lunde Chen, Wanyu Liu, Bo Li, Peng-Tao Jiang"
paper = "SDMatte: Grafting Diffusion Models for Interactive Matting（ICCV 2025）"
paper_url = "https://arxiv.org/abs/2508.00443"
repo = "https://github.com/vivoCameraResearch/SDMatte"
year = 2025
+++

## 这是什么

SDMatte 是基于 Stable Diffusion 的交互式抠像方法，支持点、框、遮罩三种视觉提示，用来从自然画面里准确抠出指定的主体。作者的三条做法：借用扩散模型里预训练 U-Net 的先验，把文字驱动的交互机制换成视觉提示驱动的交互机制；把视觉提示的坐标嵌入和主体的不透明度嵌入并进 U-Net；再配一套带遮挡的自注意力，让模型只盯着提示指定的那一块。

在 Lab2Shot 里，它做精细抠像，**不用三分图**：给一张画面和一个指路的提示（一张粗遮罩，或者它的包围盒），SDMatte 重算整张发丝级 alpha。一次前向就出结果，不做逐步去噪，所以同一张画面算两遍结果一样。**这是接入的精细抠像里唯一可商用的一个**（MIT），其余几档（MatAnyone 2、VideoMaMa）都是非商用。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一张画面加一个**视觉提示**（说要抠哪一个），没有三分图。测试流水线（`data/dataset.py` 的 `phase="test"` 和 `inference.py`）：
  画面缩成 1024 × 1024 的正方形、值域 -1..1，提示同样处理（最近邻）；提示的包围盒归一化到 0..1 作为坐标嵌入；
  `is_trans` 说这个主体是不是透明物体（玻璃、烟、纱），它驱动模型的不透明度嵌入，上游按数据集逐个设定。
  模型入口 `modeling/SDMatte/meta_arch.py` 的两路输入就是 `rgb` 和 `aux_input`，
  `aux_input_type` 取 `mask` / `bbox_mask` / `point_mask`。
- **给**：alpha，一次前向出结果（`num_inference_steps=1`，不做多步去噪）；`inference.py` 把它缩回画面尺寸、按 0–255 存成图。**只有 alpha，没有前景颜色。**

**我们怎么接的**

- 「图像」= 上游的 `rgb`；「粗遮罩」= 上游的 `aux_input`；「提示」参数选的就是上游的 `aux_input_type`（按遮罩本身，还是按它的包围盒）。
- 「半透明主体」= 上游的 `is_trans`；「处理分辨率」= 上游那个正方形边长。
- 「Alpha」= 上游的 alpha，缩回画面尺寸并保持浮点；上游只出 alpha，所以节点上也只有这一个输出口。
- **不一样的两点**：① 上游按数据集设定 `is_trans`，这里把它交给艺术家（「半透明主体」）；
  ② 上游用 detectron2 的 LazyConfig 建模型、`DetectionCheckpointer` 读权重，这里从仓库文件导入模型类、
  按 `torch.save` 的 `["model"]` 读权重（数值一样，只是不装 detectron2，见 `adapters/sdmatte/worker.py`）。

**出处**：简介来自论文摘要（`third_party/sdmatte/repo/README.md`：「we propose SDMatte, a diffusion-driven interactive matting model…」）；
输入输出依据 `repo/inference.py`、`repo/modeling/SDMatte/meta_arch.py`、`repo/data/dataset.py`（phase="test"）
和 `adapters/sdmatte/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：读取序列 →「BiRefNet 抠像」或「SAM 3 视频分割」出粗遮罩 → 接到本节点的「粗遮罩」，画面接「图像」→ 输出 Alpha → 序列图输出设置（EXR），进 Nuke。
- **粗遮罩只用来指出抠哪一个**，不参与画边：模型从画面本身重算 alpha，所以粗遮罩糙一点没关系，指错人才要紧。
- **粗遮罩每一帧都要有**：逐帧独立计算，没有「跟着第一帧往下传」。缺帧的那几帧按空提示算、抠出来是空的，节点上会写明缺了几帧。
- 关键参数：
  - 提示：**遮罩**（把粗遮罩整张交给模型，形状复杂、有镂空时更准）/ **框**（只取粗遮罩的外接矩形，粗遮罩本身很糙时反而更稳）。
  - 处理分辨率（默认 1024）：官方测试尺寸，最稳；调小只为省显存。
  - 半透明主体：抠玻璃、烟、纱、水这类整体半透明的主体才打开，模型换一套不透明度先验。抠人、动物、实体道具保持关闭。
- 适合：任何主体，**要交付给客户 / 商业项目的镜头**。
  不适合：要整段时间上稳的软边（用「MatAnyone 2 精细抠像」或「VideoMaMa 精细抠像」）。

## 效果和局限

速度和显存（1920×1080 实拍、粗遮罩用 BiRefNet）：

| 处理分辨率 | RTX 4090 每帧 | RTX 5090 每帧 | 显存保留峰值（两张卡一样） |
|---|---|---|---|
| 1024 | 0.672 秒 | 1.03 秒 | 14560 MB |
| 512 | 0.149 秒 | 0.13 秒 | 7680 MB |

（5090 那一列测量时卡上还有别的任务在跑，数字偏慢。）

**基准对比**（抠像误差 SAD，越小越好；每格是「各条素材的中位数 / 均值」）：

P3M-500-NP 5 张实拍人像（人工 alpha 真值）：

| 粗遮罩来自 | SDMatte | MatAnyone 2（非商用） |
|---|---|---|
| 「BiRefNet 抠像」 | 4.97 / **7.13** | **4.80** / 8.93 |
| 「SAM 3 视频分割」 | **10.95 / 14.26** | 14.81 / 16.41 |

CRGNN 实拍视频一段（1920×1080、102 帧，SAD / 边缘误差 grad）：

| 粗遮罩来自 | SDMatte | MatAnyone 2（非商用） |
|---|---|---|
| 「BiRefNet 抠像」 | 29.57 / 4.49 | **28.82 / 3.81** |
| 「SAM 3 视频分割」 | 31.48 / 4.37 | **29.03 / 3.80** |

（两张表的 SAD 不能横着比：它是整幅画面的误差和，随分辨率和主体大小走。）

**可商用许可里最准的一档**：和非商用最强的 MatAnyone 2 差在 3% 以内——静态图中位数差 3.5%、
视频段差 2.6%，而按均值算静态图上 SDMatte 反而更好（7.13 对 8.93）。粗遮罩换成 SAM 3 的时候它最稳——
粗遮罩只用来指哪一个，不参与画边，所以遮罩糙一点不要紧。
整段素材要「一帧都不闪」仍然是 MatAnyone 2 更稳（它有跨帧记忆，SDMatte 逐帧独立算）。

局限：

- **官方按 1024×1024 的方图计算，再把 alpha 缩回原尺寸**（这是作者的测试流程）。
  1080p 素材的长边本来就比 1024 长、而且被压成正方形，所以发丝细节到此为止：
  节点的「处理分辨率」写明了这一点。
- **逐帧独立**，没有前后帧约束：主体快速运动时边缘会有轻微呼吸。
- **只输出 alpha**，不输出去溢色的前景颜色。
- 模型大（U-Net + VAE + 文字编码器），权重 12 GB，显存和耗时都不低。
- 训练数据里有 Composition-1K、RefMatte 这些只许研究用的数据集。代码和权重的许可是 MIT，
  但严格的商业交付前建议做一次法务确认（每个「可商用」项目都有这一条提醒）。

## 团队

vivo 影像研究院（vivo Camera Research）与上海大学。论文 ICCV 2025。
同一权重页还放了 **SDMatte\***（`SDMatte_plus.pth`，换一套训练数据）和蒸馏的 **LiteSDMatte**（更小更快），
目前只接入主档 SDMatte。

## 模型下载和安装

- 安装：`lab2shot ext install sdmatte`。下载固定版本的官方代码、独立 Python 环境（PyTorch 2.10）和权重。
- 权重 `SDMatte.pth` 约 **12 GB**（Hugging Face `LongfeiHuang/SDMatte`，按 SHA-256 校验），
  加上网络配置文件；**Stability AI 的原始权重一个都不下载**——网络结构从那些配置文件搭起来，
  里面的每一个数都来自作者训练的 SDMatte.pth。
- 不需要申请权限，不需要额外手动下载。

## 许可证说明

**可商用。** 仓库的 LICENSE 是 MIT，Hugging Face 权重页也写 `license: mit`。
网络结构改自 Stable Diffusion 2，但不下载、也不再分发 Stability AI 的任何权重文件，
所以 CreativeML Open RAIL++-M 不随权重传过来。
训练数据里有仅限研究的数据集（Composition-1K、RefMatte），严格的商业交付前建议做一次法务确认。

## 参考

- 论文：https://arxiv.org/abs/2508.00443
- 代码：https://github.com/vivoCameraResearch/SDMatte
- 权重：https://huggingface.co/LongfeiHuang/SDMatte
- 蒸馏版权重（暂未接入）：https://huggingface.co/LongfeiHuang/LiteSDMatte
- 许可证原文：https://github.com/vivoCameraResearch/SDMatte/blob/main/LICENSE
