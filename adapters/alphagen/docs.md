+++
team = "Lightricks"
people = ""
paper = ""
paper_url = ""
website = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen"
repo = "https://github.com/Lightricks/LTX-2"
year = 2026
+++

## 这是什么

LTX-2.5 Alpha Gen 是 Lightricks 在 LTX-2.5 22B 视频模型上训练的 IC-LoRA（2026-10 发布，Beta 版）。给它一段普通的 RGB 画面，它输出逐帧对齐的 alpha 遮罩，不需要绿幕、遮罩或提示词。官方说它擅长难抠的半透明边缘：发丝、皮毛、烟、火、薄纱、玻璃、水。

在 Lab2Shot 里，这个节点叫「LTX-2.5 Alpha Gen 抠像」（`alphagen.matte`），**只用于抠像**：画面进，alpha 出。

## 输入输出

**官方要什么、给什么**（模型卡原话）

- **吃**：只有 RGB 画面。提示词永远为空（"The RGB video is the only guide"）；**不支持遮罩**，也不能指定抠哪个主体（"there is no prompt or mask control, and the model decides the foreground on its own"）。
- 画面要补边（不拉伸）到 32 的倍数，帧数补到 8n+1。最大 1920 × 1088、145 帧；更长时原画面会漏进遮罩，要切段。
- **给**：灰度的 alpha 视频（白色为前景），与输入逐帧对齐。不输出前景色。

**我们怎么接的**

- 「RGB」接原画面，「Alpha」输出 0–1 的浮点 alpha，尺寸与输入相同。节点属于抠像家族的无遮罩一档，和 BiRefNet 一样，可以参与抠像集成。
- 计算走共用的 LTX 运行层（`adapters/ltx/ltx_runtime.py`），配方写死在 `worker.py`：
  - 蒸馏版底模（加载时转为 fp8），LoRA 强度 1.0，空提示词，种子 1234；
  - 只跑第一阶段：原生分辨率、8 步、不做 CFG，只算视频。
- 镜头切成重叠的窗口，重叠处交叉淡化。每窗的帧数按显存定：1920 × 1080 时 49 帧，1280 时 113 帧，960 时 145 帧（官方上限）。
- 输入大于 1920 × 1088 时先缩小再算，alpha 用双线性插值放大回原尺寸。

**防止被当生成模型用**

- 节点上只有「处理分辨率」一个参数。提示词、种子、步数、LoRA、帧数都写死在 worker 里，用户看不到也改不了。
- 只交付 alpha：模型解码出来的灰度画面取三通道平均，变成单通道，模型生成的 RGB 一律不写出。
- 每个输入帧对应一个同尺寸的输出帧，不会多出帧：
  - 为满足模型要求补上的边和帧，在输出时全部裁掉；
  - worker 出口逐帧核对帧数和尺寸，对不上就报 E-ALPHAGEN-SHAPE，什么都不写出。

## 效果和局限

在 RTX 5090 上测试（默认处理分辨率 1920，每条取前 60 帧，指标口径与集零的抠像评测相同）。SAD、MSE ×1000、Grad、dtSSD 都是越小越好：

| 素材 | Alpha Gen | BiRefNet | MatAnyone 2 | VideoMaMa |
|---|---|---|---|---|
| VideoMatte motion 0000 | SAD 11.2 · MSE 1.34 · Grad 18.5 · dtSSD 4.62 | 8.1 · 0.19 · 3.0 · 1.77 | 7.7 · 0.18 · 2.4 · 1.66 | 8.5 · 0.24 · 4.1 · 2.03 |
| VideoMatte motion 0001 | 10.9 · 0.90 · 13.4 · 2.38 | 11.0 · 0.47 · 5.6 · 1.70 | 10.5 · 0.42 · 6.2 · 1.70 | 10.9 · 0.45 · 5.8 · 1.46 |
| VideoMatte motion 0002 | **6.3** · 0.19 · 3.3 · 0.69 | 7.6 · 0.15 · 2.3 · 0.69 | 7.5 · 0.18 · 2.6 · 0.71 | 7.9 · 0.18 · 3.0 · 0.58 |
| YouTubeMatte motion 0005（长卷发、甩手） | 12.3 · 2.64 · 34.6 · 6.73 | 5.3 · 0.67 · 9.9 · 3.25 | 6.3 · 1.20 · 15.8 · 4.06 | 5.5 · 0.66 · 9.9 · 3.35 |
| YouTubeMatte motion 0020（披肩长发） | 5.6 · 0.52 · 6.5 · 1.96 | 2.9 · 0.24 · 3.4 · 1.37 | 3.0 · 0.36 · 4.5 · 1.41 | 2.4 · 0.15 · 2.3 · 1.13 |

- 在这两个人像评测集上，**整体不如现有的抠像节点**：5 条平均 SAD 9.3，另外三个都在 7.0 左右；Grad（边缘清晰度）和 dtSSD（帧间稳定性）都明显更差。
  - 只有一条（VideoMatte motion 0002）SAD 最好。
  - 误差集中在主体边缘一两个像素宽的带子上：边缘偏软，快速运动的手臂边缘带着运动模糊。整体没有错位。
- 这两个评测集是绿幕人像合成或调和后的合成，正好不含它擅长的烟、火、玻璃、水，所以这些数字测不出它的强项。半透明素材上还没有带真值的对比。
- 不能选主体：画面里有多个主体时，由模型自己决定前景。
- **显存与速度**（RTX 5090）：
  - fp8 的 22B transformer 常驻显存 19.5 GB；1920 × 1080 时显存峰值 27.5 GB（torch 统计），只能在 32 GB 的卡上跑。
  - 60 帧 1080p 一段约 150–220 秒，其中加载约 50–70 秒；模型常驻后约 1.5–2.8 秒/帧。
  - 4090（24 GB）放不下。
  - 第一次运行时，要用 Gemma 文本编码器把空提示词编码一次并缓存，之后不再加载。

## 模型下载和安装

- `lab2shot ext install alphagen`。会一起装好共用底座 `ltx` 的代码、运行环境和底模文件，其他 LTX 功能扩展包已经装过的底模直接硬链接，不重复下载。下载内容（都校验 SHA-256）：
  - 底模约 70 GB；
  - LoRA 1.3 GB。
- 两个 Hugging Face 仓库（Lightricks/LTX-2.5 和 Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen）都有访问门槛：要先登录 Hugging Face，在仓库页面点「Agree and Access」，同意共享联系方式。

## 许可证说明

LTX-2.x Community License：

- 个人和年收入低于 1000 万美元的公司可以免费使用，包括商用；
- 年收入达到 1000 万美元的公司，除非商业的研究和评估外，都要向 Lightricks 购买授权；
- 再分发要附带许可证原文和其中的使用限制。

## 参考

- 模型卡：https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Alpha-Gen
- 代码：https://github.com/Lightricks/LTX-2
- 许可证原文：https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
