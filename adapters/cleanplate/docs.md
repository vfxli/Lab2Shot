+++
team = "Lightricks"
people = ""
paper = ""
paper_url = ""
website = "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate"
repo = "https://github.com/Lightricks/LTX-2"
year = 2026
+++

## 这是什么

LTX-2.5 Clean Plate 是 Lightricks 在 LTX-2.5 22B 视频模型上训练的 IC-LoRA。它把画面里的人、行人和车去掉，补出被挡住的背景，得到同一镜头的干净背景板（clean plate）。固定机位和运动机位的素材都适用。

在 Lab2Shot 里，这个节点叫「LTX-2.5 Clean Plate 干净背景板」（`cleanplate.clean_plate`）：画面进，同样帧数、同样尺寸的干净背景板出。

## 输入输出

**官方要什么、给什么**（模型卡原话）

- **吃**：原镜头作为参考视频（"the original clip containing the subjects to remove"），再加一段描述空场景的提示词，建议用负提示点名不该出现的东西。
- **不支持遮罩**（"a full-frame model that removes dynamic subjects globally; it does not take a user-supplied region mask"）。
- 训练用 1024 × 576、49 帧、25 fps，1920 × 1088 也验证过。边长补到 32 的倍数，帧数补到 8n+1。
- **给**：与参考视频逐帧对齐、取景相同的干净画面。

**我们怎么接的**

- 「RGB」接原画面，「干净背景板」输出显示 sRGB 的画面，尺寸和帧号与输入相同。
- 计算走共用的 LTX 运行层（`adapters/ltx/ltx_runtime.py`），配方写死在 `worker.py`：
  - 蒸馏版底模（fp8），LoRA 强度 1.0，种子 1234，25 fps；
  - 正提示用模型卡给的通用空场景描述，只编码一次并缓存；
  - 只跑第一阶段，不做 CFG。官方 ComfyUI 单阶段蒸馏流程的 CFG 也是 1，负提示在 CFG 1 时不起作用，所以不编码。
- 每窗最多 49 帧（模型卡的训练长度；各处理分辨率都一样），重叠 17 帧并交叉淡化。

**严格是处理，不是生成**

- 节点上只有「处理分辨率」一个参数，提示词、种子、步数、LoRA、帧数都写死，用户看不到也改不了。
- 每个输入帧对应一个同尺寸、同帧号的输出帧，不会多出帧；补上的边和帧在输出时全部裁掉。
- worker 出口核对帧数和尺寸，对不上就报 E-CLEANPLATE-SHAPE，什么都不写出。

## 效果和局限

在 RTX 5090 上实测（默认 1920，每条 49 帧）：

- **3DPW 两人过马路**（1920 × 1080，固定机位）：两个人和横穿马路的女人都去干净了，咖啡座里坐着的人也一起去掉了，桌椅和遮阳伞保留，背景稳定。
- **MOT17-11 商场手持**（1920 × 1080，手持跟拍）：整段的人都去掉了，地面反光里的人影也一起消失，结果很干净。
- **KITTI 车载前视**（1242 × 375，车在开）：开头的面包车、骑车人和行人都去掉了；但后半段镜头前移、画面变化大，补出来的背景明显糊、有涂抹感，路边停着的自行车也被一起抹掉了。
- 局限：
  - 只能整帧处理，会把画面里所有的人和车都去掉，坐着不动的人也算在内；
  - 人或车占了大半画面时，补不出背景；
  - 补出来的内容是模型「想出来」的，可能和真实背景不一样，比如墙上多出或少了涂鸦；
  - 镜头快速前移的素材效果差。
- **显存与速度**（RTX 5090）：
  - 1080p 时显存峰值 27.0 GB（torch 统计），只能在 32 GB 的卡上跑；
  - 模型常驻后，49 帧 1080p 约 75 秒（1.5 秒/帧）；第一次加载另加约 50 秒；
  - 第一次运行时，要用 Gemma 文本编码器把固定提示词编码一次并缓存，之后不再加载。

## 模型下载和安装

- `lab2shot ext install cleanplate`。共用底座 `ltx` 的底模（约 70 GB）已经装过的话直接硬链接，只下载 LoRA（327 MB）。都校验 SHA-256。
- 两个 Hugging Face 仓库（Lightricks/LTX-2.5 和 Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate）都有访问门槛：要先在仓库页面点「Agree and Access」。

## 许可证说明

LTX-2.x Community License：个人和年收入低于 1000 万美元的公司可以免费使用，包括商用；年收入达到 1000 万美元的公司，除非商业的研究和评估外，都要向 Lightricks 购买授权。

## 参考

- 模型卡：https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Clean-Plate
- 代码：https://github.com/Lightricks/LTX-2
- 许可证原文：https://github.com/Lightricks/LTX-2/blob/main/LICENSE-2_x
