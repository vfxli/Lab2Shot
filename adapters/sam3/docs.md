+++
team = "Meta 超级智能实验室（Meta Superintelligence Labs）"
people = "Nicolas Carion, Laura Gustafson, Yuan-Ting Hu, …, Pengchuan Zhang, Christoph Feichtenhofer"
paper = "SAM 3: Segment Anything with Concepts（2025）"
paper_url = "https://arxiv.org/abs/2511.16719"
website = "https://ai.meta.com/sam3"
repo = "https://github.com/facebookresearch/sam3"
year = 2025
+++

## 这是什么

SAM 3 是一个可提示分割的统一基础模型，画面和视频都管：用文字，或者点、框、遮罩这类视觉提示，检出、分割并跟住物体。和前一代 SAM 2 相比，它新增的能力是把一个开放词表概念（一句短文字词组，或者几个示例）在画面里的每一个实例都穷尽地分出来，能接住的开放词表提示也比以往的方法多得多。

在 Lab2Shot 里，我们拿它做自动 Roto，交两样：

- **遮罩**：所有物体合在一张 alpha 里；
- **物体分割图**：每个物体一个编号值（背景 0，第 1 个物体 1，第 2 个物体 2……），相当于一张 Object ID pass，合成时可以按编号单独取某个物体。

它是带记忆的视频模型，会参考前面帧的结果往后跟，所以同一个物体在整段里编号不变，边缘也不会一帧一帧乱跳。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：一张画面或一段视频，加一个提示——一句短文字词组，或点、框、遮罩这类视觉提示（`third_party/sam3/repo/README.md:49`）。
- **给**：每个实例的遮罩；视频里由它的检测器加追随器把每个实例跟住，同一个物体整段一个 id（`third_party/sam3/repo/README.md:49`、`:51` 的 decoupled detector–tracker）。

**我们怎么接的**

- 「RGB」口 = 上游的那段视频；「提示词」参数 = 上游的文字提示（只认英文短语）；
  「人物框」口 = 上游的框提示：每个人第一次出现那一帧的框作为他这个 id 的提示，接了框之后文字提示就不起作用（`adapters/sam3/worker.py:6-16`）。
- 「遮罩」= 上游 `out_binary_masks` 合成一张；「物体分割」= `out_obj_ids` 那一套编号（两个口由分割家族 `lab2shot/nodes/families/segmentation.py` 给，`adapters/sam3/nodes.py:35-36`）。
- **不一样的两点**：① 上游三种提示里的**点提示和遮罩提示我们没有做成口**，节点上只有文字和框两种；
  ② 上游一次算一整段，我们把长镜头切成有重叠的段、一段一个推理会话（显存按段长封顶而不是按镜头长度），
  文字那一路在共享帧上按遮罩重叠把编号接起来，框那一路把上一段的最后一张遮罩当作下一段的遮罩提示。

**出处**：简介抽自 `third_party/sam3/repo/README.md:49`（「SAM 3 is a unified foundation model for promptable segmentation…」那两句）；
输入输出依据同一行、`third_party/sam3/repo/sam3/model/sam3_base_predictor.py:119-245` 和 `adapters/sam3/nodes.py:35-36`、`adapters/sam3/worker.py:6-26`。

## 在 Lab2Shot 里怎么用

- **典型接法**：读取序列 → SAM 3 视频分割 → 序列图输出设置（遮罩和物体分割图各接一个）。内置模板「视频分割 · SAM 3」就是这样接的。
- **按人分别跟**：先接「ViTDet 人物框」（需要时再接「选人」），把人物框接到 SAM 3 的「人物框」输入上，每个人按他在人物框里的编号跟到底。接了人物框以后，提示词就不起作用了。
- **给相机解算做遮罩**：遮罩可以接到「COLMAP 相机解算」的「运动物体遮罩」输入上，把会动的人和车排除掉再解算。
- **什么素材效果好**：物体轮廓清楚、和背景分得开的效果最好。它是用来分出「哪块是这个物体」的，不是精细抠像：发丝、运动模糊、半透明的边缘需要精细 alpha 时，再接「BiRefNet 抠像」等抠像节点。
- **关键参数**：
  - **提示词**：只认英文短语，可以带描述（the dog on the left）。
  - **检测阈值**：默认 0.5。有物体没被找到就调低（如 0.35）；不相关的物体也被选进来了就调高。
  - **最多物体数**：默认 8，按出现的帧数多少挑。只要主角就调小，速度更快。

## 效果和局限

我们在 RTX 4090 上的实测，素材是 sh020（iPhone 手持跟拍一个走路的人，画面里还有其他人，1080×1920、300 帧）：

| 方式 | 找到的物体 | 总耗时（含加载模型约 10 秒） | 每帧 | 显存峰值 |
|---|---|---|---|---|
| 提示词 person | 7 个人 | 80 秒 | 约 0.23 秒 | 约 6.1 GB |
| 接人物框（2 个人） | 2 个人 | 65 秒 | 约 0.12 秒 | 约 5.1 GB |

已知问题：

- 模型内部把每帧压成 1008×1008 来算，遮罩再放大回原尺寸，所以 4K 素材上的边缘是软的、不是像素级的。
- 长镜头会自动切成几段来算，每段长度按物体数定（8 个物体时一段约 800 帧），保证显存不会随镜头变长一直涨。用提示词时，段和段之间靠遮罩重叠来对编号，人交叉、长时间被挡住以后编号可能会换。用人物框时编号来自人物框，比较稳。
- 没有用更新的 SAM 3.1：它要求只有 H100 这类 Hopper 架构显卡才能用的 FlashAttention 3，而且不能按物体给框。

## 团队

Meta 超级智能实验室（Meta Superintelligence Labs）出品。Segment Anything 系列前两代也是 Meta 做的：SAM（2023）实现了「点一下就分割任意物体」，SAM 2（2024，Meta FAIR）加上了视频跟踪，很多 AI Roto 工具就是在它们上面做的。SAM 3 这一代又加上了用文字（概念）找物体。Meta 超级智能实验室同期还发布了 SAM 3D（SAM 3D Objects、SAM 3D Body）。

## 模型下载和安装

- **需要先申请权限（Hugging Face gated）**：
  1. 登录 Hugging Face，打开 https://huggingface.co/facebook/sam3 ；
  2. 按页面提示填写表单（同意共享联系信息）并提交，等 Meta 审批；
  3. 在这台机器上登录 Hugging Face：`uv run hf auth login`，粘贴一个有 Read 权限的 Access Token；
  4. 审批通过后运行安装。审批前运行安装，会提示权重「需要申请权限」，批下来以后再运行一次就行。
- **自动安装**：运行 `uv run lab2shot ext install sam3`，会下载：
  - SAM 3 权重 `sam3.pt`（检测和跟踪在同一个文件里，约 3.4 GB）和它的 LICENSE 文件；
  - 一个独立的 Python 环境（含 PyTorch，约 7 GB），不用编译任何东西。
  - 我们这台机器上装完一共用了约 3 分钟。

## 许可证说明

- **代码和权重都是 SAM License，可以商用**，可以修改和再分发；再分发时要附上许可证原文，拿它做的东西也要用同样的条款。
- 用它做研究发论文，要注明用了 SAM 3。
- **禁止用途**：军事和战争、核工业、间谍活动、枪支和非法武器，以及受美国《国际武器贸易条例》（ITAR）管制的用途；使用时要遵守美国等国家的出口管制和制裁规定。
- 不能逆向工程、反编译模型。违反条款时 Meta 可以终止许可，终止后要删除并停止使用；对 Meta 就这些材料提起专利诉讼，专利许可也会终止。软件不提供任何担保。

## 参考

- 论文：https://arxiv.org/abs/2511.16719
- Meta 论文页：https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/
- 项目主页：https://ai.meta.com/sam3
- 发布博客（含后续的 SAM 3.1）：https://ai.meta.com/blog/segment-anything-model-3/
- 代码仓库：https://github.com/facebookresearch/sam3
- 许可证原文：https://github.com/facebookresearch/sam3/blob/main/LICENSE
- 模型卡（本扩展用的 SAM 3）：https://huggingface.co/facebook/sam3
- 模型卡（SAM 3.1，本扩展没用）：https://huggingface.co/facebook/sam3.1
- 前代 SAM 2：https://github.com/facebookresearch/sam2
- 初代 SAM：https://github.com/facebookresearch/segment-anything
