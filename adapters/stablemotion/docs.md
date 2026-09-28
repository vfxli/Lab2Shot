+++
team = "Simon Fraser University + Electronic Arts（EA SEED / Bones Studio）+ NRC Canada"
people = "Yuxuan Mu, Hung Yu Ling, Yi Shi, Ismael Baira Ojeda, Pengcheng Xi, Chang Shu, Fabio Zinno, Xue Bin Peng"
paper = "StableMotion: Training Motion Cleanup Models with Unpaired Corrupted Data（SIGGRAPH Asia 2025）"
paper_url = "https://arxiv.org/abs/2505.03154"
website = "https://yxmu.foo/stablemotion-page/"
repo = "https://github.com/Murrol/StableMotion"
year = 2025
+++

## 这是什么

StableMotion 是一套训练动捕清理模型的方法：动捕数据常因为传感器不准和后期处理带上刺眼的瑕疵，人工清理既贵又慢；以往的数据驱动做法要成对的「坏 — 干净」训练数据，而这种成对数据本身就得先靠人工清理才有。StableMotion 不要成对数据，作者引入逐帧的动作质量标记（人工标，或者用启发式算法标），在好坏混在一起的原始动捕数据上训练一个质量感知的动作生成模型；用一个扩散框架实现，得到一个既能判别又能生成的模型，既找得出坏掉的帧，也修得好。

在 Lab2Shot 里，我们拿它清理动捕：「动画」口接骨架动画或者蒙皮角色，先重定向到 SMPL 的 24 关节骨架、重采样到模型的 20 fps 再送进去，交出修好的动画和一条逐帧的「问题帧」曲线。「只改问题帧」默认打开，模型判为好的帧一帧不动。

## 输入输出

**官方要什么、给什么**（上游自己的话）

- **吃**：`python -m sample.fix_globsmpl --testdata_dir …`，一批 AMASS 格式的 SMPL 动作
  （20 fps，经它自己的 corrupting / canonicalise 预处理）；`--ensemble` / `--enable_sits` 是它的增强计算方式。
- **给**：修好的动作，外加每帧的质量标记——模型的第 233 个通道就是「这一帧坏没坏」
  （`sample/fix_globsmpl.py` 的 `detect_labels` 读它，`out['label']` 再喂给同一文件的 `fix_motion`）。

**我们怎么接的**

- 「动画」口（骨架动画或蒙皮角色）先重定向到 SMPL 的 24 关节骨架、重采样到模型的 20 fps，
  再转成上游自己的 232 维表示（往返误差 0.5 µm）。
- 「问题帧」= 上游那条 `label`；「动画」输出 = 上游修好的动作，「只改问题帧」默认打开，模型说好的帧一帧不动。
- 「质量」参数 = 上游 README 那两条命令：普通路径，和它的 ensemble 选择（`--ensemble`）。
- **不一样的两点**：① 判定整段一次过，但**修复按 100 帧一窗**（它训练的长度；94 cm 的根节点跳变在 100–200 帧的窗口里能修回 1.5 cm，
  400 帧 6.6 cm，800 帧 33 cm），每窗各自做规范化再还原；② 上游吃的是数据集文件夹，我们吃节点图上的动画，单位和朝向的换算在 worker 里做。

**出处**：简介来自论文摘要（arXiv 2505.03154：「The core component of our method is the introduction of motion quality indicators…」
「At test time, the model can be prompted to generate high-quality motions using the quality indicators.」
「…a unified motion generate-discriminate model, which can be used to both identify and fix corrupted frames.」）；
输入输出依据 `third_party/stablemotion/repo/README.md`、`repo/sample/fix_globsmpl.py` 和 `adapters/stablemotion/nodes.py`、`worker.py`。

## 在 Lab2Shot 里怎么用

- 典型接法：「导入 FBX」（动捕棚交来的 take）→ **StableMotion 动捕清理** →「FBX 输出设置」→「输出」。模板「动捕清理 · StableMotion」就是这套；问题帧那条曲线一起交出去（`broken_frames_ML_Lab2Shot_StableMotion`）。
- **只改问题帧**（默认开）：模型判好的帧，动画一帧不动，只重画判坏的那些，前后各 3 帧平滑过渡。这是官方自己的流程，也是动画师要的——干净的表演不该被模型重写一遍。想看模型眼里这段动作「应该」是什么样，就关掉它。
- **检测阈值**：先看「问题帧」这条曲线再调。调低多修一些（连轻微的抖动一起），调高只修最明显的。
- **质量**：「基本」是官方 README 的第一条命令（检测一遍、修一遍）；「增强」是第二条（多次采样定问题帧、多采几版按脚滑挑最好的一版、去噪时加脚锁引导），论文主结果用的就是增强档，慢很多。先用基本档，修不干净再换。
- **关节映射**：它的骨骼是 SMPL 的 22 个身体关节。接上人物后表格每行显示自动猜到的人物关节，猜错了改。手指、面部、扭转骨骼不参与，**保持原来的动画不动**。
- **帧率和长度**：模型按 20 帧/秒、一次 100 帧（5 秒）训练。节点自己换算帧率；判定整段一次过，修复只在判坏的帧那里各开一个 100 帧的窗口（坏帧落在窗口前四分之一处，模型对窗口末尾的判断最不可靠），窗口里只把判坏的帧写回去，好帧不动；没有等距分段，也不做交叉过渡，不用手动分段。
- **地面在 y = 0**：训练数据里人站在地面上。视频解出来的人物先接「自动落地」。
- **接完它再接脚滑**：它修姿势，不专门钉脚。脚滑交给「UnderPressure 脚滑清理」，模板「动捕清理 · StableMotion + UnderPressure」把两步接好了。

## 效果和局限

### RTX 4090 上的表现

素材：AMASS 里 8 段真实走路动捕（HDM05、CMU、ACCAD、BMLrub、KIT、TotalCapture、SFU、Eyes Japan，各取 5 秒、换算到 24 帧/秒），和 LAFAN1（Ubisoft 的光学动捕）的 `walk1_subject1`。

- **干净的动捕不会被乱改**：
  - LAFAN1 的 10 秒、30 秒、60 秒三段，**判坏 0 帧**，一个字节都没动；
  - AMASS 那 8 段里，6 段满 5 秒的判坏 0%–3%（合计 16/720 = 2.2%）；
  - 判好的帧**原样保留**（最大变动 0.00001 cm）——这是「只改问题帧」的保证。
- **不足 5 秒的镜头不可靠**：另外 2 段只有 2 秒（51 和 64 帧）的，判坏 70% 和 98%。模型一次看 100 帧（它自己的 20 帧/秒，5 秒），短于它就没法比较，节点会提醒一句。**接一整段动捕再切，别切好再清理。**
- **注入一次已知的错误**（左上臂转 150°，连续 6 帧）：3 段里 2 段判中了全部 6 帧并修回去，左臂误差 35.0 → 5.4 cm、27.0 → 3.9 cm；第 3 段没判出来。这和它的训练方式一致——它学的是抖动、过度平滑、脚滑、漂移这几类毛病，**一个保持不动的大幅错误姿势它认不出来**（用同一份权重跑官方推理脚本，结论一样）。
- **速度和显存**：60 秒的动捕 3.5 秒、215 MB 显存；10 秒的第一次 6.7 秒（含加载模型），之后模型常驻，30 秒的 1.5 秒。判坏的帧越多越慢（每 5 秒一段重画约 0.7 秒）。

### 局限

- 训练数据是 AMASS（日常动作、20 帧/秒）。**卡通、特技、夸张表演**它会大面积判成「有问题」然后按写实动捕重画——节点会在判坏帧超过六成时警告一句；
- **手指和面部不参与**（SMPL 的 22 个身体关节里没有）；
- **一个保持不动的错误姿势认不出来**（见上）：它认的是「前后帧之间不该有的变化」；
- 扩散模型每次重画结果不同，同一个种子才给同一个结果；
- 它判断的是「姿势像不像真的动捕」，不是「这是不是你要的表演」：动画师故意做的极端姿势也可能被判成坏帧，所以「检测阈值」和那条「问题帧」曲线要一起看；
- 官方自己说：要修自己动捕棚里特有的毛病，最好拿自己的数据重训一个模型（代码是 MIT，可以这么做）。

## 团队

西蒙弗雷泽大学（SFU）Xue Bin "Jason" Peng 的组和 Electronic Arts 一起做的：第一作者 Yuxuan Mu 是 SFU 的博士生，Hung Yu Ling 和 Fabio Zinno 来自 EA（EA 的动捕流程里本来就要处理这类脏数据），另外还有加拿大国家研究院的人。Xue Bin Peng 是 DeepMimic、ASE、AMP 这一系角色动画强化学习工作的作者。

## 模型下载和安装

- 自动安装：`lab2shot ext install stablemotion`。
- **权重要自己下载一次**（程序下不了）：作者把 `stablemotion_ckpt_seed3407.tar.gz`（270 MB）放在 OneDrive 网盘，要浏览器点过才给文件。到仓库 README 的「Pretrained Checkpoint」那一节点那个链接，下载好的文件**原样**放进 Lab2Shot 的 `downloads` 文件夹（不用解压、不用改名），后台管理页「扩展包」里的「手动下载」会认出它并装好。
- 模型的**归一化统计量**（`mean.pt` / `std.pt`）就在上游仓库里，随代码一起检出，不另外下载任何东西。
- 磁盘：仓库 13 MB + 权重 145 MB + 环境约 6 GB。

### 归一化统计量是从哪来的

扩散模型在归一化后的特征空间里工作，所以推理必须有训练时的每通道均值和标准差。它们就在上游仓库里、而且进了 git：
`third_party/stablemotion/repo/dataset/meta_AMASS_20.0_fps_nh_globsmpl_corrupted_cano/mean.pt`、`std.pt`（232 个数）。
代码直接读这两个文件（`Normalizer(repo)`），第 233 个标签通道的 0.5 按上游自己的 `add_label_channel` 补上；不下载别的文件，
文件不在时报 `E-STABLEMOTION-NOSTATS`（扩展包装坏了，重装）。

Hugging Face 上的 `KitsuMate/stablemotion-onnx`（同一份权重的第三方 ONNX 转换）也带了一份 JSON 格式的统计量，前 232 个数和上游的 `.pt` 逐位相同；不需要它，以上游仓库里的文件为准。

## 许可证说明

**仅限研究**。代码是 MIT，可以随便用；但放出来的权重 StableMotion-BrokenAMASS 是在 **AMASS** 上训练的，AMASS 的许可只允许非商业的学术研究，所以这个节点按更严的一档标（许可证有两档时按严的算）。

要用在商业镜头上，按官方 README 的说法只有一条路：拿自己（或自己有权使用）的动捕数据，用这份 MIT 代码训练一个自己的模型。

## 参考

- 论文：https://arxiv.org/abs/2505.03154
- 项目页：https://yxmu.foo/stablemotion-page/
- 仓库：https://github.com/Murrol/StableMotion
- 权重（OneDrive，要浏览器下载）：仓库 README 的「Pretrained Checkpoint: StableMotion-BrokenAMASS」
- 第三方 ONNX 转换（不需要，归一化统计量读上游仓库的 mean.pt / std.pt）：https://huggingface.co/KitsuMate/stablemotion-onnx
