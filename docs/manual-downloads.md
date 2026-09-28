# 需自行下载的文件

<!-- 本文件由 tools/gen_manual_downloads.py 依据各扩展包的声明生成，请勿手工修改。 -->

扩展包所需的代码与绝大多数模型权重由安装器自动下载，无需任何操作。以下两类文件除外：

1. **须自行下载的文件**：其网站要求注册、登录或在浏览器中同意许可协议，安装器无法代为获取。
2. **须申请访问的 Hugging Face 权重**：其仓库须先申请访问，获批后本机须登录 Hugging Face，安装器方可下载。

仅在使用相应扩展包时才需要这些文件。`./setup.sh` 的「安装与环境」菜单中，「安装手动下载的文件」与「登录 Hugging Face」两项可随时查看每一项的当前状态。

## 一、须自行下载的文件

### 操作步骤

1. 打开下表中的下载页面，按页面要求注册或登录，下载表中所列的那一项。
2. 将下载得到的文件**原样**放入 Lab2Shot 目录下的 `downloads/` 文件夹：无需解压，无需改名。
3. 执行 `./setup.sh downloads`（或在管理后台「扩展包 → 手动下载」中点击「重新检查」）。Lab2Shot 依据文件内容识别每一个文件，安装到对应位置，并将原始文件移入 `downloads/installed/`（可删除，也可保留作为备份）。
4. 对于须同意许可协议的项目（如 Autodesk FBX SDK），请在管理后台「扩展包 → 手动下载」中阅读许可协议并选择「同意并安装」，或执行 `./setup.sh downloads`，在终端阅读许可协议全文后同意。

人体与面部模型（SMPL、SMPL-X、MANO、FLAME）全机只保存一份，存放于 `third_party/_body_models/`，由所有需要它的扩展包共用；其余文件安装到 `third_party/` 下对应的目录中。

### SMPL-X

| 项目 | 内容 |
| --- | --- |
| 用途 | 身体 + 手 + 脸模型 |
| 所需扩展包 | GVHMR |
| 下载页面 | https://smpl-x.is.tue.mpg.de |
| 下载哪一项 | SMPL-X v1.1（NPZ+PKL） |
| 下载得到的文件 | `models_smplx_v1_1.zip` |
| 许可与说明 | 要先在官网注册登录，仅限非商用科研，禁止再分发 |

### SMPL

| 项目 | 内容 |
| --- | --- |
| 用途 | 身体模型 |
| 所需扩展包 | TRAM、WHAM |
| 下载页面 | https://smpl.is.tue.mpg.de |
| 下载哪一项 | SMPL for Python users 1.1.0 版 |
| 下载得到的文件 | `SMPL_python_v.1.1.0.zip` |
| 许可与说明 | 要先在官网注册登录，仅限非商用科研，禁止再分发 |

### MANO

| 项目 | 内容 |
| --- | --- |
| 用途 | 手部模型 |
| 所需扩展包 | HaMeR |
| 下载页面 | https://mano.is.tue.mpg.de |
| 下载哪一项 | Models & Code |
| 下载得到的文件 | `mano_v1_2.zip` |
| 许可与说明 | 要先在官网注册登录，仅限非商用科研，禁止再分发 |

### FLAME

| 项目 | 内容 |
| --- | --- |
| 用途 | 面部模型 |
| 所需扩展包 | Pixel3DMM、SMIRK |
| 下载页面 | https://flame.is.tue.mpg.de |
| 下载哪一项 | FLAME 2020 |
| 下载得到的文件 | `FLAME2020.zip` |
| 许可与说明 | 要先在官网注册登录，仅限非商用科研，禁止再分发 |

### Autodesk FBX SDK

| 项目 | 内容 |
| --- | --- |
| 用途 | 读写 FBX 文件的开发库 |
| 所需扩展包 | Autodesk FBX SDK |
| 下载页面 | https://aps.autodesk.com/developer/overview/fbx-sdk |
| 下载哪一项 | FBX SDK 的 Linux 版（gcc） |
| 下载得到的文件 | `fbx2020310_fbxsdk_gcc_linux.tar.gz` |
| 许可与说明 | 下载前后都需要同意 Autodesk 的许可协议 |

### StableMotion 权重

| 项目 | 内容 |
| --- | --- |
| 用途 | 动捕清理模型 StableMotion-BrokenAMASS（270 MB 的压缩包） |
| 所需扩展包 | StableMotion |
| 下载页面 | https://github.com/Murrol/StableMotion#pretrained-checkpoint-stablemotion-brokenamass |
| 下载哪一项 | README 里「Pretrained Checkpoint」那一节的 OneDrive 链接 |
| 下载得到的文件 | `stablemotion_ckpt_seed3407.tar.gz` |
| 许可与说明 | 作者把权重放在 OneDrive 网盘，要浏览器点过才给文件，程序下不了；权重是在 AMASS 上训练的，只许学术研究 |

## 二、须申请访问的 Hugging Face 权重

### 操作步骤

1. 注册 Hugging Face 账号，打开下表中的页面，按页面要求填写信息并同意许可条款，等待获批（多数仓库即时通过，部分须人工审核）。
2. 在 https://huggingface.co/settings/tokens 创建一个访问令牌，权限选择 Read。
3. 执行 `./setup.sh hf-login`，按提示粘贴令牌；或在启动服务前设置环境变量 `HF_TOKEN`。
4. 在管理后台安装或重新安装相应的扩展包。

| 扩展包 | 权重 | 申请访问的页面 |
| --- | --- | --- |
| Fast SAM 3D Body | sam-3d-body-dinov3 | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| Fast SAM 3D Body | mhr-model | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| Fast SAM 3D Body | sam-license | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| Kimodo | smplx-rp-v1 | https://huggingface.co/nvidia/Kimodo-SMPLX-RP-v1 |
| Kimodo | llama3-8b-instruct | https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct |
| Meta VGGT | VGGT-1B-Commercial | https://huggingface.co/facebook/VGGT-1B-Commercial |
| SAM 3 | sam3 | https://huggingface.co/facebook/sam3 |
| SAM 3D Body | sam-3d-body-dinov3 | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| SAM 3D Body | mhr-model | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| SAM 3D Body | sam-license | https://huggingface.co/facebook/sam-3d-body-dinov3 |
| SAM 3D Body | sam-3d-body-config | https://huggingface.co/facebook/sam-3d-body-dinov3 |

## 三、网络

若服务器无法直接访问 Hugging Face、PyPI 或 GitHub，可执行 `./setup.sh mirrors`（「安装与环境 → 扩展包编译与下载设置 → 设置下载镜像」）配置镜像地址，并执行 `./setup.sh retries`（同一菜单的「设置下载重试」）调整下载失败时的重试策略。
