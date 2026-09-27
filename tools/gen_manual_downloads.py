"""Generate docs/manual-downloads.md from the extension declarations.

The document lists every file the installer cannot fetch by itself (items of lab2shot.extensions.manual) and every
weight whose Hugging Face repository requires an approved access request. Run after changing either:

    uv run python tools/gen_manual_downloads.py
"""

from __future__ import annotations

from pathlib import Path

from lab2shot.extensions import extensions, manual

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "manual-downloads.md"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render() -> str:
    needs = manual.needed_by()
    lines = [
        "# 需自行下载的文件",
        "",
        "<!-- 本文件由 tools/gen_manual_downloads.py 依据各扩展包的声明生成，请勿手工修改。 -->",
        "",
        "扩展包所需的代码与绝大多数模型权重由安装器自动下载，无需任何操作。以下两类文件除外：",
        "",
        "1. **须自行下载的文件**：其网站要求注册、登录或在浏览器中同意许可协议，安装器无法代为获取。",
        "2. **须申请访问的 Hugging Face 权重**：其仓库须先申请访问，获批后本机须登录 Hugging Face，安装器方可下载。",
        "",
        "仅在使用相应扩展包时才需要这些文件。`./setup.sh` 的「安装与环境」菜单中，「手动下载」与「Hugging Face 登录」两项"
        "可随时查看每一项的当前状态。",
        "",
        "## 一、须自行下载的文件",
        "",
        "### 操作步骤",
        "",
        "1. 打开下表中的下载页面，按页面要求注册或登录，下载表中所列的那一项。",
        "2. 将下载得到的文件**原样**放入 Lab2Shot 目录下的 `downloads/` 文件夹：无需解压，无需改名。",
        "3. 执行 `./setup.sh downloads`（或在管理后台「扩展包 → 手动下载」中点击「重新检查」）。Lab2Shot 依据文件内容识别每一个文件，"
        "安装到对应位置，并将原始文件移入 `downloads/installed/`（可删除，也可保留作为备份）。",
        "4. 对于须同意许可协议的项目（如 Autodesk FBX SDK），请在管理后台「扩展包 → 手动下载」中阅读许可协议并选择「同意并安装」。",
        "",
        "人体与面部模型（SMPL、SMPL-X、MANO、FLAME）全机只保存一份，存放于 `third_party/_body_models/`，由所有需要它的扩展包共用；"
        "其余文件安装到 `third_party/` 下对应的目录中。",
        "",
    ]
    for key, item in manual.items().items():
        users = "、".join(e["title"] for e in needs.get(key, [])) or "—"
        lines += [
            f"### {item.title}",
            "",
            "| 项目 | 内容 |",
            "| --- | --- |",
            f"| 用途 | {_cell(item.what)} |",
            f"| 所需扩展包 | {_cell(users)} |",
            f"| 下载页面 | {item.page} |",
            f"| 下载哪一项 | {_cell(item.download)} |",
            f"| 下载得到的文件 | `{item.filename}` |",
            f"| 许可与说明 | {_cell(item.note)} |",
            "",
        ]
    lines += [
        "## 二、须申请访问的 Hugging Face 权重",
        "",
        "### 操作步骤",
        "",
        "1. 注册 Hugging Face 账号，打开下表中的页面，按页面要求填写信息并同意许可条款，等待获批"
        "（多数仓库即时通过，部分须人工审核）。",
        "2. 在 https://huggingface.co/settings/tokens 创建一个访问令牌，权限选择 Read。",
        "3. 执行 `./setup.sh hf-login`，按提示粘贴令牌；或在启动服务前设置环境变量 `HF_TOKEN`。",
        "4. 在管理后台安装或重新安装相应的扩展包。",
        "",
        "| 扩展包 | 权重 | 申请访问的页面 |",
        "| --- | --- | --- |",
    ]
    for ext in sorted(extensions().values(), key=lambda e: e.title.lower()):
        for w in ext.weights:
            if w.gated:
                lines.append(f"| {_cell(ext.title)} | {w.key} | {w.page} |")
    lines += [
        "",
        "## 三、网络",
        "",
        "若服务器无法直接访问 Hugging Face、PyPI 或 GitHub，可在 `./setup.sh kernel` 的「镜像设置」中配置镜像地址，"
        "并在「重试设置」中调整下载失败时的重试策略。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
