#!/usr/bin/env bash
# Lab2Shot 配置入口。本脚本不执行任何自动操作：仅列出步骤，由使用者选择后执行对应步骤。
#
#   ./setup.sh              # 进入菜单
#   ./setup.sh <step>       # 直接执行指定步骤（步骤名见 ./setup.sh --help，例如 wizard、env、web、check、downloads、password、start、db-backup、db-restore）
#
# 菜单由 `lab2shot setup`（Python）实现，其运行依赖 uv 与 Python 环境；本脚本仅负责这两项的引导，
# 同样由使用者选择后执行，不自动安装。每一步失败时均提供替代方案（镜像或其他安装方式）。
# 仅执行使用者明确选择的步骤：进入菜单时使用 `uv run --no-sync`，不会联网同步环境（同步环境请选择菜单第 1 项）。
set -e
cd "$(dirname "$0")"

PYPI_MIRROR=${LAB2SHOT_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
# uv 下载 Python 解释器的地址（python-build-standalone 托管于 GitHub）；无法访问 GitHub 时应设置镜像
[ -n "${LAB2SHOT_PYTHON_MIRROR:-}" ] && export UV_PYTHON_INSTALL_MIRROR="$LAB2SHOT_PYTHON_MIRROR"

pip_uv() {
  # PEP 668（Debian 12 / Ubuntu 23.04 及更高版本的系统 Python）拒绝以 --user 方式安装包（externally-managed-environment）。
  # uv 为独立可执行文件，安装至用户目录不影响系统包，因此此时附加 --break-system-packages 是安全的。
  for index in "" "-i $PYPI_MIRROR"; do
    # shellcheck disable=SC2086
    python3 -m pip install --user $index uv && return 0
    # shellcheck disable=SC2086
    python3 -m pip install --user --break-system-packages $index uv && return 0
  done
  return 1
}

install_uv() {
  echo "  1  使用官方安装脚本（astral.sh，经由 GitHub 下载）"
  echo "  2  使用 pip 安装（python3 -m pip install --user uv；失败时改用镜像 $PYPI_MIRROR）"
  echo "  0  退出（手动安装请参阅：https://docs.astral.sh/uv/）"
  read -r -p "请选择一项 [0]: " pick
  case "$pick" in
    1) curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "官方安装脚本执行失败（通常是由于无法访问 GitHub）。请重新运行并选择 2，通过 PyPI 安装。"; return 1; } ;;
    2) pip_uv || { echo "pip 安装失败，请查看上方输出。"; return 1; } ;;
    *) exit 0 ;;
  esac
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv > /dev/null 2>&1 || { echo "uv 已安装，但尚未加入 PATH。请打开新终端后重新运行 ./setup.sh。"; exit 1; }
}

sync_env() {
  echo "  1  uv sync"
  echo "  0  退出"
  read -r -p "请选择一项 [0]: " pick
  [ "$pick" = 1 ] || exit 0
  if uv sync; then return 0; fi
  echo "uv sync 执行失败。"
  echo "（系统中没有 Python 3.13 时，uv 需要从 GitHub 下载解释器。如无法访问 GitHub，请设置镜像后重试：LAB2SHOT_PYTHON_MIRROR=https://<镜像>/astral-sh/python-build-standalone/releases/download ./setup.sh）"
  read -r -p "是否使用 PyPI 镜像重试？（$PYPI_MIRROR） [Y/n]: " again
  case "$again" in
    n|N) return 1 ;;
    *) UV_INDEX_URL="$PYPI_MIRROR" uv sync || { echo "重试仍然失败。如网络不可用，请配置代理或更换镜像（LAB2SHOT_PYPI_MIRROR=… ./setup.sh）。"; return 1; } ;;
  esac
}

if ! command -v uv > /dev/null 2>&1; then
  echo "Lab2Shot 配置 — 步骤 1：安装 uv（未检测到）"
  echo "  操作      安装 uv（Python 环境管理工具）"
  echo "  用途      创建环境、运行服务及配置菜单均依赖 uv"
  echo "  写入位置  ~/.local/bin/uv 或 pip 用户目录；不修改系统 Python"
  echo "  影响范围  用户目录（唯一写入项目目录之外的步骤）"
  install_uv || exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "Lab2Shot 配置 — 步骤 2：Python 环境（未检测到 .venv）"
  echo "  操作      执行 uv sync，按 uv.lock 安装依赖"
  echo "  用途      提供服务、命令行及配置菜单的运行环境"
  echo "  写入位置  .venv/；包缓存位于 ~/.cache/uv"
  echo "  影响范围  项目目录内"
  sync_env || exit 1
fi

# 此后的菜单由 Python 实现（网页、内核、管理员、服务器、启动等），每一项仅在使用者选择后执行。
# --no-sync：`uv run` 默认会先联网同步环境；同步为菜单第 1 项，仅在选择后执行。
exec uv run --no-sync lab2shot setup "$@"
