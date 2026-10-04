#!/usr/bin/env bash
# Lab2Shot setup entry. This script runs nothing on its own: it lists the steps and runs the one the user picks.
#
#   ./setup.sh              # choose the language, then the menu
#   ./setup.sh <step>       # run one step directly (names: ./setup.sh --help; e.g. wizard, update, env, web, check,
#                           # downloads, password, start, db-backup, db-restore), in the language chosen last time
#
# The first thing it does is ask for its language (中文 / English); it does not read LANG, LC_ALL or LAB2SHOT_LANG
# (user's rule, 2026-10-03: setup.sh manages its own language). The choice is kept in config/setup.lang and passed on
# to the menu (`lab2shot --lang <it> setup`). Its own words are below, both languages side by side: it runs before
# Python exists, so it cannot read the catalogues in lab2shot/i18n.
# The menu is `lab2shot setup` (Python) and needs uv and the Python environment; this script only guides those two,
# each on the user's choice, nothing installed unasked. Entering the menu uses `uv run --no-sync`, so it never syncs
# the environment over the network (that is 「安装与环境 → 安装 Python 依赖」 / "Install & Environment → Install
# Python dependencies" in the menu).
set -e
cd "$(dirname "$0")"

PYPI_MIRROR=${LAB2SHOT_PYPI_MIRROR:-https://pypi.tuna.tsinghua.edu.cn/simple}
# where uv downloads Python interpreters (python-build-standalone, hosted on GitHub); set a mirror when GitHub is out of reach
[ -n "${LAB2SHOT_PYTHON_MIRROR:-}" ] && export UV_PYTHON_INSTALL_MIRROR="$LAB2SHOT_PYTHON_MIRROR"

LANG_FILE=config/setup.lang
SETUP_LANG=$(cat "$LANG_FILE" 2>/dev/null || true)
case "$SETUP_LANG" in zh|en) ;; *) SETUP_LANG=zh ;; esac

choose_lang() {
  local default=1
  [ "$SETUP_LANG" = en ] && default=2
  echo "Lab2Shot setup"
  echo "  1  中文"  # both languages
  echo "  2  English"
  read -r -p "选择语言 / Choose a language [$default]: " pick  # both languages
  case "${pick:-$default}" in
    2) SETUP_LANG=en ;;
    *) SETUP_LANG=zh ;;
  esac
  mkdir -p config
  echo "$SETUP_LANG" > "$LANG_FILE"
}

# say <zh> <en>: the line in the chosen language
say() { if [ "$SETUP_LANG" = en ]; then echo "$2"; else echo "$1"; fi; }
# ask <var> <zh prompt> <en prompt>
ask() { local p; p=$(say "$2" "$3"); read -r -p "$p" "$1"; }

pip_uv() {
  # PEP 668 (Debian 12 / Ubuntu 23.04 and later system Python) refuses `pip install --user` (externally-managed-
  # environment). uv is a standalone executable installed in the user's directory without touching system packages,
  # so adding --break-system-packages is safe here.
  for index in "" "-i $PYPI_MIRROR"; do
    # shellcheck disable=SC2086
    python3 -m pip install --user $index uv && return 0
    # shellcheck disable=SC2086
    python3 -m pip install --user --break-system-packages $index uv && return 0
  done
  return 1
}

install_uv() {
  say "  1  使用官方安装脚本（astral.sh，经由 GitHub 下载）" "  1  Use the official install script (astral.sh, downloads from GitHub)"
  say "  2  使用 pip 安装（python3 -m pip install --user uv；失败时改用镜像 $PYPI_MIRROR）" \
      "  2  Install with pip (python3 -m pip install --user uv; falls back to the mirror $PYPI_MIRROR)"
  say "  0  退出（手动安装请参阅：https://docs.astral.sh/uv/）" "  0  Quit (to install by hand see https://docs.astral.sh/uv/)"
  ask pick "请选择一项 [0]: " "Choose one [0]: "
  case "$pick" in
    1) curl -LsSf https://astral.sh/uv/install.sh | sh || {
         say "官方安装脚本执行失败（通常是由于无法访问 GitHub）。请重新运行并选择 2，通过 PyPI 安装。" \
             "The official install script failed (usually because GitHub is out of reach). Run again and choose 2 to install from PyPI."
         return 1; } ;;
    2) pip_uv || { say "pip 安装失败，请查看上方输出。" "pip install failed; see the output above."; return 1; } ;;
    *) exit 0 ;;
  esac
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv > /dev/null 2>&1 || {
    say "uv 已安装，但尚未加入 PATH。请打开新终端后重新运行 ./setup.sh。" \
        "uv is installed but not on PATH yet. Open a new terminal and run ./setup.sh again."
    exit 1; }
}

sync_env() {
  echo "  1  uv sync"
  say "  0  退出" "  0  Quit"
  ask pick "请选择一项 [0]: " "Choose one [0]: "
  [ "$pick" = 1 ] || exit 0
  if uv sync; then return 0; fi
  say "uv sync 执行失败。" "uv sync failed."
  say "（系统中没有 Python 3.13 时，uv 需要从 GitHub 下载解释器。如无法访问 GitHub，请设置镜像后重试：LAB2SHOT_PYTHON_MIRROR=https://<镜像>/astral-sh/python-build-standalone/releases/download ./setup.sh）" \
      "(Without Python 3.13 on the system, uv downloads an interpreter from GitHub. If GitHub is out of reach, set a mirror and try again: LAB2SHOT_PYTHON_MIRROR=https://<mirror>/astral-sh/python-build-standalone/releases/download ./setup.sh)"
  ask again "是否使用 PyPI 镜像重试？（$PYPI_MIRROR） [Y/n]: " "Try again with the PyPI mirror? ($PYPI_MIRROR) [Y/n]: "
  case "$again" in
    n|N) return 1 ;;
    *) UV_INDEX_URL="$PYPI_MIRROR" uv sync || {
         say "重试仍然失败。如网络不可用，请配置代理或更换镜像（LAB2SHOT_PYPI_MIRROR=… ./setup.sh）。" \
             "The retry failed too. If the network is unavailable, configure a proxy or another mirror (LAB2SHOT_PYPI_MIRROR=… ./setup.sh)."
         return 1; } ;;
  esac
}

# the language first, when the menu is entered; a step named on the command line uses the language chosen last time
[ $# -eq 0 ] && choose_lang

if ! command -v uv > /dev/null 2>&1; then
  say "Lab2Shot 配置 — 步骤 1：安装 uv（未检测到）" "Lab2Shot setup — step 1: install uv (not found)"
  say "  操作      安装 uv（Python 环境管理工具）" "  Action    install uv (the Python environment manager)"
  say "  用途      创建环境、运行服务及配置菜单均依赖 uv" "  Purpose   creating the environment, running the service and this menu all need uv"
  say "  写入位置  ~/.local/bin/uv 或 pip 用户目录；不修改系统 Python" "  Writes    ~/.local/bin/uv or pip's user directory; the system Python is left alone"
  say "  影响范围  用户目录（唯一写入项目目录之外的步骤）" "  Scope     your user directory (the only step that writes outside the project)"
  install_uv || exit 1
fi

if [ ! -x .venv/bin/python ]; then
  say "Lab2Shot 配置 — 步骤 2：Python 环境（未检测到 .venv）" "Lab2Shot setup — step 2: Python environment (.venv not found)"
  say "  操作      执行 uv sync，按 uv.lock 安装依赖" "  Action    run uv sync, installing the dependencies pinned in uv.lock"
  say "  用途      提供服务、命令行及配置菜单的运行环境" "  Purpose   the environment the service, the command line and this menu run in"
  say "  写入位置  .venv/；包缓存位于 ~/.cache/uv" "  Writes    .venv/; the package cache is ~/.cache/uv"
  say "  影响范围  项目目录内" "  Scope     inside the project directory"
  sync_env || exit 1
fi

# The menu from here on is Python (web, extension builds and downloads, administrator, server, start …); each item runs
# only when chosen. --no-sync: `uv run` would otherwise sync the environment over the network first; syncing is the
# menu item 「安装与环境 → 安装 Python 依赖」, run only when chosen. The menu speaks the language chosen above.
exec uv run --no-sync lab2shot --lang "$SETUP_LANG" setup "$@"
