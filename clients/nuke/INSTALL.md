# Lab2Shot Nuke 插件：安装说明

支持 Nuke 17.0（Windows），使用 Nuke 17 的新 3D 系统（Camera4、GeoImport）。插件只用 Nuke 自带的 Python 和 Qt，不往 Nuke 里装任何东西，也不改 Nuke 安装目录。

## 安装

1. 解压这个压缩包，把里面的 `lab2shot` 文件夹整个复制到 `C:\Users\<你的用户名>\.nuke\`。
   复制完是这样：
   ```
   .nuke\lab2shot\init.py
   .nuke\lab2shot\menu.py
   .nuke\lab2shot\lab2shot_nuke\...
   ```
2. 用文本编辑器打开 `.nuke\init.py`（没有就新建一个），在末尾加上这四行：
   ```python
   # >>> Lab2Shot (managed block: delete from this line to the <<< line to uninstall)
   import nuke
   nuke.pluginAddPath('./lab2shot')
   # <<< Lab2Shot
   ```
3. 重启 Nuke。菜单栏出现「Lab2Shot」就装好了。

升级：用新的 `lab2shot` 文件夹覆盖旧的，重启 Nuke。卸载：删掉 `.nuke\lab2shot` 文件夹和 `init.py` 里这四行，重启 Nuke。脚本里已有的 Lab2Shot 节点和结果节点会留着；没装插件时它们只是普通节点，不影响打开脚本。

## 第一次使用

1. 菜单「Lab2Shot → 登录…」，填服务器地址（管理员告诉你的网址）、账号、密码。还没有账号？请到网页端注册。
2. 「Lab2Shot → 新建节点」：脚本里多一个 Lab2Shot 节点（`lab2shot1`，一个 NoOp），并打开面板。
   面板停靠在属性面板旁的一个标签页里；关掉以后从菜单「Lab2Shot → 打开面板」或窗格菜单（Pane）里的「Lab2Shot」再打开，保存工作区后下次启动会恢复。
3. 在面板里选工具，给每个输入在节点图里选中节点（画面选 Read，摄影机选 Camera），点「用选中的」，或者「选文件」。
   在节点图里选中一个 Lab2Shot 节点，面板会自动切到它。摄影机以 USD 交给工具，所以绑在「摄影机（USD）」那个输入上。
   Read 的帧偏移（frame offset / start at）会被读出来，结果按同样的帧对回 Nuke 的帧。
4. 点「计算」。计算在服务器上进行，Nuke 照常可用；随时可以「取消」。
5. 算完自动导入，结果是新的节点，放在 Lab2Shot 节点右边的一个 Backdrop（`lab2shot1_v001`）里，节点名以 `l2s_lab2shot1_v001_` 开头：
   摄影机是 Camera4，点云和模型是 GeoImport，图像（含多层 EXR、深度）是 Read，ST map 另建 STMap（去畸变的 STMap 的 src 接你绑定的画面 Read）。
   摄影机和点云在同一个空间里（单位都是厘米），直接对得上。插件不改、不删、不移动脚本里原有的任何节点，也不改工程设置。
6. 结果文件在脚本所在文件夹的 `data\lab2shot\<节点名>\v001\` 下（设了工程目录就在工程目录下）。**脚本没保存时不能计算**：「计算」按钮是灰的，提示「请先保存工程」；保存一次脚本就可以了。
7. 版本卡片点一下：节点图缩放到这个版本的 Backdrop（不会改动你的选择）。
