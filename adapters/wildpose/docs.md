+++
team = "Stanford Gradient Spaces"
people = "Jianhao Zheng, Liyuan Zhu, Zihan Zhu, Iro Armeni"
paper = "WildPose: A Unified Framework for Robust Pose Estimation in the Wild"
paper_url = "https://arxiv.org/abs/2605.12774"
repo = "https://github.com/GradientSpaces/WildPose"
year = 2026
+++

使用官方 SLAM 跟踪、束调整和 trajectory_filler。适配器将项目画面接为 Dataset，已知镜头参数对应官方 benchmark 的内参输入。
相机覆盖所有输入帧；官方保存的优化深度只覆盖关键帧。本节点保留这个区别，未优化帧不会填造深度。
RTX 5090 使用 CUDA 13 的 PyTorch 和工具链重新编译官方 CUDA 后端，安装自检通过才标记就绪。
