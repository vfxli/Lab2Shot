+++
team = "ETH Zurich Computer Vision and Geometry Group"
people = "Zador Pataki, Paul-Edouard Sarlin, Marc Pollefeys"
paper = "VidMap: Exploiting Temporal Structure for Video-Based Structure-from-Motion"
repo = "https://github.com/cvg/vidmap"
year = 2026
+++

直接使用官方视频前端与原生 COLMAP 全局建图。输入画面，输出相机及稀疏点云，未注册帧不会被当作真实求解结果。
已知焦距通过官方 calibrated 配置和 intrinsics.json 输入；未知焦距使用官方 GeoCalib/DA3 校准。平滑轨迹对应官方实验配置。
模型前端使用 GPU，几何求解先使用 CPU。COLMAP 4.2 与 VidMap 原生扩展在独立环境中编译，自检失败不会启用。
