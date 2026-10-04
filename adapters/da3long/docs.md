+++
team = "ByteDance Seed"
paper = "Depth Anything 3: Recovering the Visual Space from Any Views"
paper_url = "https://arxiv.org/abs/2511.10647"
repo = "https://github.com/ByteDance-Seed/Depth-Anything-3"
year = 2025
+++

官方 DA3-Streaming 流水线。输入画面序列，输出联合对齐的逐帧相机、深度和置信度。
「每段最多帧数」与「重叠帧数」决定官方分块；回环使用 SALAD。长片深度与相机采用相同的累积 Sim3 尺度。
Giant + Metric-Large 权重按近似米制输出；实际尺度仍需用已知距离核对，可显式调整「尺度」。
单段短片调用官方同一个 process_single_chunk，补齐上游单段导出遗漏，不修改上游代码。
