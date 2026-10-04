# Open3D 点云转网格

year = 2024

## 这是什么

Open3D（isl-org/Open3D，MIT）的经典表面重建：把一片点云转成三角网格，三种方法共用一个节点。

- **Poisson**（`create_from_point_cloud_poisson`）：求解指示函数后提取等值面，输出水密、光滑的网格；`depth` 是八叉树深度（越大越细、越慢）。输入点云须带法线（法线取向影响重建质量）。
- **Ball Pivoting**（`create_from_point_cloud_ball_pivoting`）：让一个球在点云表面滚动，碰到三个点就成一个三角形；半径越大越能跨过稀疏处、细节越少。本节点用 `[r, 2r, 4r]` 三档半径滚球。
- **Alpha Shape**（`create_from_point_cloud_alpha_shape`）：三维 Delaunay 四面体化的 Alpha 复形；Alpha 值越大越接近凸包，越小越贴点、越容易出洞。

三种方法都要求点云带法线，worker 先用 `estimate_normals` + `orient_normals_consistent_tangent_plane` 按官方用法补上（法线半径按点云自身平均最近邻距离的 3 倍取）。

## 数据契约

- 输入：`scene.points`（厘米，Y 上）。核心侧 `scene_arrays` 把所有点云抽成一份 `points.npz` 交给 worker。
- 输出：`scene.model`（USD 网格，静态，厘米）。worker 写 `mesh.npz`（vertices / triangles），核心侧 `write_mesh` 转成 USD。

## 成本与资源

- CPU（Open3D 的重建都在 CPU），不占 GPU。
- 内存随点数和 Poisson 八叉树深度增长；`ram_gb` 以实测为准（见下）。

## 实测

ETH3D table 3（200 帧真值深度经「点云融合」TSDF，43.6 万点，体素 1 cm），Open3D 0.19.0，CPU：

- **Poisson**（depth 8）：130 180 顶点 / 260 280 三角形，23.2 s，峰值内存 1.6 GB。
- **Ball Pivoting**（半径 [1, 2, 4] cm）：18.4 万顶点 / 27.5 万三角形（20 万点抽样子集上测）。
- **Alpha Shape**（alpha 2 cm）：17.9 万顶点 / 41.8 万三角形（20 万点抽样子集上测）。

三种方法都写出 PLY 并回读成功。`ram_gb=2.0` 按 Poisson 峰值取整。
