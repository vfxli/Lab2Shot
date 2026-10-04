"""Nodes provided by the Open3D extension (MIT): 点云转网格。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from lab2shot.sdk import Cost, Invalid, Msg, NodeParams, Official, P, Param, Port, WorkerNode


class Mesh(WorkerNode):
    id = "open3d.mesh_from_points"
    version = 3  # 结果变化时递增（2：Poisson 按官方裁掉低密度顶点；半径 / Alpha 留空按点间距；3：Poisson 再按到点云的距离裁外插假面、网格带顶点色）
    category = "geometry_tools"
    runtime = "open3d"
    streams = False  # 整段一次重建：点都到齐才有网格
    learned = False  # 经典的网格重建，不跑模型
    cost = Cost(whole=True, ram_gb=2.0)  # CPU；Poisson depth 8 在 43.6 万点上实测峰值 1.6 GB（docs.md）
    inputs = (Port("points", "scene.points"),)
    outputs = (Port("model", "scene.model"),)
    main = "model"
    # 上游三种官方重建函数的 pybind 声明（cpp/pybind/geometry/trianglemesh.cpp）：
    # create_from_point_cloud_poisson / _ball_pivoting / _alpha_shape，输入都是带法线的 PointCloud，
    # 输出都是 TriangleMesh；depth / radii / alpha 是各自的参数。
    official = Official(
        cite="third_party/open3d/repo/cpp/pybind/geometry/trianglemesh.cpp:312-353",
        takes={"points": "pcd", "depth": "depth", "radius": "radii", "alpha": "alpha"},
        gives={"model": "TriangleMesh"},
    )

    class Params(NodeParams):
        method: Literal["poisson", "ball", "alpha"] = P("poisson", group="reconstruction")
        depth: int = P(8, ge=1, le=12, group="reconstruction", applies=Param("method").one_of("poisson"))
        # 留空 = 按点云自己的平均点间距取（worker BALL_NN / ALPHA_NN）：相对尺度的点云（VidMap、COLMAP）单位不定，
        # 固定的厘米值可能一个三角形都滚不出来
        radius: float | None = P(None, unit="cm", gt=0, group="reconstruction", applies=Param("method").one_of("ball"))
        alpha: float | None = P(None, unit="cm", gt=0, group="reconstruction", applies=Param("method").one_of("alpha"))
        # Poisson 在开放场景会外插假面：离输入点云超过「这么多倍点间距」（每点最近邻距离的中位数）的顶点删掉；0 = 不裁
        trim_distance: float = P(3.0, ge=0, le=100, group="reconstruction", applies=Param("method").one_of("poisson"))

    @classmethod
    def prepare(cls, ctx):
        from lab2shot.sdk import scene_arrays

        src = ctx.input("points")
        arrays = scene_arrays(src, samples=True)
        items = arrays.items["points"]
        pts = [np.asarray(item["points"], np.float64) for item in items]
        if not pts or not sum(len(p) for p in pts):
            raise Invalid(Msg("E-OPEN3D-NOPOINTS"))
        xyz = np.concatenate(pts).astype(np.float32)
        cols = [np.asarray(item.get("colors") if item.get("colors") is not None else [], np.float32).reshape(-1, 3)
                for item in items]
        saved = {"points": xyz}
        if all(len(c) == len(p) for c, p in zip(cols, pts)):
            rgb = np.concatenate(cols)
            # 读点云时没有颜色的点被填成 0.7 灰：全是这个灰就当没有颜色
            if len(rgb) and not np.allclose(rgb, 0.7, atol=1e-4):
                saved["colors"] = rgb
        np.savez_compressed(ctx.work / "points.npz", **saved)
        from lab2shot.sdk import Job

        return Job(None, inputs={"points": ctx.work / "points.npz"})

    @classmethod
    def convert(cls, ctx, raw, job):
        import numpy as np

        from lab2shot.sdk import SCENE_FILE, create_stage, save_stage, scene_packet, write_mesh

        data = raw.arrays("mesh.npz")
        verts = np.asarray(data["vertices"], np.float32)
        tris = np.asarray(data["triangles"], np.int32)
        stage = create_stage([])
        mesh = write_mesh(stage, "/shot/mesh", verts, tris)
        if "colors" in data:
            from pxr import UsdGeom, Vt

            rgb = np.ascontiguousarray(np.clip(np.asarray(data["colors"], np.float32), 0.0, 1.0))
            if len(rgb) == len(verts):
                mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(rgb))
        save_stage(stage, ctx.outputs["model"] / SCENE_FILE)
        result = raw.result()
        ctx.say("I-OPEN3D-RESULT", method=result.get("method", ""), vertices=int(len(verts)),
                triangles=int(len(tris)))
        return {"model": scene_packet(ctx.outputs["model"], [], "scene.model")}


NODES = (Mesh,)
