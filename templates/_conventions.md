# 模板对外参数名约定

模板 `exposed` 树里每个条目的 `name` 是命令行（`lab2shot cook --set 名=值`）、DCC 插件和批量脚本使用的接口。规则：**一个对外名 = 一个含义**，所有模板遵守；`lab2shot check templates` 按本文件校验。

**通则**
1. 不带前缀的名字 = 这张卡**核心算法节点**的该参数（`model` `mode` `resolution` `seed` `steps` `threshold` `quality` `fp16` `max_frames` `step` `loops` `overlap` `point_step` `point_size` …，含义随卡的核心算法，但总是「这张卡的主算法的 X」）。
2. 辅助节点的参数加**角色前缀**：`rough_`（精细抠像的粗遮罩）、`probe_`（估光）、`deshake_`（去抖）、`dome_`（穹顶灯）、`vipe_`（ViPE 相机解算）、`depth_`（非核心的深度模型；ViPE 深度卡上深度虽是主交付，仍按 ViPE 那一路的前缀叫）、`moge_`（深度对齐卡上作为对齐参照的 MoGe；核心 VDA 按通则 1 不带前缀）、`detect_`（检人）、`people_`（选人）、`cam_`（相机来源 / 相机导入 / 相机空间转换）、`char_`（目标角色导入）、`feet_`（清理里的 UnderPressure 一段：钉脚）、`fg_`（边缘解混合）、`comp_`（预览合成）、`uv_`（UV 投射）、`solve_`（辅助的相机解算，如 GeoCalib 卡上的 Pi3）、`undist_`（去畸变）、`occlusion_`（遮挡遮罩）、`sketch_`（手画简笔画）、`body_`（标准人）、`ground_`（自动落地）、`input_`（输入素材里选哪个：角色 / 骨架 / 手 / 网格、BVH 单位）。第二份素材按内容作名（`reference` / `hdri` / `paint` 及其 `colorspace_in_<内容>`），不加前缀。`lab2shot check templates` 校验：不带前缀的名字必须落在核心节点上或在固定名表里，带前缀的前缀必须在这张表里。
3. 同一个概念在各卡用同一个名字，不论目标节点类型（例：各跟踪卡的「参考帧」都叫 `ref_frame`）；带前缀的名字和固定名表里的名字不得在两张卡上指不同概念或不同取值集合（通则 1 的不带前缀名含义随卡，不受此限）。
4. 交付名字：**只有一种交付**的卡叫 `file_name`；**多种交付**时按类型 `exr_name` / `usd_name` / `nuke_name` / `curves_name` / `tracks_name`（`file_name` 不再出现）；同一卡**两个同类输出**时按内容 `<内容>_name`。格式同理：单一交付 `format`，多种交付 `<类型>_format`。
5. 参数放在驱动它的按钮之前：一个阶段按钮之后不得再出现作用于该阶段的参数（「高级」也放在按钮前，可折叠）；`lab2shot check templates` 按上下游校验。
6. 专名：`fit`（深度对齐的拟合）、`depth_is`、`cam_fit`（相机空间转换）虽属辅助节点，但作为专名不带角色前缀，固定名表为准。
7. 按钮（`.cook` / `.download` / `_pick`）不是值接口；值参数不叫 `_pick`（输入素材里选哪个用 `input_*`）：命令行 `apply_values` 跳过按钮，插件也不传。命名 `cook`（最后一步）/ `download` / `cook_<阶段>` / `load_target` / `<参数>_pick`，不强制跨卡同义。

**固定名表**（目标一律写成 `扩展.节点.参数`，同一节点的其他参数只写参数名；`lab2shot check templates` 读不出目标的行会报错）

| 对外名 | 含义 | 目标（扩展.节点.参数） | 值 / 备注 |
|---|---|---|---|
| `input` | 主素材文件 | `core.import_bvh.path` `core.import_usd.path` `core.read_picture.path` `core.read_sequence.path` | 显示名按素材写（输入序列 / 输入图片 / 输入动画 / 模型文件 / 画面） |
| `colorspace_in` | 主素材的输入色彩空间 | `core.read_picture.colorspace` `core.read_sequence.colorspace` | 留空 = 按文件自动（EXR = ACEScg）；放在 `input` 紧后；show_on_change |
| `reference` / `hdri` / `paint` | 第二份素材（按内容作名） | `core.read_sequence.path` / `core.read_picture.path` / `core.read_picture.path` | RoMa v2 的参考图、Relight 的 HDRI、AllTracker 的画图；其色彩空间叫 `colorspace_in_<内容>` |
| `first_frame` | 读取的首帧 / 末帧 | `core.read_sequence.first` | 留空 = 整条 |
| `last_frame` | 读取的首帧 / 末帧 | `core.read_sequence.last` | 留空 = 整条 |
| `focal` | 已知 Focal Length（mm）；留空 = 解算器自己估或用它的默认视角（人体卡：GVHMR / WHAM 按画面对角线取固定视角，HaMeR 50 mm）；例外：深度对齐卡指作为参照的 MoGe，GeoCalib 卡指 GeoCalib（重力估算） | `colmap.camera_solve.focal_mm` `core.create_camera.focal_mm` `depthanything3.geometry.focal_mm` `diffusionlight.light_probe.focal_mm` `geocalib.calibrate.focal_mm` `mapanything.reconstruct.focal_mm` `moge.geometry.focal_mm` `monst3r.reconstruct.focal_mm` `pixel3dmm.face.focal_mm` `smirk.face.focal_mm` `unidepth.geometry.focal_mm` `unik3d.geometry.focal_mm` `vipe.camera_solve.focal_mm` | 这张卡收焦距的那个节点：人体卡 = 「解算器的相机」（core.create_camera，再分发给 ViPE 与解算器）；TAPIP3D = ViPE；其余 = 主解算器。留空 = 解算器自己估 |
| `filmback` | Filmback（mm） | `anycalib.calibrate.filmback_mm` `colmap.camera_solve.filmback_mm` `core.create_camera.filmback_mm` `depthanything3.geometry.filmback_mm` `diffusionlight.light_probe.filmback_mm` `geocalib.calibrate.filmback_mm` `mapanything.reconstruct.filmback_mm` `moge.geometry.filmback_mm` `monst3r.reconstruct.filmback_mm` `pixel3dmm.face.filmback_mm` `smirk.face.filmback_mm` `unidepth.geometry.filmback_mm` `unik3d.geometry.filmback_mm` `vipe.camera_solve.filmback_mm` | 与 `focal` 同一节点，成对公开，`disable_when: "not focal"`；AnyCalib 例外（无 focal） |
| `scale_cm` | 解算器的尺度（场景 1 单位 = 多少 cm） | `colmap.camera_solve.unit_cm` `cut3r.reconstruct.unit_cm` `depthanything3.reconstruct.unit_cm` `lingbotmap.reconstruct.unit_cm` `mesh4d.solve.unit_cm` `monst3r.reconstruct.unit_cm` `pi3.reconstruct.unit_cm` `vggt.reconstruct.unit_cm` | 数字 |
| `scale` | 重定向「髋高倍率」 | `core.retarget.scale` | 数字；只此一义 |
| `size_by` | 重定向「角色缩放」/「缩放倍率」 | `core.retarget.size_by` |  |
| `size` | 重定向「角色缩放」/「缩放倍率」 | `core.retarget.size` |  |
| `scale_by` | 髋高依据 / 跟法 / 高度偏移 | `core.retarget.scale_by` |  |
| `height_mode` | 髋高依据 / 跟法 / 高度偏移 | `core.retarget.height_mode` |  |
| `height_offset` | 髋高依据 / 跟法 / 高度偏移 | `core.retarget.height_offset` |  |
| `motion_rest` | 重定向的基准姿势与摆正 | `core.retarget.motion_rest` |  |
| `motion_rest_frame` | 重定向的基准姿势与摆正 | `core.retarget.motion_rest_frame` |  |
| `target_rest` | 重定向的基准姿势与摆正 | `core.retarget.target_rest` |  |
| `target_rest_frame` | 重定向的基准姿势与摆正 | `core.retarget.target_rest_frame` |  |
| `rest_fix` | 重定向的基准姿势与摆正 | `core.retarget.rest_fix` |  |
| `motion_pose` | 重定向的初始姿势修正 | `core.retarget.motion_pose` | 「对应关系」弹窗里编辑 |
| `target_pose` | 重定向的初始姿势修正 | `core.retarget.target_pose` | 「对应关系」弹窗里编辑 |
| `unit` | 输出单位 | `core.output_usd.unit` | `cm` / `m`；只此一义 |
| `fps` | 交付帧率；动作生成 / 补帧卡上它同时是模型的帧率，放在生成按钮之前（清理卡由 BVH 自带帧率接线，不公开） | `core.output_usd.fps` `core.value_float.value` | 输出设置的帧率，或驱动多个输出的「帧率」数值节点；一张卡只公开一次 |
| `vipe_mode` | ViPE 相机解算方式 | `vipe.camera_solve.mode` | 所有卡同名 |
| `depth_model` | 非核心的深度模型 | `vipe.depth.model` |  |
| `matte_model` | BiRefNet 抠像模型 | `birefnet.matte.model` | 抠像卡、网格序列卡；精细抠像卡上 BiRefNet 是粗遮罩，叫 `rough_model` |
| `ref_frame` | 参考帧（撒点 / 画图的那一帧） | `alltracker.track.query_frame` `cotracker.track.query_frame` `tapip3d.track.query_frame` `tapnext.track.query_frame` `track4world.track.query_frame` |  |
| `picks` | 跟踪点 / 在视图里点选 | `cotracker.track.picks` `tapip3d.track.picks` `tapnext.track.picks` `track4world.track.picks` |  |
| `picks_pick` | 跟踪点 / 在视图里点选 | `cotracker.track.picks_pick` `tapip3d.track.picks_pick` `tapnext.track.picks_pick` `track4world.track.picks_pick` |  |
| `cam_src` | 相机来源（菜单） | `core.value_int.value` | 驱动相机切换与焦距切换；1 FBX / 2 USD / 3 ViPE / 4 解算器的相机（TRAM 例外：4 = TRAM 自己用 SLAM 解出的移动相机） |
| `target_src` | 目标角色（菜单） | `core.value_int.value` | 1 不重定向（解算器的人 / 解算出的手 / 原骨架 / 原角色 / 面具网格 / FLAME 头模）/ 2 FBX / 3 USD / 4 标准人（只有两张生成卡有） |
| `cam_fbx_path` | 相机文件 / 选哪台相机 | `fbx.import.path` |  |
| `cam_fbx_camera` | 相机文件 / 选哪台相机 | `fbx.import.camera` |  |
| `cam_usd_path` | 相机文件 / 选哪台相机 | `core.import_usd.path` |  |
| `cam_usd_camera` | 相机文件 / 选哪台相机 | `core.import_usd.camera` |  |
| `char_fbx_path` | 角色文件 / 选哪个角色 | `fbx.import.path` |  |
| `char_fbx_characters` | 角色文件 / 选哪个角色 | `fbx.import.characters` |  |
| `char_usd_path` | 角色文件 / 选哪个角色 | `core.import_usd.path` |  |
| `char_usd_characters` | 角色文件 / 选哪个角色 | `core.import_usd.characters` |  |
| `input_character` | 输入动画里选哪个角色（不是目标角色） | `core.import_usd.characters` |  |
| `input_skeleton` | 输入 BVH 里选哪副骨架 | `core.import_bvh.skeletons` |  |
| `input_unit` | 输入 BVH 的单位 | `core.import_bvh.unit` |  |
| `input_hand` | 输入动画里选哪只手 | `core.retarget.motion_skeleton` |  |
| `input_models` | 输入 USD 里要绑定的网格 | `core.import_usd.models` |  |
| `fit` | 深度对齐的拟合 | `core.depth_align.fit` |  |
| `depth_is` | 深度对齐：待对齐的是哪一路 | `core.depth_align.depth_is` |  |
| `ground_source` | 自动落地的地面依据 | `core.auto_ground.source` |  |
| `mapping` | 目标角色的对应关系 | `core.expression_retarget_arkit52.mapping` `core.retarget.mapping` |  |
| `model_mapping` | 算法模型骨架的骨骼对应 | `kimodo.motion.mapping` `stablemotion.cleanup.mapping` `two_stage_transformer.inbetween.mapping` `underpressure.footskate.mapping` |  |
| `feet_mapping` | 清理全流程里 UnderPressure 一段的骨骼对应 | `underpressure.footskate.mapping` |  |
| `cam_fit` | 相机空间转换的贴合方式 | `core.camera_space.fit` | 与 `fit`（深度对齐的拟合）区分 |
| `static_camera` | 固定机位 | `tram.solve.static_camera` | 只在 TRAM 卡公开：GVHMR / WHAM 接了所选相机的旋转，固定机位不起作用 |
| `least_frames` | 最少解出帧数 | `fast_sam_3d_body.solve.least_frames` `gvhmr.solve.least_frames` `hamer.hands.least_frames` `pixel3dmm.face.least_frames` `sam_3d_body.solve.least_frames` `smirk.face.least_frames` `tram.solve.least_frames` `wham.solve.least_frames` | 人体解算家族的公共参数；放「高级」 |
| `detect_threshold` | 检人阈值 | `sam_3d_body.detect_people.threshold` |  |
| `people` | 选人方式 / 人数 / 编号 / 点选 | `core.select_people.mode` |  |
| `people_count` | 选人方式 / 人数 / 编号 / 点选 | `core.select_people.count` |  |
| `people_ids` | 选人方式 / 人数 / 编号 / 点选 | `core.select_people.ids` |  |
| `people_picks` | 选人方式 / 人数 / 编号 / 点选 | `core.select_people.picks` |  |
| `people_picks_pick` | 选人方式 / 人数 / 编号 / 点选 | `core.select_people.picks_pick` |  |
| `file_name` | 唯一一种交付的名字 | `core.output_exr.name` `core.output_tracks.name` `core.output_usd.name` | 只在单交付卡 |
| `exr_name` | 多交付卡里各类型输出的名字 | `core.output_exr.name` | 穹顶灯 USD 也是 `usd_name` |
| `usd_name` | 多交付卡里各类型输出的名字 | `core.output_usd.name` | 穹顶灯 USD 也是 `usd_name` |
| `nuke_name` | 多交付卡里各类型输出的名字 | `core.output_nuke_camera.name` | 穹顶灯 USD 也是 `usd_name` |
| `curves_name` | 多交付卡里各类型输出的名字 | `core.output_curves.name` | 穹顶灯 USD 也是 `usd_name` |
| `colorspace_out` | 卡上唯一一个带色彩空间的输出的色彩空间 | `core.output_exr.colorspace` | 放「交付」组；只含数据层的 EXR 引擎自动灰掉（applies） |
| `colorspace_out_exr` | 同时有 EXR 与穹顶灯 USD 贴图时 | `core.output_exr.colorspace` | 三张光照卡 |
| `exr_bit_depth` | EXR 位深 | `core.output_exr.exr_bit_depth` | 显示名「EXR 位深」，放「交付 / 高级」 |
| `exr_compression` | EXR 压缩 | `core.output_exr.exr_compression` | 显示名「EXR 压缩」，放「交付 / 高级」；模板下拉只列无损 `zips`（ZIP 逐行，默认）/ `zip` / `piz` / `none` |
| `format` | 交付格式 | `core.output_tracks.format` | 按通则 4 |
| `curves_format` | 交付格式 | `core.output_curves.format` | 按通则 4 |
| `tracks_format` | 交付格式 | `core.value_text.value` | 按通则 4 |
| `origin` | 2D 跟踪点坐标原点 | `core.output_tracks.origin` `core.value_text.value` | RoMa v2 用一个文字数值节点同时驱动两个跟踪点输出 |
| `min_visible` | 2D 跟踪点最少可见帧数 | `core.output_tracks.min_visible` |  |
