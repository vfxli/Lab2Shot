# 模板对外参数名约定

模板 `exposed` 树里每个条目的 `name` 是命令行（`lab2shot cook --set 名=值`）、DCC 插件和批量脚本使用的接口。规则：**一个对外名 = 一个含义**，所有模板遵守；`lab2shot check templates` 按本文件校验。

**通则**
1. 对外名只有两种来源，没有「主算法」「辅助节点」之分：一张卡就是一张节点图，图里每个节点地位相同。
   - **共同概念**：下面固定名表里的名字（主素材 `input`、焦距 `focal`、开始帧、结束帧、交付名、「摄影机来源」这类菜单……）。它们在所有卡上指同一件事，批量脚本写一次各卡通用。只有在多张卡上意思相同的东西才进这张表。
   - **其余一律是「节点 id_参数名」**：目标 `solve.step` 的对外名就是 `solve_step`，`fuse.voxel_size` 就是 `fuse_voxel_size`。节点 id 在一张图里唯一，名字天然不重复。
2. 节点 id 是看得懂的英文词（小写字母开头，只用小写字母、数字和 _，至少 3 个字符）：`read`、`solve`、`depth`、`fuse`、`retarget`，不用 `r`、`tr` 这类缩写。对外名从它来，它就是接口的一部分。
3. 固定名表里的名字不得在两张卡上指不同概念或不同取值集合；菜单的每个取值在各卡含义一致（备注里写明的取值表为准）。
4. 交付名字：**只有一种交付**的卡叫 `file_name`；**多种交付**时按类型 `exr_name` / `usd_name` / `nuke_name` / `curves_name` / `tracks_name`（`file_name` 不再出现）；同一卡**两个同类输出**时第二个按通则 1 叫「节点 id_name」。格式同理：单一交付 `format`，多种交付 `<类型>_format`。
5. 参数放在驱动它的按钮之前：一个阶段按钮之后不得再出现作用于该阶段的参数（「高级」也放在按钮前，可折叠）。
6. 按钮不是值接口（命令行 `apply_values` 跳过按钮，插件也不传）：最后一步叫 `cook`（显示名不自己写，写共用按钮词 `"word": "pack"`——卡上有其他阶段按钮时，显示「打包」——否则 `"word": "cook"`，显示「计算」），下载叫 `download`，其余阶段按钮叫 `cook_<节点 id>`，其他按钮（在视图里点选等）按通则 1 叫「节点 id_按钮名」。
7. 一项可以驱动多个同类参数：`target` 写成列表（`["cotracker.picks", "tapnext.picks"]`），值写进每一个，界面显示第一个的；各目标必须是同一种参数、现在的值相同（`lab2shot check templates` 校验）。对外名照第一个目标起，固定名表的名字要求每个目标都在表里。

`lab2shot check templates` 按这些规则逐条校验（engine/conventions.py）。

**固定名表**（目标一律写成 `扩展.节点.参数`，同一节点的其他参数只写参数名；`lab2shot check templates` 读不出目标的行会报错）

| 对外名 | 含义 | 目标（扩展.节点.参数） | 值 / 备注 |
|---|---|---|---|
| `input` | 主素材文件 | `bvh.import.path` `usd.import.path` `file_still.path` `file.path` | 显示名按素材写（输入图像序列 / 输入图像 / 输入动画 / 模型文件 / 图像） |
| `colorspace_in` | 主素材的输入颜色空间 | `file_still.colorspace` `file.colorspace` | 留空 = 按文件自动（EXR = ACEScg）；放在 `input` 紧后；show_on_change |
| `reference` | 第二份素材（参考图） | `file.path` `file_pattern.folder` | RoMa v2 的参考序列、QwenImage 的参考图文件夹 |
| `first_frame` | 读取的开始帧 / 结束帧 | `file.first` | 留空 = 整条 |
| `last_frame` | 读取的开始帧 / 结束帧 | `file.last` | 留空 = 整条 |
| `focal` | 已知 焦距（mm）；留空 = 解算器自己估或用它的默认视角（人体卡：GVHMR / WHAM 按画面对角线取固定视角，HaMeR 50 mm）；例外：深度对齐卡指作为参照的 MoGe，GeoCalib 卡指 GeoCalib（重力估算） | `colmap.camera_solve.focal_mm` `camera.focal_mm` `depthanything3.depth.focal_mm` `diffusionlight.light_probe.focal_mm` `geocalib.calibrate.focal_mm` `mapanything.reconstruct.focal_mm` `moge.depth.focal_mm` `monst3r.reconstruct.focal_mm` `pixel3dmm.face_solve.focal_mm` `smirk.face_solve.focal_mm` `unidepth.depth.focal_mm` `unik3d.depth.focal_mm` `vidmap.camera_solve.focal_mm` `vipe.camera_solve.focal_mm` `wildpose.reconstruct.focal_mm` | 这张卡收焦距的那个节点：人体卡 = 「解算器的摄影机」（camera，再分发给 ViPE 与解算器）；TAPIP3D = ViPE；其余 = 主解算器。留空 = 解算器自己估 |
| `filmback` | 胶片背（mm） | `anycalib.calibrate.filmback_mm` `colmap.camera_solve.filmback_mm` `camera.filmback_mm` `depthanything3.depth.filmback_mm` `diffusionlight.light_probe.filmback_mm` `geocalib.calibrate.filmback_mm` `mapanything.reconstruct.filmback_mm` `moge.depth.filmback_mm` `monst3r.reconstruct.filmback_mm` `pixel3dmm.face_solve.filmback_mm` `smirk.face_solve.filmback_mm` `unidepth.depth.filmback_mm` `unik3d.depth.filmback_mm` `vidmap.camera_solve.filmback_mm` `vipe.camera_solve.filmback_mm` `wildpose.reconstruct.filmback_mm` | 与 `focal` 同一节点，成对公开，`disable_when: "not focal"`；AnyCalib 例外（无 focal） |
| `scale_cm` | 解算器的尺度（场景 1 单位 = 多少 cm） | `colmap.camera_solve.unit_cm` `cut3r.reconstruct.unit_cm` `da3long.reconstruct.unit_cm` `depthanything3.reconstruct.unit_cm` `hyworld2.reconstruct.unit_cm` `lingbotmap.reconstruct.unit_cm` `mesh4d.reconstruct.unit_cm` `monst3r.reconstruct.unit_cm` `neoverse.reconstruct.unit_cm` `pi3.reconstruct.unit_cm` `vggt.reconstruct.unit_cm` `vidmap.camera_solve.unit_cm` `wildpose.reconstruct.unit_cm` | 数字 |
| `scale` | 重定目标「髋高倍率」 | `retarget.scale` | 数字；只此一义 |
| `size_by` | 重定目标「角色缩放」/「缩放倍率」 | `retarget.size_by` |  |
| `size` | 重定目标「角色缩放」/「缩放倍率」 | `retarget.size` |  |
| `scale_by` | 髋高依据 / 跟法 / 高度偏移 | `retarget.scale_by` |  |
| `height_mode` | 髋高依据 / 跟法 / 高度偏移 | `retarget.height_mode` |  |
| `height_offset` | 髋高依据 / 跟法 / 高度偏移 | `retarget.height_offset` `retarget_post.height_offset` `value_float.value` |  |
| `motion_rest` | 重定目标的静止姿势与摆正 | `retarget.motion_rest` `retarget_prepare.motion_rest` |  |
| `motion_rest_frame` | 重定目标的静止姿势与摆正 | `retarget.motion_rest_frame` `retarget_prepare.motion_rest_frame` |  |
| `target_rest` | 重定目标的静止姿势与摆正 | `retarget.target_rest` `retarget_prepare.target_rest` |  |
| `target_rest_frame` | 重定目标的静止姿势与摆正 | `retarget.target_rest_frame` `retarget_prepare.target_rest_frame` |  |
| `motion_pose` | 重定目标的初始姿势修正 | `retarget.motion_pose` `retarget_prepare.motion_pose` | 「对应关系」弹窗里编辑 |
| `target_pose` | 重定目标的初始姿势修正 | `retarget.target_pose` `retarget_prepare.target_pose` | 「对应关系」弹窗里编辑 |
| `unit` | 输出单位 | `usd.output.unit` | `cm` / `m`；只此一义 |
| `fps` | 交付帧速率；动作生成 / 补帧卡上它同时是模型的帧速率，放在生成按钮之前（清理卡由输入动画自带帧速率接线，不公开） | `usd.output.fps` `value_float.value` `value_int.value` | 格式输出节点的帧速率，或驱动多个输出的「帧速率」数值节点；一张卡只公开一次 |
| `vipe_mode` | ViPE 摄影机解算方式 | `vipe.camera_solve.mode` | 所有卡同名 |
| `depth_model` | 非核心的深度模型 | `depthanything3.depth.model` `vipe.depth.model` |  |
| `depth_resolution` | 非核心深度模型的处理分辨率 | `depthanything3.depth.resolution` |  |
| `matte_model` | BiRefNet 抠像模型 | `birefnet.matte.model` | 抠像卡、网格序列卡；精细抠像卡上 BiRefNet 是粗遮罩，叫 `rough_model` |
| `ref_frame` | 参考帧（撒点 / 画图的那一帧） | `alltracker.track.query_frame` `cotracker.track.query_frame` `tapip3d.track.query_frame` `tapnext.track.query_frame` `track4world.track.query_frame` |  |
| `picks` | 跟踪点 / 在视图里点选 | `cotracker.track.picks` `tapip3d.track.picks` `tapnext.track.picks` `track4world.track.picks` |  |
| `picks_pick` | 跟踪点 / 在视图里点选 | `cotracker.track.picks_pick` `tapip3d.track.picks_pick` `tapnext.track.picks_pick` `track4world.track.picks_pick` |  |
| `cam_src` | 摄影机来源（菜单） | `value_int.value` | 驱动摄影机切换与焦距切换；1 FBX / 2 USD / 3 ViPE / 4 解算器的摄影机（TRAM 例外：4 = TRAM 自己用 SLAM 解出的移动摄影机） |
| `target_src` | 目标角色（菜单） | `value_int.value` | 1 不重定目标（解算器的人 / 解算出的手 / 原骨架 / 原角色 / 面部网格 / FLAME 头部网格）/ 2 FBX / 3 USD / 4 标准人（只有两张生成卡有） |
| `motion_src` | 源动作来源（菜单） | `value_int.value` | 1 动画（导入的 FBX：有带蒙皮的角色就用角色，没有就用骨架，由「有没有」节点 fbx_has_char 接切换 fbx_kind 自动选，不让用户先分）/ 2 BVH / 3 提示词生成（Kimodo）/ 5 角色动画（导入的 USD）（4 已并进 1，不再用；AI 重定目标卡列 1、2、3，默认 1；要源一侧带网格的 STaR、MeshRet 只列 1 和 3，只接角色；动捕清理卡列 1、2，默认 1；Kimodo 中间帧卡列 1、5，默认 1） |
| `cam_fbx_path` | 摄影机文件 / 选哪台摄影机 | `fbx.import.path` |  |
| `cam_fbx_camera` | 摄影机文件 / 选哪台摄影机 | `fbx.import.camera` |  |
| `cam_usd_path` | 摄影机文件 / 选哪台摄影机 | `usd.import.path` |  |
| `cam_usd_camera` | 摄影机文件 / 选哪台摄影机 | `usd.import.camera` |  |
| `char_fbx_path` | 角色文件 / 选哪个角色 | `fbx.import.path` |  |
| `char_fbx_characters` | 角色文件 / 选哪个角色 | `fbx.import.characters` |  |
| `char_usd_path` | 角色文件 / 选哪个角色 | `usd.import.path` |  |
| `char_usd_characters` | 角色文件 / 选哪个角色 | `usd.import.characters` |  |
| `input_character` | 输入动画里选哪个角色（不是目标角色） | `usd.import.characters` |  |
| `input_skeleton` | 输入 BVH 里选哪副骨架 | `bvh.import.skeletons` |  |
| `input_unit` | 输入 BVH 的单位 | `bvh.import.unit` |  |
| `mapping` | 目标角色的对应关系 | `retarget_expression.mapping` `retarget.mapping` `retarget_prepare.mapping` |  |
| `model_mapping` | 算法模型骨架的骨骼对应 | `kimodo.motion.mapping` `stablemotion.cleanup.mapping` `two_stage_transformer.inbetween.mapping` `underpressure.footskate_cleanup.mapping` |  |
| `cam_fit` | 摄影机空间转换的贴合方式 | `camera_space.fit` |  |
| `least_frames` | 最少解出帧数 | `fast_sam_3d_body.solve.least_frames` `gvhmr.solve.least_frames` `hamer.hand_solve.least_frames` `pixel3dmm.face_solve.least_frames` `sam_3d_body.solve.least_frames` `smirk.face_solve.least_frames` `tram.solve.least_frames` `wham.solve.least_frames` | 人体解算家族的公共参数；放「高级」 |
| `detect_threshold` | 检人阈值 | `sam_3d_body.detect_people.threshold` |  |
| `people` | 选人方式 / 人数 / 编号 / 点选 | `select_people.mode` |  |
| `people_count` | 选人方式 / 人数 / 编号 / 点选 | `select_people.count` |  |
| `people_ids` | 选人方式 / 人数 / 编号 / 点选 | `select_people.ids` |  |
| `people_picks` | 选人方式 / 人数 / 编号 / 点选 | `select_people.picks` |  |
| `people_picks_pick` | 选人方式 / 人数 / 编号 / 点选 | `select_people.picks_pick` |  |
| `file_name` | 唯一一种交付的名字 | `exr.output.name` `image.output.name` `tracks.output.name` `usd.output.name` | 只在单交付卡 |
| `exr_name` | 多交付卡里各类型输出的名字 | `exr.output.name` | 穹顶灯 USD 也是 `usd_name` |
| `usd_name` | 多交付卡里各类型输出的名字 | `usd.output.name` | 穹顶灯 USD 也是 `usd_name` |
| `nuke_name` | 多交付卡里各类型输出的名字 | `nuke.output.name` | 穹顶灯 USD 也是 `usd_name` |
| `curves_name` | 多交付卡里各类型输出的名字 | `curves.output.name` | 穹顶灯 USD 也是 `usd_name` |
| `colorspace_out` | 卡上唯一一个带颜色空间的输出的颜色空间 | `exr.output.colorspace` | 放「交付」组；只含数据层的 EXR 引擎自动灰掉（applies） |
| `colorspace_out_exr` | 同时有 EXR 与穹顶灯 USD 贴图时 | `exr.output.colorspace` | 三张光照卡 |
| `exr_bit_depth` | EXR 位深 | `exr.output.exr_bit_depth` | 显示名「EXR 位深」，放「交付 / 高级」 |
| `exr_compression` | EXR 压缩 | `exr.output.exr_compression` | 显示名「EXR 压缩」，放「交付 / 高级」；模板下拉只列无损 `zips`（ZIP 逐行，默认）/ `zip` / `piz` / `none` |
| `format` | 交付格式 | `tracks.output.format` | 按通则 4 |
| `curves_format` | 交付格式 | `curves.output.format` | 按通则 4 |
| `prompt` | 卡的中文提示词（内嵌「翻译」节点转成英文后驱动上游提示词参数）；QwenImage 生成卡直接写英文 | `value_string.value` `diffusers.generate.prompt` | 语言路内嵌翻译的七张卡 |
| `origin` | 2D 跟踪点坐标原点 | `tracks.output.origin` `value_string.value` | RoMa v2 用一个文字数值节点同时驱动两个跟踪点输出 |
| `min_visible` | 2D 跟踪点最少可见帧数 | `tracks.output.min_visible` |  |

