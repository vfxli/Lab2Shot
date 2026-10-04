/** model/rigPair.ts 的测试：Node 直接跑（node src/model/rigPair.test.ts，Node 22.6+ 自带去类型），不进页面的包。
 * 数据是 model/rigPair.fixture.ts 的那份 handle_data。全部通过时打印一行，有一条不对就抛错（退出码非 0）。 */

import {
  autoAllPatch, ruleProblems, anchorJoint, confidenceOf, unsureParts, autoPatch, autoState, basePosition, currentRecord, effective, jointStates, link, linksOf, mappingSummary, mergedOf, partFor, partOf, problems, recordOf,
  ruleNow, scaledPoint, scaledWorld, suggestedRecord, shownMask, sideOffsets, spread, toggleIgnored, unlinkJoint, unlinkLink, valueOf, wiredRule, withTouched,
  type RigRoles,
} from "./rigPair.ts";
import { DATA, FIXED } from "./rigPair.fixture.ts";
import { invertAffine, skinMatrices, skinPoints } from "./skinning.ts";
import { addWords } from "../i18n/words.ts";
import { WORDS } from "../messages/generatedCatalogue.ts";
import { setLang } from "../i18n/lang.ts";

addWords(WORDS); // the words it says (lab2shot/i18n/zh/ui/rig.toml), in Chinese: the expectations below are written in it
setLang("zh");

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  const g = JSON.stringify(got), w = JSON.stringify(want);
  if (g !== w) throw new Error(`${what}\n  得到 ${g}\n  应为 ${w}`);
}

const ROLES: RigRoles = {
  mapping: "mapping", src_pose: "motion_pose", dst_pose: "target_pose", src_ignore: "motion_ignore", dst_ignore: "target_ignore",
  ignore_rule: "ignore_rule", auto_record: "auto_record",
};

// ---- 合并：空参数 = 全部推测，写回为 null
const auto = mergedOf(null, DATA);
eq(auto.manual.size, 0, "空参数没有手动部位");
eq(auto.rows["spine"], { part: "spine", src: ["Spine"], dst: ["spine_01", "spine_02"] }, "没写的部位按推测");
eq(valueOf(auto, DATA), null, "没有手动部位时写回 null");
eq(partOf(auto, "src").get("Twist"), undefined, "辅助骨没有部位");

// ---- 状态：配对 / 未配对 / 忽略
eq(jointStates(auto, DATA, "src", ["RightUpLeg"]), ["paired", "paired", "unpaired", "paired", "paired", "ignored"], "源骨点的状态");
eq(jointStates(mergedOf([{ part: "hips", src: ["Hips"], dst: [] }], DATA), DATA, "src", [])[0], "unpaired", "另一栏空着：未配对");

// ---- 连线：链按 spread 均分
eq(spread(2, ["Spine"]), ["Spine", null], "spread：驱动一侧更短时多出来的为 null");
eq(spread(3, ["a", "b", "c", "d", "e"]), ["a", "c", "e"], "spread：首对首、尾对尾");
eq(linksOf(auto, DATA).map((l) => `${l.part}:${l.src}-${l.dst}`),
  ["hips:Hips-pelvis", "spine:Spine-spine_01", "l.thigh:LeftUpLeg-thigh_l", "l.shin:LeftLeg-calf_l", "r.thigh:RightUpLeg-thigh_r"], "配对连线");

// ---- 配对：先点的骨点没有部位 → 要选部位；选「脊柱」后按层级深度插进链里
const twist = { col: "src" as const, joint: "Twist" }, spine2 = { col: "dst" as const, joint: "spine_02" };
eq(partFor(auto, DATA, twist, spine2), null, "先点的骨点没有部位：弹部位菜单");
eq(partFor(auto, DATA, spine2, twist), "spine", "先点目标的 spine_02：取它的部位");
const linked = link(auto, DATA, "spine", twist, spine2);
eq(linked.rows["spine"], { part: "spine", src: ["Spine", "Twist"], dst: ["spine_01", "spine_02"] }, "链按深度插入");
eq(valueOf(linked, DATA), [{ part: "spine", src: ["Spine", "Twist"], dst: ["spine_01", "spine_02"] }], "只写手动定过的部位");

// ---- 配对把关节从别的部位拿走（一个关节只在一处），非链部位换成这一个
const moved = link(auto, DATA, "l.thigh", { col: "src", joint: "LeftLeg" }, { col: "dst", joint: "thigh_l" });
eq(moved.rows["l.thigh"].src, ["LeftLeg"], "非链部位：换成新点的关节");
eq(moved.rows["l.shin"].src, [], "从原来的部位拿走");
eq(valueOf(moved, DATA)?.map((r) => r.part), ["l.thigh", "l.shin"], "两个部位都成了手动");

// ---- 解除：点已配对的骨点、点连线
const cut = unlinkJoint(auto, DATA, "src", "LeftUpLeg");
eq(cut.rows["l.thigh"], { part: "l.thigh", src: [], dst: ["thigh_l"] }, "点已配对骨点：从部位行去掉");
eq(jointStates(cut, DATA, "dst", [])[3], "unpaired", "对面的骨点变成未配对");
const cutLink = unlinkLink(auto, DATA, { part: "spine", src: "Spine", dst: "spine_01" });
eq(cutLink.rows["spine"], { part: "spine", src: [], dst: ["spine_02"] }, "点连线：两端都去掉");

// ---- 显示用校验
eq(Object.keys(problems(auto, DATA)), [], "推测的对应没有问题");
eq(problems(mergedOf([{ part: "hips", src: ["Hips"], dst: [] }], DATA), DATA)["hips"], "必需的部位：目标这边还没配", "必需部位缺一侧");
eq(mappingSummary([{ part: "hips", src: ["Hips"], dst: [] }], DATA), { text: "手动 1 个部位，其余自动 · 有问题：髋", bad: true }, "摘要行");
eq(mappingSummary(null, null), { text: "全部自动", bad: false }, "摘要行（没有手柄数据）");

// ---- 忽略：点 = 连同子孙，Alt 点 = 只这一个
eq(toggleIgnored([], DATA.src, "Spine", false), ["Spine", "Twist"], "连同子孙一起忽略");
eq(toggleIgnored(["Spine", "Twist"], DATA.src, "Spine", false), [], "已忽略的再点：连同子孙一起取消");
eq(toggleIgnored(["Spine", "Twist"], DATA.src, "Spine", true), ["Twist"], "Alt 点只切换这一个");
eq(toggleIgnored(["Ghost"], DATA.src, "LeftLeg", true), ["LeftLeg", "Ghost"], "不认得的名字留着");

// ---- 眼睛：隐藏关节和它的子孙
eq(shownMask(DATA.src, ["Spine"]), [true, false, false, true, true, true], "眼睛隐藏子树");

// ---- 三个「自动」：写服务端的建议、记进 auto_record；状态行
let params: Record<string, unknown> = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "", auto_record: "" };
const state = (kind: "mapping" | "pose" | "ignore") => {
  const rule = ruleNow(DATA, params.ignore_rule);
  return autoState(kind, recordOf(params.auto_record), currentRecord(ROLES, params), { id: rule, label: DATA.rules!.find((r) => r.id === rule)!.label }).text;
};
eq([state("mapping"), state("pose"), state("ignore")], ["还没执行", "自动（还没改过）", "自动（还没改过）"], "都还没执行：姿态、忽略按默认自动");
eq(ruleNow(DATA, ""), "generic", "规则空着时预选 default_rule");

params = { ...params, ...autoPatch("mapping", DATA, ROLES, params)! };
eq(params.mapping, DATA.auto_mapping, "自动对应 = 推测原样写进参数");
eq(state("mapping"), "已自动", "刚自动完");
params = { ...params, mapping: valueOf(link(mergedOf(params.mapping, DATA), DATA, "spine", twist, spine2), DATA) };
eq(state("mapping"), "已自动 · 之后手动改了 1 个", "手动改了一个部位");

params = { ...params, ...autoPatch("pose", DATA, ROLES, params)! };
eq(params.motion_pose, DATA.src.auto_pose, "自动姿态 = 服务端的行原样写进参数");
eq(params.target_pose, [], "目标没有建议的行");
eq(state("pose"), "已自动", "姿态刚自动完");
params = { ...params, motion_pose: [{ joint: "LeftUpLeg", translate: [0, 0, 0], rotate: [0, 0, 7], scale: [1, 1, 1] }, { joint: "Spine", translate: [1, 0, 0], rotate: [0, 0, 0], scale: [1, 1, 1] }] };
eq(state("pose"), "已自动 · 之后手动改了 2 个", "改了两个关节的姿态");
eq(state("mapping"), "已自动 · 之后手动改了 1 个", "姿态的自动不动对应关系的记录");

params = { ...params, ...autoPatch("ignore", DATA, ROLES, params, ruleNow(DATA, params.ignore_rule))! };
eq([params.ignore_rule, params.motion_ignore, params.target_ignore], ["generic", ["Twist"], []], "自动忽略 = 这条规则的名单，并记下规则");
eq(state("ignore"), "已自动", "忽略刚自动完");
params = { ...params, motion_ignore: toggleIgnored(params.motion_ignore as string[], DATA.src, "LeftLeg", true) };
eq(state("ignore"), "已自动 · 之后手动改了 1 个", "多忽略了一个");
params = { ...params, ignore_rule: "star" };
eq(state("ignore"), "规则已换成「STaR（固定身体部位）」，还没按它执行", "换了规则还没执行");
eq(recordOf(params.auto_record)?.ignore?.rule, "generic", "记录里还是上次执行的规则");
eq(autoPatch("pose", FIXED, { mapping: "mapping" }, {}), null, "没有姿态角色：自动姿态写不了");

// ---- 「全部自动」：三项叠成一份补丁，记录里三项都在（不被后一项盖掉），和逐个点完全一样
{
  const fresh: Record<string, unknown> = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "", auto_record: "" };
  const all = { ...fresh, ...autoAllPatch(["mapping", "ignore", "pose"], DATA, ROLES, fresh, ruleNow(DATA, ""))! };
  let one = { ...fresh };
  for (const k of ["mapping", "ignore", "pose"] as const) one = { ...one, ...autoPatch(k, DATA, ROLES, one, ruleNow(DATA, ""))! };
  eq(Object.keys(recordOf(all.auto_record) ?? {}).sort(), ["ignore", "mapping", "pose"], "全部自动：记录里三项都在");
  eq([all.mapping, all.motion_pose, all.motion_ignore, all.ignore_rule], [one.mapping, one.motion_pose, one.motion_ignore, one.ignore_rule], "全部自动 = 依次点三个");
  eq(autoAllPatch([], DATA, ROLES, fresh), null, "一项都做不了：没有补丁");
}

// ---- 模型固定骨架：那一栏由节点定死
const fixed = mergedOf([{ part: "spine", src: [], dst: ["whatever"] }], FIXED);
eq(fixed.rows["spine"].dst, ["spine"], "固定一栏永远是模型自己的关节");
eq(partFor(fixed, FIXED, { col: "src", joint: "Twist" }, { col: "dst", joint: "rhip" }), "r.thigh", "有固定一侧时按固定关节的部位配");
const fixedLinked = link(fixed, FIXED, "r.thigh", { col: "src", joint: "RightUpLeg" }, { col: "dst", joint: "rhip" });
eq(fixedLinked.rows["r.thigh"], { part: "r.thigh", src: ["RightUpLeg"], dst: ["rhip"] }, "配到固定一侧：只改可编辑的一栏");
eq(unlinkJoint(fixedLinked, FIXED, "dst", "rhip").rows["r.thigh"], { part: "r.thigh", src: [], dst: ["rhip"] }, "点固定一侧的骨点解除：去掉另一栏");
eq(jointStates(fixedLinked, FIXED, "dst", []), ["unpaired", "paired", "unpaired", "paired", "paired"], "固定一侧的状态（spine 那一栏源被清空）");

// ---- 默认自动（effective）：四种情形
const empty = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "star", auto_record: "" };
const e1 = effective(DATA, ROLES, empty);
eq([e1.poses.src, e1.poses.dst, e1.auto.pose], [DATA.src.auto_pose, [], { src: true, dst: false }], "姿态：没执行过、行空着 → 用建议行（目标的建议是空的）");
eq([e1.ignored, e1.auto.ignore], [{ src: ["Twist", "LeftLeg"], dst: ["spine_02"] }, true], "忽略：没执行过、两侧都空 → 现在规则的名单");
const own = [{ joint: "Spine", translate: [0, 0, 0], rotate: [5, 0, 0], scale: [1, 1, 1] }];
const e2 = effective(DATA, ROLES, { ...empty, motion_pose: own, target_ignore: ["thigh_r"] });
eq([e2.poses.src, e2.auto.pose.src], [own, false], "姿态：这一侧有行 → 参数的行原样");
eq([e2.ignored, e2.auto.ignore], [{ src: [], dst: ["thigh_r"] }, false], "忽略：有一侧不空 → 名单原样");
const ran = JSON.stringify({ pose: { motion: [], target: [] }, ignore: { rule: "generic", motion: [], target: [] } });
const e3 = effective(DATA, ROLES, { ...empty, auto_record: ran });
eq([e3.poses.src, e3.ignored, e3.auto], [[], { src: [], dst: [] }, { pose: { src: false, dst: false }, ignore: false, size: { src: false, dst: false } }], "执行过（记录里有）且空着 → 空就是空");
eq(autoState("pose", recordOf(ran), currentRecord(ROLES, { ...empty, auto_record: ran })).text, "已自动", "执行过：状态照常");
eq(autoState("pose", null, currentRecord(ROLES, { ...empty, motion_pose: own })).text, "还没执行", "没执行过但手动改过：还没执行");
// 第一次手动改从实际用的那一份开始：动一个关节，自动的其余行都在
eq(toggleIgnored(e1.ignored.src, DATA.src, "RightUpLeg", true), ["Twist", "LeftLeg", "RightUpLeg"], "从默认自动的名单上改");

// ---- 规则按连线预选（目录里解算器的 retarget_rules，第一条是它的默认）
eq(wiredRule(["star", "star_x"], "generic", "generic"), "star", "接着 STaR、规则还是默认 → 写它的第一条");
eq(wiredRule(["star"], "", "generic"), "star", "规则空着也算默认");
eq(wiredRule(["star"], "star", "generic"), null, "已经是它：不写");
eq(wiredRule(undefined, "generic", "generic"), null, "接着的不是声明规则的解算器：不写");
eq(wiredRule(["star"], "sata", "generic"), null, "使用者选过别的：不改");

// ---- 第一次手动改补上 auto_record：默认自动 → 清空 → 留在空
const fresh = { ...empty, ignore_rule: "generic" };
const DATA2 = { ...DATA, dst: { ...DATA.dst, auto_pose: [{ joint: "thigh_l", translate: [0, 0, 0], rotate: [0, 0, -5], scale: [1, 1, 1] }] } };
const clearPose = withTouched({ motion_pose: [] }, DATA2, ROLES, fresh)!;
const afterPose = { ...fresh, ...clearPose };
eq(effective(DATA2, ROLES, afterPose).poses, { src: [], dst: DATA2.dst.auto_pose }, "清空源姿态：源留在空，目标的自动行写实留着");
eq(recordOf(afterPose.auto_record)?.pose, { manual: true, motion: DATA2.src.auto_pose, target: DATA2.dst.auto_pose }, "记下当时的默认自动结果");
eq(autoState("pose", recordOf(afterPose.auto_record), currentRecord(ROLES, afterPose)).text, "手动改过 1 个", "状态行：手动改过");
eq(withTouched({ motion_pose: [] }, DATA2, ROLES, afterPose), { motion_pose: [] }, "已有记录：原样写");
const clearIgnore = withTouched({ motion_ignore: [], target_ignore: [] }, DATA, ROLES, fresh)!;
const afterIgnore = { ...fresh, ...clearIgnore };
eq(effective(DATA, ROLES, afterIgnore).ignored, { src: [], dst: [] }, "清空忽略：留在空，不再回到规则的名单");
eq(autoState("ignore", recordOf(afterIgnore.auto_record), currentRecord(ROLES, afterIgnore), { id: "star", label: "STaR" }).text, "手动改过 1 个", "手动记下的忽略不按规则比");
const oneSide = { ...fresh, ...withTouched({ motion_ignore: ["Twist", "LeftLeg"] }, DATA, ROLES, { ...fresh, ignore_rule: "star" })! };
eq(oneSide.target_ignore, ["spine_02"], "只改一侧：另一侧的自动名单写实留着");
eq(withTouched({ motion_pose: [] }, null, ROLES, fresh), null, "手柄数据不在手：补不了，不写");
eq(withTouched({ mapping: null }, null, ROLES, fresh), { mapping: null }, "对应关系不受影响");

// ---- 舞台上的摆放：锚点（髋，没有就是根）水平挪回原点，Y 不动，再前后拉开
eq(anchorJoint(DATA.src), 0, "锚点：髋部位的关节");
eq(anchorJoint({ parents: [-1, 0, 1], parts: {} }), 0, "没配髋：第一个根关节");
eq(anchorJoint({ parents: [1, -1, 1], parts: { hips: [2] } }), 2, "髋不在根上也用它");
const far = { ...DATA.src, before: { local: DATA.src.before!.local.map((m, i) => (i === 0 ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 300, 95, -800, 1] : m)) } };
eq(basePosition(far, 4), [310, 50, -800], "基准位置沿父子链累乘");
eq(sideOffsets({ ...DATA, src: far }, 100), { src: [-300, 0, 750], dst: [-0, 0, 50] }, "源挪回原点（Y 不动）、拉开 ±gap/2");
eq(sideOffsets({ ...DATA, src: far }, 0).src, [-300, 0, 800], "前后距离 0：两侧锚点重合");
eq(sideOffsets(FIXED, 100), { src: [-0, 0, -0], dst: [0, 0, 0] }, "一侧固定：不拉开，固定一侧不挪");

// ---- 多于四个影响的网格在 CPU 上蒙皮（model/skinning.ts）
{
  // 两个关节：0 在原点，1 在 (0,10,0)；绑定姿势就是这样。新姿势：关节 1 绕 Z 转 90°、整体挪 (100,0,0)
  const bind = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 10, 0, 0, 1, 0, 0, 0, 0, 1]);
  const world = new Float32Array([1, 0, 0, 100, 0, 1, 0, 0, 0, 0, 1, 0, 0, -1, 0, 100, 1, 0, 0, 10, 0, 0, 1, 0]);
  const skin = skinMatrices(world, bind, 2);
  const round = (a: Float32Array) => Array.from(a).map((v) => Math.round(v * 1000) / 1000 + 0);
  // 点 (0,20,0)：全跟关节 1 → 关节 1 处 (100,10,0) 再沿转过的 +Y（= -X）走 10 → (90,10,0)
  eq(round(skinPoints([0, 20, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], 5, skin)), [90, 10, 0], "单个影响（每点 5 个槽）");
  // 一半跟关节 0（挪到 (100,20,0)）、一半跟关节 1（(90,10,0)）→ 平均
  eq(round(skinPoints([0, 20, 0], [0, 1, 0, 0, 0], [0.5, 0.5, 0, 0, 0], 5, skin)), [95, 15, 0], "两个影响按权重混合");
  eq(round(skinPoints([1, 2, 3], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], 5, skin)), [1, 2, 3], "没有权重的点留在原处");
  eq(invertAffine([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), null, "缩放为 0 的绑定不可逆");
}

// ---- 换过骨架：参数里点了现在骨架上没有的关节的行，按推测（同服务端 drop_stale），并记下是哪些部位
{
  const stale = mergedOf([{ part: "spine", src: ["NoSuchJoint"], dst: [] }, { part: "l.thigh", src: [], dst: [] }], DATA);
  eq([stale.stale, stale.manual.has("spine"), stale.manual.has("l.thigh")], [["spine"], false, true], "换过骨架的行按推测，故意留空的行照旧算手动");
  eq(stale.rows.spine, mergedOf(null, DATA).rows.spine, "那个部位就是推测的行");
  eq(Object.keys(problems(stale, DATA)).includes("spine"), false, "不再当成问题报");
}

// ---- 尺寸（「动作尺寸」「目标尺寸」）：空着 = 自动（不看记录），「自动尺寸」写服务端的建议，显示按实际用的系数
{
  const SR: RigRoles = { ...ROLES, src_scale: "motion_scale", dst_scale: "target_scale" };
  const D = { ...DATA, src: { ...DATA.src, size: { auto: 1, leg_cm: 85.6, ground: 0 } },
    dst: { ...DATA.dst, size: { auto: 0.194, leg_cm: 437.3, ground: -10 } } };
  let q: Record<string, unknown> = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "",
    motion_scale: null, target_scale: null, auto_record: "" };
  const st = () => autoState("size", recordOf(q.auto_record), currentRecord(SR, q)).text;
  eq([effective(D, SR, q).scales, effective(D, SR, q).auto.size], [{ src: 1, dst: 0.194 }, { src: true, dst: true }], "空着：两侧按自动的系数");
  eq(st(), "自动（还没改过）", "尺寸还没执行：默认自动");
  eq(effective(D, ROLES, q).scales, { src: 1, dst: 1 }, "节点没有尺寸角色：不缩放");
  q = { ...q, ...autoPatch("size", D, SR, q)! };
  eq([q.motion_scale, q.target_scale, recordOf(q.auto_record)?.size], [1, 0.194, { motion: 1, target: 0.194 }], "自动尺寸 = 建议原样写进参数并记下");
  eq(st(), "已自动", "尺寸刚自动完");
  q = { ...q, target_scale: 1 };
  eq([st(), effective(D, SR, q).scales.dst], ["已自动 · 之后手动改了 1 个", 1], "手填 1：不缩放，状态行算一处改动");
  q = { ...q, target_scale: null };
  eq([effective(D, SR, q).scales.dst, st()], [0.194, "已自动 · 之后手动改了 1 个"], "执行过之后清空：仍是自动（空 = 自动）");
  eq(autoPatch("size", DATA, SR, {}), null, "服务端没给尺寸（腿没配上）：自动尺寸写不了");
  const fresh4: Record<string, unknown> = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "", auto_record: "" };
  const all4 = { ...fresh4, ...autoAllPatch(["mapping", "ignore", "pose", "size"], D, SR, fresh4, ruleNow(D, ""))! };
  eq([Object.keys(recordOf(all4.auto_record) ?? {}).sort(), all4.target_scale], [["ignore", "mapping", "pose", "size"], 0.194], "全部自动：四项都在，含尺寸");
  // 显示：绕地面原点缩放、脚底落到 y = 0（地面在 -10），轴一起乘
  eq(scaledPoint([110, 50, 4], D.dst, 0.5), [55, 30, 2], "点：p ↦ s·(p − (0, 地面, 0))");
  eq(scaledPoint([0, -10, 0], D.dst, 0.5), [0, 0, 0], "地面上的点落到 y = 0");
  eq(scaledWorld([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 110, 50, 4, 1]], D.dst, 0.5)[0], [0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 55, 30, 2, 1], "世界矩阵：轴乘系数、位置同点");
  eq(scaledWorld([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 3, 4, 5, 1]], D.dst, 1)[0][12], 3, "系数 1 原样");
  const far2 = { ...D, src: { ...D.src, before: { local: D.src.before!.local.map((m, i) => (i === 0 ? [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 300, 95, -800, 1] : m)) } } };
  eq(sideOffsets(far2, 0, { src: 0.5, dst: 1 }).src, [-150, 0, 400], "摆放按缩放后的锚点挪回原点");
}

// ---- 点「自动」之后服务端的建议变了（软件更新改了自动姿态、换了骨架）：状态行提示重新执行，不写「已自动」
{
  let q: Record<string, unknown> = { mapping: null, motion_pose: [], target_pose: [], motion_ignore: [], target_ignore: [], ignore_rule: "", auto_record: "" };
  q = { ...q, ...autoPatch("pose", DATA, ROLES, q)! };
  eq(autoState("pose", recordOf(q.auto_record), currentRecord(ROLES, q), undefined, suggestedRecord(DATA, ROLES)).text, "已自动", "建议没变：已自动");
  const moved = { ...DATA, src: { ...DATA.src, auto_pose: [{ joint: "LeftUpLeg", translate: [0, 0, 0], rotate: [0, 0, 5], scale: [1, 1, 1] }, { joint: "LeftLeg", translate: [0, 0, 0], rotate: [0, 9, 0], scale: [1, 1, 1] }] } };
  eq(autoState("pose", recordOf(q.auto_record), currentRecord(ROLES, q), undefined, suggestedRecord(moved, ROLES)),
    { text: "建议已更新（1 处不同）：点「自动姿态」按新的重新执行", tone: "stale" }, "建议多了一行：提示重新执行");
  q = { ...q, motion_pose: [] };
  eq(autoState("pose", recordOf(q.auto_record), currentRecord(ROLES, q), undefined, suggestedRecord(moved, ROLES)).text,
    "建议已更新（1 处不同）：点「自动姿态」按新的重新执行 · 之后手动改了 1 个", "建议变了又手动改过：两件都说");
  q = { ...q, ...autoPatch("pose", moved, ROLES, q)! };
  eq(autoState("pose", recordOf(q.auto_record), currentRecord(ROLES, q), undefined, suggestedRecord(moved, ROLES)).text, "已自动", "重新执行之后：已自动");
}

// ---- 按选中规则检查（解算器的固定骨架）：编辑时就说缺部位、节数过多
{
  const withRule = { ...DATA, rules: [...(DATA.rules ?? []), { id: "fixed1", label: "固定", ignore: {}, slots: { hips: 1, spine: 1 }, required: ["hips"] },
    { id: "needhead", label: "要头", ignore: {}, slots: { head: 1 }, required: ["head"] }] };
  const m0 = mergedOf(null, withRule);
  const twoSpine = { ...m0, rows: { ...m0.rows, spine: { part: "spine", src: ["Spine", "Spine1"], dst: m0.rows.spine?.dst ?? [] } } };
  eq(ruleProblems(m0, withRule, "fixed1", { src: [], dst: [] }).filter((t) => t.startsWith("源")).length, 0, "合规则：源这边没问题");
  eq(ruleProblems(twoSpine, withRule, "fixed1", { src: [], dst: [] }).some((t) => t.includes("源的脊柱配了 2 节")), true, "节数多于模型：说出来");
  eq(ruleProblems(twoSpine, withRule, "fixed1", { src: ["Spine1"], dst: [] }).some((t) => t.includes("源的脊柱配了")), false, "多的那节忽略了：不再报");
  eq(ruleProblems(m0, withRule, "needhead", { src: [], dst: [] }).some((t) => t.startsWith("源缺")), true, "缺必需部位（这副骨架没有头）：说出来");
  eq(ruleProblems(m0, withRule, "none", { src: [], dst: [] }), [], "规则没有固定骨架：不查");
}

// ---- 识别置信度：按推测的部位标在第一个关节上，三档；手动的、没分配的不标；没认的部位在树头说
{
  const rec = { threshold: 0.5, parts: {
    hips: { confidence: 0.95, evidence: ["按结构：两腿的分叉"] },
    spine: { confidence: 0.7, evidence: ["按结构：髋与胸之间的躯干"] },
    "l.thigh": { confidence: 0.55, evidence: ["按比例：大腿 / 小腿 = 1.02"] },
    "l.shin": { confidence: 0.9, evidence: [] },
    "r.shin": { confidence: 0.4, evidence: ["置信度 0.4 低于 0.5：不分配，留给手动"], assigned: false },
  } };
  const withRec = { ...DATA, dst: { ...DATA.dst, recognition: rec } };
  const m0 = mergedOf(null, withRec);
  const c = confidenceOf(m0, withRec, "dst");
  eq([...c.entries()].map(([n, v]) => `${n}:${v.part}:${v.level}`), ["pelvis:hips:high", "spine_01:spine:mid", "thigh_l:l.thigh:low", "calf_l:l.shin:high"],
    "置信度：链只标第一节，三档");
  eq(c.get("thigh_l")?.tip, "左大腿 · 识别置信度 0.55\n按比例：大腿 / 小腿 = 1.02", "悬停：数值与依据");
  eq(confidenceOf(m0, withRec, "src").size, 0, "这一侧没有识别结果：不标");
  const same = mergedOf([{ part: "l.thigh", src: ["LeftUpLeg"], dst: ["thigh_l"] }], withRec);
  eq(confidenceOf(same, withRec, "dst").has("thigh_l"), true, "写进参数、与推测相同（点过「自动对应」）：照样标");
  const manual = mergedOf([{ part: "l.thigh", src: ["LeftUpLeg"], dst: ["spine_02"] }], withRec);
  eq([...confidenceOf(manual, withRec, "dst").values()].some((v) => v.part === "l.thigh"), false, "手动改过的部位不标");
  eq(unsureParts(rec, withRec, m0, "dst"), { count: 1, tip: "识别没认、留给手动配的部位：\n右小腿：置信度 0.4 低于 0.5：不分配，留给手动" }, "没认的部位");
  const done = mergedOf([{ part: "r.shin", src: [], dst: ["thigh_r"] }], withRec);
  eq(unsureParts(rec, withRec, done, "dst").count, 0, "手动配上了：不再算没认");
}

console.log(`model/rigPair.ts：${count} 项都对`);
