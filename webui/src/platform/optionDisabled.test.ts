/** platform/conditions.ts optionDisabled（公开下拉一项的 Disable When + 为什么）的测试：node src/platform/optionDisabled.test.ts。 */
import { optionDisabled } from "./conditions.ts";

let count = 0;
function eq(got: unknown, want: unknown, what: string): void {
  count++;
  if (JSON.stringify(got) !== JSON.stringify(want)) throw new Error(`${what}\n  得到 ${JSON.stringify(got)}\n  应为 ${JSON.stringify(want)}`);
}
const o = { disable_when: "ref_bvh_path and not ref_fbx_path", disable_why: "要网格" };
eq(optionDisabled(o, { ref_bvh_path: "up:a.bvh", ref_fbx_path: "" }), "要网格", "只传了 BVH：置灰并说为什么");
eq(optionDisabled(o, { ref_bvh_path: "", ref_fbx_path: "" }), null, "都没传：能选");
eq(optionDisabled(o, { ref_bvh_path: "up:a.bvh", ref_fbx_path: "up:b.fbx" }), null, "有 FBX：能选");
eq(optionDisabled({}, { ref_bvh_path: "x" }), null, "没写条件：能选");
eq(optionDisabled({ disable_when: "x == 1" }, { x: 1 }), "x == 1", "没写为什么时退回条件本身");
console.log(`optionDisabled：${count} 项都对`);
