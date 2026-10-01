/** 「对应关系」编辑器中间那张人形图：每个部位槽摆在哪（和 Maya HumanIK Character Definition 的人形图一样，人面向
 * 使用者，人物的左侧在画面右侧——与项目火柴人 view/figure2d.ts 同一约定）。这里只有位置，部位 id 来自
 * lab2shot/data/joints.py PARTS，中文名、区域、是不是链、是不是必需都由服务端给（RigChoice.parts），本文件不写部位名。
 *
 * 坐标是人形图框里的百分比（x 向右，y 向下），槽以该点为中心。手指放在「手指」那一页（HumanIK 也把手放在单独的
 * 手部视图里），主图不挤。 */

export type Page = "body" | "hands";

const side = (s: "l" | "r", x: number) => (s === "l" ? 100 - x : x); // 人物的左边在画面右边

const BODY: Record<string, [number, number]> = {
  head: [50, 6],
  jaw: [66, 12],
  "r.eye": [34, 4],
  "l.eye": [66, 4],
  neck: [50, 15],
  chest: [50, 26],
  spine: [50, 36],
  hips: [50, 47],
};
for (const s of ["l", "r"] as const) {
  BODY[`${s}.clavicle`] = [side(s, 33), 21];
  BODY[`${s}.upperarm`] = [side(s, 24), 30];
  BODY[`${s}.forearm`] = [side(s, 15), 39];
  BODY[`${s}.hand`] = [side(s, 9), 48];
  BODY[`${s}.thigh`] = [side(s, 40), 58];
  BODY[`${s}.shin`] = [side(s, 40), 71];
  BODY[`${s}.foot`] = [side(s, 40), 83];
  BODY[`${s}.toe`] = [side(s, 40), 94];
}

const FINGERS = ["thumb", "index", "middle", "ring", "pinky"];
const HANDS: Record<string, [number, number]> = {};
for (const s of ["l", "r"] as const) {
  HANDS[`${s}.hand`] = [side(s, 25), 12];
  FINGERS.forEach((f, k) => (HANDS[`${s}.${f}`] = [side(s, 25), 28 + k * 14]));
}

export const PLACES: Record<Page, Record<string, [number, number]>> = { body: BODY, hands: HANDS };

/** 人形图背后的火柴人：哪些槽之间连一根线（只是示意，按槽的位置画）。 */
export const LINKS: Record<Page, [string, string][]> = {
  body: [
    ["head", "neck"], ["neck", "chest"], ["chest", "spine"], ["spine", "hips"],
    ...(["l", "r"] as const).flatMap((s): [string, string][] => [
      ["chest", `${s}.clavicle`], [`${s}.clavicle`, `${s}.upperarm`], [`${s}.upperarm`, `${s}.forearm`], [`${s}.forearm`, `${s}.hand`],
      ["hips", `${s}.thigh`], [`${s}.thigh`, `${s}.shin`], [`${s}.shin`, `${s}.foot`], [`${s}.foot`, `${s}.toe`],
    ]),
  ],
  hands: (["l", "r"] as const).flatMap((s) => FINGERS.map((f): [string, string] => [`${s}.hand`, `${s}.${f}`])),
};

/** 哪一页放得下这个部位（点树里一个手指关节时翻到「手指」页）。 */
export const pageOf = (part: string): Page => (part in BODY ? "body" : "hands");
