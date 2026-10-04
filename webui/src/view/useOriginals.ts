import { useEffect, useMemo, useState } from "react";
import { fileKeyOf } from "../transfer/localProxy";
import { manifestOf } from "../transfer/frames";
import { keepOriginals, lookingFor, type Originals } from "../transfer/originals";
import { dirHolding, listNames } from "../files/localDirs";
import { framesMatchingPattern, nameForFrame, readKey } from "../model/frameNames";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useLocalDirs } from "../state/localDirs";
import type { ViewItem } from "./plan";
import { t } from "../i18n/t";

/** 在使用者本机为当前画面所需的每份数据查找原件：读取类节点读的是用户自己的文件，
 * 预览直接解码本地文件，不使用压缩代理。
 *
 * 读取节点读取的是用户自己的文件。若只识别本标签页中刚选择的文件，刷新后即会丢失
 * （浏览器只在用户亲自选择时将文件交给页面），因此还识别已授权的目录（`files/localDirs.ts`），刷新后仍可识别。
 * 看「输出」显示的是服务器上接进它的数据的代理；要看无损的结果，下载后在 DCC 里看。
 *
 * 找到后只做一件事：将各帧对应的文件登记到 `transfer/originals.ts`。
 * 取帧路径（`transfer/sources.ts serverFrames`）会自行查询。
 * 其余均不变：包说明、画面尺寸、时间线、下游计算、交付均以服务器数据为准，
 * 因此同一帧使用原件或代理时在屏幕上占据相同的矩形。
 *
 * 找不到不属于错误：静默回退到服务器代理，不报错、不提示，
 * 也不在视图上绘制新的通知控件。 */

/** 已登记数据的「钥匙」：其变化时取帧路径随之重算（`view/stageSources.ts` 的 memo 依赖）。 */
type OriginalsKey = string;

interface Ask {
  // 本次查询的身份：只有身份变化时才重新查找。
  //
  // 唯一规则：答案所依赖的每一项都必须包含在身份中。
  // 遗漏任何一项，该项变化时不会重新查询，答案将一直停留在旧值上，且不会报错。
  // 例如：用户在「文件 · 本机目录」中指定目录后，须重新查询。
  // 因此身份由纯函数拼接（`model/frameNames.ts readKey`）。
  key: string;
  fp: string;
  find: () => Promise<Originals | null>;
}

/** 序列的文件名模式（参数中存储为 `upload:<id>/plate.####.exr`）。 */
const patternOf = (ref: string) => ref.slice(ref.indexOf("/") + 1);

/** 读取部分：该节点各文件参数读取的内容及其所在文件夹。 */
function readAsk(nodeId: string, fp: string, dirs: number): Ask | null {
  const { nodes } = useCookInputs.getState();
  const node = nodes[nodeId];
  const def = node && getNodeDefs()[node.typeId];
  if (!node || !def) return null;
  for (const p of def.params) {
    if (p.widget !== "file" && p.widget !== "sequence") continue;
    const ref = String(node.params[p.name] ?? "");
    if (!ref) continue;
    const pattern = patternOf(ref);
    const pick = node.picked?.[p.name];
    const folder = pick?.folder ?? "";
    const space = String(node.params.colorspace ?? "");
    // 帧号表优先使用本机记录，没有时才回退到服务器的包说明。
    //
    // 包说明只有计算后才存在：若只依赖 `manifestOf(fp)`，读取节点查看用户自己的文件时须先在服务器上计算一次，
    // 双击读取序列节点将无法看到无损原图。
    // 选择文件时浏览器即已知道帧范围（`ui.picked` 的 first / last，已保存到节点图文件中，
    // `graph/document.ts`），因此刷新后仍然存在；而本标签页中的两个模块级 Map
    // （`transfer/local.ts`）刷新后即失效。
    //
    // 跳号不依据这两个数推算：实际存在的帧由目录中的文件名决定（`framesMatchingPattern`），
    // first / last 仅用于：① 确定目录；② 排除范围外的文件名（同一目录中可能存放其他镜头）。
    const span = pick && pick.ref === ref && pick.first != null && pick.last != null
      ? ([pick.first, pick.last] as const) : null;
    return {
      // 答案所依赖的每一项都必须包含在身份中：帧号范围改变时，找到的是另一份数据
      key: readKey({ fp, ref, folder, dirs, span: span ? `${span[0]}-${span[1]}` : "", space }),
      fp,
      find: async () => {
        // 单张图（模式中不含 `#`）不在此推断帧号：其帧号只有包说明知道，
        // 因此该情况仍回退到服务器数据（`framesMatchingPattern` 对不含 `#` 的模式返回空表）
        const known = span ? null : (await manifestOf(fp).catch(() => null))?.meta;
        const listed = Array.isArray(known?.frames) ? (known.frames as number[]) : [];
        const first = span ? span[0] : listed[0];
        if (first === undefined) return null;
        // 目录只需列出一次（listNames）：逐帧查询是否存在需要数百次系统调用
        const dir = await dirHolding(folder, nameForFrame(pattern, first));
        if (!dir) return null;
        const have = await listNames(dir);
        const inside = span ? (f: number) => f >= span[0] && f <= span[1] : (f: number) => listed.includes(f);
        const found = framesMatchingPattern(have, pattern).filter(inside);
        // 单张图路径（模式中不含 `#`）：文件名即为其本身，帧号取自包说明
        const mine = found.length ? found : listed.filter((f) => have.has(nameForFrame(pattern, f)));
        if (!mine.length) return null;
        // 各帧的文件键（名称、大小、修改时间；只读取元数据，不读取字节）：账本据此检查代理是否存在
        const keys = new Map<number, string>();
        await Promise.all(mine.map(async (f) => {
          const file = await dir.getFileHandle(nameForFrame(pattern, f)).then((h) => h.getFile()).catch(() => null);
          if (file) keys.set(f, fileKeyOf(file));
        }));
        return {
          frames: mine,
          keys,
          space,
          where: t("ui.view.local_folder", { name: dir.name }),
          // 整段按同一文件名查询色彩空间（`transfer/sources.ts fromFile` 的 `rules`）：一段序列只有一个
          name: nameForFrame(pattern, mine[0]),
          fileOf: async (frame) =>
            await dir.getFileHandle(nameForFrame(pattern, frame)).then((h) => h.getFile()).catch(() => null),
        };
      },
    };
  }
  return null;
}

/** 当前画面需要查找的数据（同一个包只查找一次）。 */
function asksFor(items: readonly ViewItem[], dirs: number): Ask[] {
  const out = new Map<string, Ask>();
  for (const it of items) {
    if (!it.fp || out.has(it.fp)) continue;
    const ask = readAsk(it.nodeId, it.fp, dirs);
    if (ask) out.set(it.fp, ask);
  }
  return [...out.values()];
}

/** 查找并在找到后登记。返回的「钥匙」变化时，取帧路径随之重算。 */
export function useOriginals(outputNodeId: string | null, items: readonly ViewItem[]): OriginalsKey {
  // 用户已授权本机目录的修改次数（`state/localDirs.ts`）：该查询的输入之一，指定目录后重新查询
  const dirs = useLocalDirs((s) => s.version);
  const asks = useMemo(
    () => (outputNodeId ? asksFor(items, dirs) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [outputNodeId, items.map((it) => `${it.key}:${it.fp ?? ""}`).join("|"), dirs],
  );
  const wanted = asks.map((a) => a.key).join("|");
  const [found, setFound] = useState<OriginalsKey>("");
  useEffect(() => {
    let alive = true;
    // 先声明「正在查找」：查找原件需两三百毫秒（IndexedDB、请求权限、列目录），
    // 在此期间后台路径暂不取回整段代理（见 `transfer/originals.ts lookingFor` 的注释）
    const tickets = asks.map((a) => lookingFor(a.fp));
    void Promise.all(
      asks.map(async (a, i) => {
        const one = await a.find().catch(() => null);
        return { fp: a.fp, key: a.key, one, asked: tickets[i] };
      }),
    ).then((all) => {
      // 登记必须执行，即使组件已卸载：否则「正在查找」状态将一直保留，
      // 后台取回被其永久阻止（`stillLooking`）。组件卸载时唯一不执行的是 `setFound`，它仅供当前画面使用
      const got: string[] = [];
      for (const { fp, key, one, asked } of all) {
        keepOriginals(fp, one, asked);
        // 查找结束即生效，无论是否找到：未找到时也须使当前画面重算，
        // 否则被「正在查找」阻止的后台取回将无法再启动
        got.push(`${key}@${one ? one.frames.length : "none"}`); // a dependency key, not words
      }
      const next = got.join("|");
      if (alive) setFound((had) => (had === next ? had : next));
    });
    return () => void (alive = false);
  }, [wanted]); // eslint-disable-line react-hooks/exhaustive-deps
  return found;
}
