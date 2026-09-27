import { useEffect, useMemo, useState } from "react";
import { fileKeyOf } from "../transfer/localProxy";
import { api } from "../api";
import { manifestOf } from "../transfer/frames";
import { keepOriginals, lookingFor, type Originals } from "../transfer/originals";
import { localPackage } from "../files/delivered";
import { dirHolding, listNames } from "../files/localDirs";
import { deliveredKey, framesMatchingPattern, framesOfNames, nameForFrame, readKey } from "../model/frameNames";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useLocalDirs } from "../state/localDirs";
import { useResults } from "../state/results";
import type { ViewItem } from "./plan";

/** 在使用者本机为当前画面所需的每份数据查找原件：读取类节点和输出节点的文件位于本机，
 * 预览直接解码本地文件，不使用压缩代理。
 *
 * 分为两部分，使用同一机制：
 * - 读取部分：读取节点读取的是用户自己的文件。若只识别本标签页中刚选择的文件，刷新后即会丢失
 *   （浏览器只在用户亲自选择时将文件交给页面），因此还识别已授权的目录（`files/localDirs.ts`），刷新后仍可识别。
 * - 输出部分：计算完成后交付已写入用户磁盘，双击「输出」查看的即为该文件。
 *
 * 找到后只做一件事：将各帧对应的文件登记到 `transfer/originals.ts`。
 * 取帧路径（`transfer/sources.ts serverFrames`）会自行查询。
 * 其余均不变：包说明、画面尺寸、时间线、下游计算、交付均以服务器数据为准，
 * 因此同一帧使用原件或代理时在屏幕上占据相同的矩形。
 *
 * 找不到不属于错误：静默回退到服务器代理，不报错、不提示，
 * 也不在视图上绘制新的通知控件。 */

/** 已登记数据的「钥匙」：其变化时取帧路径随之重算（`view/stageSources.ts` 的 memo 依赖）。 */
export type OriginalsKey = string;

interface Ask {
  // 本次查询的身份：只有身份变化时才重新查找。
  //
  // 唯一规则：答案所依赖的每一项都必须包含在身份中。
  // 遗漏任何一项，该项变化时不会重新查询，答案将一直停留在旧值上，既不报错也不会使测试失败。
  // 例如：交付刚到达时文件尚未写入磁盘，答案为「没有」；写入完成后（状态由「待取回」变为「已保存」）
  // 身份须随之变化；用户在「文件 · 本机目录」中指定目录后，也须重新查询。
  // 因此身份由下方两个纯函数拼接（由 `webui/tests/originals.test.ts` 按其验证）。
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
          where: `本机目录 ${dir.name}`,
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

/** 该节点图中该「输出」最近一次交付对应的计算（服务器的记录；没有时为 null）。
 *
 * 判据与页面其他位置同样严格（见 `files/deliver.ts` 的注释）：只认同一张节点图
 * （`Delivery.graph` 为节点图文件自身的身份），其他节点图以相同节点 id 交付的结果不计入。 */
async function latestRun(graphId: string, node: string): Promise<string | null> {
  const all = await api.deliveries.mine().catch(() => [] as Awaited<ReturnType<typeof api.deliveries.mine>>);
  const mine = all.filter((d) => d.node === node && (d.graph ?? "") === graphId);
  if (!mine.length) return null;
  const at = (d: (typeof mine)[number]) => Number((d as { created?: number }).created ?? 0);
  return mine.reduce((best, d) => (at(d) > at(best) ? d : best)).run;
}

/** 输出部分：该数据由哪个「输出设置」写出（`ViewItem.deliveredBy`），
 * 在交付包中对应一个子文件夹，名称为其「名字」参数
 * （`lab2shot/transfer/deliveries.py make`：`<名字>/<文件>`）。
 *
 * 子文件夹的实际名称可能带有机器学习标记（`_ML_Lab2Shot_<项目>`，`nodes/output.py marked`：
 * 用于标明由模型生成）。该后缀由服务器添加，页面无法计算，
 * 因此此处按前缀识别：等于「名字」，或以「名字_ML_Lab2Shot」开头。 */
function deliveredAsk(outputNodeId: string, settingsNodeId: string, fp: string, dirs: number): Ask | null {
  const { nodes, graphId } = useCookInputs.getState();
  const settings = nodes[settingsNodeId];
  const def = settings && getNodeDefs()[settings.typeId];
  if (!settings || !def) return null;
  const unique = def.params.find((q) => q.unique); // nodes/output.py：唯一声明 unique 的参数即「名字」
  const name = unique ? String(settings.params[unique.name] ?? "") : "";
  if (!name) return null;
  // 由哪一次计算交付：本标签页见过的那一次（`state/results.ts` 的 `deliveries`）。
  // 刷新后该记录即丢失：它只存在于本标签页中，且只记录「尚未取回」的交付（`checkPending`）；
  // 交付早已写入用户磁盘、状态为「已保存」时，刷新后不会保留任何条目，画面将变回代理。
  // 因此此处不强制依赖标签页中的记录：没有时查询服务器自身的记录（`/api/deliveries`，
  // 一次请求、数百字节），按「同一张节点图的同一个「输出」」选择最近的一次。
  const seen = useResults.getState().deliveryFor(graphId, outputNodeId);
  const saveHandle = nodes[outputNodeId]?.saveTo?.handle;
  return {
    // 交付的进度也须计入该身份，不能只记录是哪一次计算：交付事件到达时即会发起一次查询，
    // 此时服务器的记录为「待取回」，浏览器尚未将包写入用户的文件夹，「第一帧能否取得」的答案为否，
    // 该数据即被记为「没有原件」。待文件实际写入后（记录变为「已保存」），
    // 若身份不随之变化，该查询不会再次发起，画面将一直停留在服务器代理上，且没有任何提示。
    //
    // 「已保存」并非推测：`files/deliver.ts` 在包写入完成后才记录此状态
    // （`saveDelivery` 成功 → `noteDelivery(now, "saved")`），因此它即代表文件已在用户磁盘上。
    // （写入成功但「通知服务器」一步失败的情况（`W-DELIVER-UNREPORTED`）停留在「待取回」，
    //  该查询须等到下次刷新才能生效；届时 `seen` 为空，经由服务器的记录，同样可以找到。）
    key: deliveredKey({ fp, graphId, node: outputNodeId, name, handle: saveHandle, run: seen?.run, state: seen?.state, dirs }),
    fp,
    find: async () => {
      const run = seen?.run ?? (await latestRun(graphId, outputNodeId));
      if (!run) return null;
      const d = await api.deliveries.get({ run, node: outputNodeId }).catch(() => null);
      if (!d) return null;
      const mark = `${name}_ML_Lab2Shot`;
      const mine = d.files.filter((rel) => {
        const top = rel.slice(0, rel.indexOf("/"));
        return top === name || top.startsWith(mark);
      });
      if (!mine.length) return null;
      const meta = (await manifestOf(fp).catch(() => null))?.meta;
      const still = Array.isArray(meta?.frames) ? (meta.frames as number[])[0] : undefined;
      const byFrame = framesOfNames(mine, still);
      if (!byFrame.size) return null;
      const pack = await localPackage(d, saveHandle);
      if (!pack) return null;
      // 包中记录了这些帧，但磁盘上的文件可能不完整：第一帧能够取得才计入，否则整份不计
      const first = [...byFrame.keys()].sort((a, b) => a - b)[0];
      if (!(await pack.fileAt(byFrame.get(first)!).catch(() => null))) return null;
      const keys = new Map<number, string>();
      await Promise.all([...byFrame].map(async ([f, rel]) => {
        const file = await pack.fileAt(rel).catch(() => null);
        if (file) keys.set(f, fileKeyOf(file));
      }));
      return {
        frames: [...byFrame.keys()],
        keys,
        space: "",
        where: pack.where,
        // 整段按同一文件名查询色彩空间（`transfer/sources.ts fromFile` 的 `rules`）。
        // 取 basename，与 `File.name` 为同一字符串：只查询一次，结果相同
        name: byFrame.get(first)!.split("/").pop(),
        fileOf: async (frame) => {
          const rel = byFrame.get(frame);
          return rel ? await pack.fileAt(rel).catch(() => null) : null;
        },
      };
    },
  };
}

/** 当前画面需要查找的数据（同一个包只查找一次）。 */
function asksFor(outputNodeId: string, items: readonly ViewItem[], dirs: number): Ask[] {
  const out = new Map<string, Ask>();
  for (const it of items) {
    if (!it.fp || out.has(it.fp)) continue;
    const ask = it.deliveredBy ? deliveredAsk(outputNodeId, it.deliveredBy, it.fp, dirs)
                               : readAsk(it.nodeId, it.fp, dirs);
    if (ask) out.set(it.fp, ask);
  }
  return [...out.values()];
}

/** 查找并在找到后登记。返回的「钥匙」变化时，取帧路径随之重算。 */
export function useOriginals(outputNodeId: string | null, items: readonly ViewItem[]): OriginalsKey {
  // 交付写入完成后、该「输出」计算过一次后，均须重新查询：双击「输出」时它可能尚未计算完成，
  // 此时查询的答案为「没有」；交付实际写入用户磁盘后，`items` 不会发生任何变化（上游的包早已计算完成），
  // 若当前画面没有任何变化，该查询将不会再次发起，画面将一直停留在服务器代理上，且没有任何提示。
  // 因此此处显式跟踪「已交付的内容」与「该输出节点是否已计算」，二者即为该查询的输入。
  // 另外加上本标签页记录的交付（`deliveries`）：任一先到达均可重新发起查询。
  // 用户已授权本机目录的修改次数（`state/localDirs.ts`）：同为该查询的输入之一
  const dirs = useLocalDirs((s) => s.version);
  const delivered = useResults((s) =>
    [s.byNode[outputNodeId ?? ""]?.status ?? "",
     Object.entries(s.deliveries).map(([at, d]) => `${at}=${d.run}:${d.state}`).sort().join("|")].join("/"));
  const asks = useMemo(
    () => (outputNodeId ? asksFor(outputNodeId, items, dirs) : []),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [outputNodeId, items.map((it) => `${it.key}:${it.fp ?? ""}:${it.deliveredBy ?? ""}`).join("|"), delivered, dirs],
  );
  const wanted = asks.map((a) => a.key).join("|");
  const [found, setFound] = useState<OriginalsKey>("");
  useEffect(() => {
    let alive = true;
    // 先声明「正在查找」：查找原件需两三百毫秒（IndexedDB、请求权限、列目录），
    // 在此期间后台路径暂不取回整段代理（见 `transfer/originals.ts lookingFor` 的注释）
    for (const a of asks) lookingFor(a.fp);
    void Promise.all(
      asks.map(async (a) => {
        const one = await a.find().catch(() => null);
        return { fp: a.fp, key: a.key, one };
      }),
    ).then((all) => {
      // 登记必须执行，即使组件已卸载：否则「正在查找」状态将一直保留，
      // 后台取回被其永久阻止（`stillLooking`）。组件卸载时唯一不执行的是 `setFound`，它仅供当前画面使用
      const got: string[] = [];
      for (const { fp, key, one } of all) {
        keepOriginals(fp, one);
        // 查找结束即生效，无论是否找到：未找到时也须使当前画面重算，
        // 否则被「正在查找」阻止的后台取回将无法再启动
        got.push(`${key}@${one ? one.frames.length : "无"}`);
      }
      const next = got.join("|");
      if (alive) setFound((had) => (had === next ? had : next));
    });
    return () => void (alive = false);
  }, [wanted]); // eslint-disable-line react-hooks/exhaustive-deps
  return found;
}
