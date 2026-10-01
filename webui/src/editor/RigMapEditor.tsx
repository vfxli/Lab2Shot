// 「对应关系」编辑器的窗里内容（editor/RigMap.tsx 登记，按需加载：它引入三维舞台 view/HandleStage.tsx，three.js 只随它
// 进来，不进主包）。怎么配、写回什么见 editor/RigMap.tsx 的说明。

import { useEffect, useMemo, useState } from "react";
import type { RigChoice, RigPart, RigSide } from "../api";
import { Button, Segmented } from "../ui/Button";
import { Menu, type MenuRow } from "../ui/Menu";
import { SheetFoot, useDraft, useSheetLock, Writes, type SheetEditorProps } from "./ParamSheet";
import { LINKS, PLACES, pageOf, type Page } from "./rigMap/figure";
import { autoRow, byDepth, COLS, draftOf, partOf, pathBetween, problems, stateOf, treeOf, valueOf, type Col, type Draft } from "./rigMap/rules";
import { HandleStage } from "../view/HandleStage";
import { usePoseSelection } from "../view/skeletonPose";
import { forward, rowsOf as poseRows, type PoseRow } from "../model/skeletonPose";
import { useHandleView } from "../state/handleView";
import { useCookInputs } from "../state/cookInputs";
import { same, type Json } from "../model/graphPatch";
import { getNodeDefs } from "../state/catalog";
import { useResults } from "../state/results";
import { Switch } from "../ui/Button";
import { SkeletonTree } from "./rigMap/SkeletonTree";

import { handleOf, itemOf, nounOf, onFigure, posesOf, rigOf, slotWord } from "./RigMap";

const LIST = 400; // px：两侧树 / 3D 的高度

/** 窗里要的两副骨架（NodeDef.choices 的 rig）没有时只说一句：窗框架只把能用的数据交进来（ParamControls.tsx
 * OpenSheet），这里不断言它一定在。 */
export default function RigMapEditor(props: SheetEditorProps) {
  const rig = rigOf(props.p, props.choices);
  if (!rig) return <div className="psheet-none">{props.choices?.[""]?.empty ?? "两副骨架还没读到"}</div>;
  return <RigMapBody {...props} rig={rig} />;
}

function RigMapBody({ value, base, set, cancel, nodeId, rig }: SheetEditorProps & { rig: RigChoice }) {
  // 左右两块三维视图画的是节点声明的「骨架姿势」手柄（舞台按状态回复 handle_data 画）：弹窗开着时手柄数据要这个节点的
  // （state/handleView.ts dialog → 状态请求的 handle_node），不改显示节点、不记撤销
  useEffect(() => {
    useHandleView.getState().setDialog(nodeId);
    return () => useHandleView.getState().setDialog(null);
  }, [nodeId]);
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId);
  const handleAt = { src: handleOf(rig.src), dst: handleOf(rig.dst) };
  // 姿势修正是弹窗草稿的一部分：确定时与对应关系一起写回（一步撤销），取消 / Esc 不写；弹窗里「撤销」退一步修正
  const poseParams = posesOf(nodeId, rig); // 窗的基准也按这一份比（RigMap.tsx also）
  const [poseDraft, setPoseDraft] = useDraft<Record<string, PoseRow[]>>(() => Object.fromEntries(poseParams.map((q) => [q, poseRows(base[q])])));
  const [poseUndo, setPoseUndo] = useDraft<Record<string, PoseRow[]>[]>([]);
  const [skin, setSkin] = useState(false);
  const writePose = (param: string, rows: PoseRow[]) => {
    setPoseUndo((u) => [...u.slice(-49), poseDraft]);
    setPoseDraft((d) => ({ ...d, [param]: rows }));
  };
  // 两侧关节的基准位置（骨长比例条用）：这一侧骨架手柄的 handle_data，基准局部经 FK（不带修正）
  const handleData = useResults((s) => s.reply?.handle_data);
  const bases = useMemo(() => Object.fromEntries(COLS.map((col) => {
    const h = handleAt[col];
    const d = h !== null && handleData?.node === nodeId ? handleData.handles?.[String(h)] : undefined;
    if (!d) return [col, null];
    const { world } = forward(d, []);
    return [col, new Map(d.names.map((n, i) => [n, [world[i][12], world[i][13], world[i][14]] as const]))];
  })) as Record<Col, Map<string, readonly number[]> | null>, [handleData, nodeId, handleAt.src, handleAt.dst]); // eslint-disable-line react-hooks/exhaustive-deps
  const placeOf = (col: Col): ((n: string) => readonly number[] | undefined) | null => {
    const at = bases[col];
    return at ? (n) => at.get(n) : null;
  };
  // 与窗的基准比（ParamSheet.tsx SheetWindow）：没碰过的姿势不写，别处改过的新姿势不被旧的盖回
  const posesChanged = poseParams.some((q) => !same((poseDraft[q] ?? []) as unknown as Json, poseRows(base[q]) as unknown as Json));
  const lock = useSheetLock();
  const [draft, setDraft] = useDraft<Draft>(() => draftOf(value, rig)); // 草稿：只读时改不动（ParamSheet.tsx useDraft）
  const [armed, setArmed] = useState<string | null>(null); // 蓝框的槽：在骨架里点关节就配给它
  const [picked, setPicked] = useState<Record<Col, string | null>>({ src: null, dst: null });
  const [page, setPage] = useState<Page>("body");
  const [menu, setMenu] = useState<{ x: number; y: number; part: string } | null>(null);
  const [views, setViews] = useState<Record<Col, "tree" | "3d">>({ src: "tree", dst: "tree" });
  // 三维视图里点中的关节也当作这一侧「选中的关节」（右键槽「指定选中的关节」用）
  const posePicked = usePoseSelection((s) => s.by);
  useEffect(() => {
    for (const col of COLS) {
      const at = posePicked[`rigmap-${col}`];
      if (at && at.handle === handleAt[col]) setPicked((was) => (was[col] === at.joint ? was : { ...was, [col]: at.joint }));
    }
  }, [posePicked]); // eslint-disable-line react-hooks/exhaustive-deps
  const trees = useMemo(() => ({ src: treeOf(rig.src), dst: treeOf(rig.dst) }), [rig]);
  const bad = useMemo(() => problems(draft, rig), [draft, rig]);
  const owners = useMemo(() => ({ src: partOf(draft, "src"), dst: partOf(draft, "dst") }), [draft]);
  const parts = useMemo(() => new Map(rig.parts.map((q) => [q.id, q] as [string, RigPart])), [rig]);
  const state = (id: string) => (parts.get(id) ? stateOf(parts.get(id)!, draft, bad) : "none");

  const edit = (part: string, col: Col, joints: string[]) =>
    setDraft((d) => ({ rows: { ...d.rows, [part]: { ...d.rows[part], part, [col]: joints } }, manual: new Set(d.manual).add(part) }));
  const restore = (part: string) =>
    setDraft((d) => {
      const manual = new Set(d.manual);
      manual.delete(part);
      return { rows: { ...d.rows, [part]: autoRow(part, rig) }, manual };
    });
  const unpair = (part: string) =>
    setDraft((d) => ({
      rows: { ...d.rows, [part]: { part, src: [], dst: rig.dst.fixed ? [...(d.rows[part]?.dst ?? [])] : [] } },
      manual: new Set(d.manual).add(part),
    }));

  /** 在一侧点了一个关节：记下它（右键槽「指定选中的关节」用）；有蓝框的槽就配给它。 */
  const pick = (col: Col, name: string, e: React.MouseEvent) => {
    setPicked((was) => ({ ...was, [col]: name }));
    if (rig[col].fixed) return;
    if (!armed) {
      const owner = owners[col].get(name);
      if (owner) setPage(pageOf(owner));
      return;
    }
    const chain = parts.get(armed)?.chain;
    const had = draft.rows[armed]?.[col] ?? [];
    if (chain && e.shiftKey && had.length) {
      const path = pathBetween(had[0], name, rig[col], trees[col]);
      if (path) return edit(armed, col, path);
    }
    if (chain && (e.ctrlKey || e.metaKey)) {
      const next = had.includes(name) ? had.filter((n) => n !== name) : byDepth([...had, name], trees[col]);
      return edit(armed, col, next);
    }
    edit(armed, col, [name]);
  };

  const assignPicked = (part: string) => {
    for (const col of COLS) {
      const n = picked[col];
      if (n && !rig[col].fixed) edit(part, col, [n]);
    }
  };

  const slotMenu = (part: string): MenuRow[] => {
    const q = parts.get(part)!;
    const pickedSaid = COLS.filter((c) => picked[c] && !rig[c].fixed).map((c) => `${rig[c].label} ${picked[c]}`).join("，");
    // 菜单行都是改草稿的：改不了时一律置灰，提示写原因（ParamSheet.tsx useSheetLock）
    const rows: MenuRow[] = [
      { key: "assign", label: "指定选中的关节", desc: pickedSaid || undefined, off: !pickedSaid,
        tip: pickedSaid ? `把两边列表里选中的配给「${q.label}」` : "先在左右两边的列表里点一个", run: () => assignPicked(part) },
      { key: "auto", label: "恢复自动", off: !draft.manual.has(part), tip: "这个部位不再手动指定：每次计算按关节名和层级重新推测", run: () => restore(part) },
      { key: "none", label: "此部位不配", tip: "明确不配这个部位：它跟着父关节走、保持静止姿势的朝向", run: () => unpair(part) },
      ...COLS.filter((c) => !rig[c].fixed).map((c): MenuRow => ({
        key: `clear-${c}`, label: `清空${rig[c].label}这一栏`, tip: `只清掉${rig[c].label}这一边`,
        off: !draft.rows[part]?.[c].length, run: () => edit(part, c, []),
      })),
    ];
    return lock ? rows.map((r) => ({ ...r, off: true, tip: lock })) : rows;
  };

  const required = rig.parts.filter((q) => q.required);
  const requiredOk = required.filter((q) => state(q.id) === "ok").length;
  const both = rig.parts.filter((q) => draft.rows[q.id]?.src.length && draft.rows[q.id]?.dst.length).length;
  const current = armed ? parts.get(armed) : undefined;

  const side = (col: Col) => {
    const s: RigSide = rig[col];
    const active = armed ? draft.rows[armed]?.[col] ?? [] : [];
    if (s.fixed) return <FixedSide side={s} rig={rig} armed={armed} state={state} onArm={(id) => (setArmed(id), setPage(pageOf(id)))} />;
    const tagOf = (name: string) => {
      const owner = owners[col].get(name);
      return owner ? { label: parts.get(owner)?.label ?? owner, state: state(owner) } : undefined;
    };
    return (
      <div className="rmap-side">
        <div className="rmap-side-head">
          <b>{s.label}{nounOf(s)}</b>
          {s.pose && <span className="rmap-dim" data-tip="3D 里画的就是重定向对齐用的这个姿势（「动作静止姿势」「目标静止姿势」「基准姿势摆正」）">{s.pose}</span>}
          <span className="rmap-dim" data-user-data data-tip={s.path}>{s.names.length} {itemOf(s)}</span>
          <span style={{ flex: 1 }} />
          {handleAt[col] !== null && (
            <Segmented label={`${s.label}${nounOf(s)}的显示`} value={views[col]} onChange={(v) => setViews((was) => ({ ...was, [col]: v }))}
              options={[{ value: "tree", label: "树", tip: "按层级列出全部关节" }, { value: "3d", label: "3D", tip: "基准姿势的骨架：点关节选中，拖手柄或填数修正这一侧的初始姿势；另一侧半透明叠着对照" }]} />
          )}
        </div>
        {views[col] === "tree" || handleAt[col] === null ? (
          <SkeletonTree side={s} tree={trees[col]} height={LIST} tagOf={tagOf} active={active} picked={picked[col]}
            onPick={(n, e) => pick(col, n, e)} />
        ) : (
          <HandleStage node={nodeId} handle={handleAt[col]!} height={LIST} slot={`rigmap-${col}`} skin={skin}
            reference={handleAt[other(col)] !== null ? { node: nodeId, handle: handleAt[other(col)] } : null}
            values={poseDraft} write={writePose} />
        )}
      </div>
    );
  };

  return (
    <div className="rmap" onContextMenu={(e) => e.preventDefault()}>
      <div className="rmap-head">
        <span className="rmap-dim" data-user-data>{rig.src.label}：{rig.src.path ?? ""} · {rig.src.names.length} {itemOf(rig.src)}</span>
        <span className="rmap-dim" data-user-data>{rig.dst.label}：{rig.dst.path ?? (rig.dst.fixed ? "节点固定" : "")} · {rig.dst.names.length} {itemOf(rig.dst)}</span>
        <span style={{ flex: 1 }} />
        {required.length > 0 && <span className={`rmap-lamp${requiredOk === required.length ? " ok" : " bad"}`}
          data-tip={required.map((q) => `${q.label}：${bad[q.id] ?? "配好了"}`).join("\n")}>
          {required.map((q) => <i key={q.id} className={state(q.id)} />)}
          必需 {requiredOk}/{required.length}
        </span>}
        <span className="rmap-dim">已配 {both} 个{slotWord(rig)}</span>
        <Writes>
        <Button tip={`所有${slotWord(rig)}都回到按名字${onFigure(rig) ? "和层级" : ""}推测（参数变回空值）`} tone="ghost" onClick={() => setDraft(draftOf(null, rig))}>
          全部重新自动
        </Button>
        <Button tip={`所有${slotWord(rig)}都写成「不配」，从零开始配`} tone="ghost"
          onClick={() => setDraft({ rows: Object.fromEntries(rig.parts.map((q) => [q.id, { part: q.id, src: [], dst: rig.dst.fixed ? [...(rig.dst.parts?.[q.id] ?? [])] : [] }])), manual: new Set(rig.parts.map((q) => q.id)) })}>
          清空
        </Button>
        </Writes>
      </div>
      <div className="rmap-grid">
        {side("src")}
        <div className="rmap-figure">
          {onFigure(rig) ? (
            <>
              <Segmented label="人形图" value={page} onChange={setPage} layout="rmap-pages"
                options={[{ value: "body", label: "身体", tip: "躯干、四肢、脸" }, { value: "hands", label: "手指", tip: "两只手的五根手指" }]} />
              <Figure page={page} rig={rig} draft={draft} bad={bad} armed={armed} state={state}
                onArm={(id) => setArmed((was) => (was === id ? null : id))}
                onMenu={(id, x, y) => (setArmed(id), setMenu({ x, y, part: id }))} />
            </>
          ) : (
            <SlotGrid rig={rig} draft={draft} bad={bad} armed={armed} state={state}
              onArm={(id) => setArmed((was) => (was === id ? null : id))}
              onMenu={(id, x, y) => (setArmed(id), setMenu({ x, y, part: id }))} />
          )}
        </div>
        {side("dst")}
      </div>
      <div className="rmap-part">
        {current ? (
          <>
            <div className="rmap-part-title">
              <b>{current.label}</b>
              <span className="rmap-dim">{current.chain ? "链，可以多节：点第一节，Shift+点最后一节选一整串，Ctrl+点加减一节" : onFigure(rig) ? "一个关节" : "一对一"}</span>
              <span className="rmap-dim">{draft.manual.has(current.id) ? "手动" : "自动"}</span>
              <span style={{ flex: 1 }} />
              <Writes>
              <Button tip={`这个${slotWord(rig)}不再手动指定：每次计算按名字重新推测`} tone="ghost" disabled={!draft.manual.has(current.id)}
                onClick={() => restore(current.id)}>
                恢复自动
              </Button>
              <Button tip={onFigure(rig) ? "明确不配这个部位：它跟着父关节走、保持静止姿势的朝向" : "明确不配这条：目标那个形变保持 0"} tone="ghost"
                onClick={() => unpair(current.id)}>
                {onFigure(rig) ? "此部位不配" : "这条不配"}
              </Button>
              </Writes>
            </div>
            {bad[current.id] && <div className="rmap-why">{bad[current.id]}</div>}
            <div className="rmap-cols">
              {COLS.map((col) => (
                <div key={col} className="rmap-col">
                  <span className="rmap-dim">{rig[col].label}{rig[col].fixed ? "（节点固定）" : ""}</span>
                  <Joints names={draft.rows[current.id]?.[col] ?? []} />
                  {current.chain && <Lengths names={draft.rows[current.id]?.[col] ?? []} at={placeOf(col)} end={endOf(current, draft, col)} />}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="rmap-dim">{onFigure(rig)
            ? "点人形图上的一个部位槽（变蓝），再在左右两边的骨架里点关节；也可以先点关节，再右键槽「指定选中的关节」。"
            : `点中间的一个${slotWord(rig)}槽（变蓝），再在左右两边的列表里点名字；也可以先点名字，再右键槽「指定选中的关节」。`}</div>
        )}
      </div>
      {poseParams.length > 0 && (
        <div className="rmap-head">
          <span className="rmap-dim">初始姿势修正：{poseParams.map((q) => `${(typeId ? getNodeDefs()[typeId]?.params.find((d) => d.name === q)?.label : undefined) ?? q} ${poseDraft[q]?.length ?? 0} 个关节`).join(" · ")}</span>
          <span style={{ flex: 1 }} />
          <label className="rmap-dim" data-tip="三维视图里按当前修正后的姿势蒙皮画出角色（与计算时用的姿势一致）">
            显示蒙皮结果 <Switch on={skin} onChange={setSkin} />
          </label>
          <Writes>
            <Button tone="ghost" disabled={!poseUndo.length} tip="退回上一步姿势修正（只在这个弹窗里）"
              onClick={() => { setPoseDraft(poseUndo[poseUndo.length - 1]); setPoseUndo((u) => u.slice(0, -1)); }}>撤销</Button>
          </Writes>
        </div>
      )}
      <SheetFoot ok={() => {
        // 对应关系与改过的姿势修正一起写回（弹窗框架的 set 带上同一节点的附带参数：一步撤销）
        const map = valueOf(draft, rig);
        set(map, posesChanged ? Object.fromEntries(poseParams.map((q) => [q, poseDraft[q] ?? []])) : undefined);
      }} cancel={cancel}
        okTip={requiredOk < required.length ? `必需的${slotWord(rig)}没配齐也可以先存：计算时节点会报错说明` : "用这个对应关系"}>
        <span className="rmap-dim">绿 = 两边都配了 · 黄 = 只有一边有 · 灰 = 两边都没有 · 紫 = 有问题（项目的错误色；权威以计算时为准）</span>
      </SheetFoot>
      {menu && <Menu at={menu} rows={slotMenu(menu.part)} label={`「${parts.get(menu.part)?.label}」`} onClose={() => setMenu(null)} />}
    </div>
  );
}

const other = (col: Col): Col => (col === "src" ? "dst" : "src");

/** 链状部位末端接的那个关节（脊柱接胸、颈接头：部位表的 end，服务端给），只用来画比例条。 */
function endOf(part: RigPart, d: Draft, col: Col): string | undefined {
  return part.end ? d.rows[part.end]?.[col]?.[0] : undefined;
}

function Joints({ names }: { names: string[] }) {
  if (!names.length) return <span className="rmap-none">不配</span>;
  const shown = names.length <= 3 ? names.join(" → ") : `${names[0]} → … → ${names[names.length - 1]}`;
  return <span className="rmap-joints" data-user-data data-tip={names.join("\n")}>{shown}{names.length > 1 ? `（${names.length} 节）` : ""}</span>;
}

/** 链的骨长比例条：每节骨占的长度（基准姿势，来自这一侧骨架手柄的 handle_data），直接看出「2 节分到 4 节」怎么对上
 * （重定向按这个比例分摊弯曲）。这一侧没有骨架数据时不画。 */
function Lengths({ names, at, end }: { names: string[]; at: ((n: string) => readonly number[] | undefined) | null; end?: string }) {
  if (!at || names.length < 1) return null;
  const ends = [...names.slice(1), ...(end && at(end) ? [end] : [])];
  const lengths = names.map((n, k) => {
    const a = at(n), b = ends[k] ? at(ends[k]) : undefined;
    return a && b ? Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]) : 0;
  });
  const fill = lengths.map((l, k) => l || (k ? lengths[k - 1] : 0) || 1);
  const total = fill.reduce((a, b) => a + b, 0) || 1;
  return (
    <div className="rmap-bar" data-tip="按骨长的比例：两边节数不同时，弯曲按这个比例分摊到每一节">
      {fill.map((l, k) => <i key={k} style={{ flexGrow: l / total }} />)}
    </div>
  );
}

/** 中间的人形图：部位槽摆在身体上，背后一个火柴人示意。 */
function Figure({ page, rig, draft, bad, armed, state, onArm, onMenu }: {
  page: Page; rig: RigChoice; draft: Draft; bad: Record<string, string>; armed: string | null; state: (id: string) => string;
  onArm: (id: string) => void; onMenu: (id: string, x: number, y: number) => void;
}) {
  const places = PLACES[page];
  const known = new Set(rig.parts.map((q) => q.id));
  return (
    <div className="rmap-body">
      <div className="rmap-stage">
      <svg className="rmap-links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
        {LINKS[page].filter(([a, b]) => places[a] && places[b]).map(([a, b]) => (
          <line key={`${a}-${b}`} x1={places[a][0]} y1={places[a][1]} x2={places[b][0]} y2={places[b][1]} />
        ))}
      </svg>
      {rig.parts.filter((q) => places[q.id] && known.has(q.id)).map((q) => {
        const row = draft.rows[q.id];
        const count = q.chain ? Math.max(row?.src.length ?? 0, row?.dst.length ?? 0) : 0;
        const tip = [
          `${q.label}${q.chain ? "（链）" : ""}${q.required ? " · 必需" : ""} · ${draft.manual.has(q.id) ? "手动" : "自动"}`,
          `${rig.src.label}：${row?.src.join(" → ") || "不配"}`,
          `${rig.dst.label}：${row?.dst.join(" → ") || "不配"}`,
          ...(bad[q.id] ? [bad[q.id]] : []),
        ].join("\n");
        return (
          <button key={q.id} type="button" className={`rmap-slot ${state(q.id)}${armed === q.id ? " armed" : ""}${q.required ? " req" : ""}`}
            style={{ left: `${places[q.id][0]}%`, top: `${places[q.id][1]}%` }} data-tip={tip}
            onClick={() => onArm(q.id)} onContextMenu={(e) => (e.preventDefault(), onMenu(q.id, e.clientX, e.clientY))}>
            {q.label}
            {q.chain && count > 1 && <em>×{count}</em>}
            {!draft.manual.has(q.id) && <small>自动</small>}
            {rig.diffs?.[q.id] !== undefined && (
              <small className={rig.diffs[q.id] > 10 ? "rmap-diff far" : "rmap-diff"}
                data-tip="两副骨架的基准姿势里这根骨头差多少度（各自的身体坐标里比）：差得多就是初始形态没对上，换「静止姿势」或「基准姿势摆正」">
                Δ{Math.round(rig.diffs[q.id])}°
              </small>
            )}
          </button>
        );
      })}
      </div>
    </div>
  );
}

/** 摆不上人形图的槽（「表情重定向（ARKit52）」的表情）：按区域（眉毛、眼睛、嘴……）一格一格排开，点、右键和人形图上的槽一样。 */
function SlotGrid({ rig, draft, bad, armed, state, onArm, onMenu }: {
  rig: RigChoice; draft: Draft; bad: Record<string, string>; armed: string | null; state: (id: string) => string;
  onArm: (id: string) => void; onMenu: (id: string, x: number, y: number) => void;
}) {
  const regions = [...new Set(rig.parts.map((q) => q.region))];
  return (
    <div className="rmap-body rmap-grid-slots">
      {regions.map((region) => (
        <div key={region} className="rmap-slot-group">
          <span className="rmap-dim">{region}</span>
          <div className="rmap-slot-row">
            {rig.parts.filter((q) => q.region === region).map((q) => {
              const row = draft.rows[q.id];
              const tip = [`${q.label} · ${draft.manual.has(q.id) ? "手动" : "自动"}`, `${rig.src.label}：${row?.src.join("、") || "不配"}`,
                `${rig.dst.label}：${row?.dst.join("、") || "不配"}`, ...(bad[q.id] ? [bad[q.id]] : [])].join("\n");
              return (
                <button key={q.id} type="button" className={`rmap-slot ${state(q.id)}${armed === q.id ? " armed" : ""}`} data-tip={tip}
                  data-user-data onClick={() => onArm(q.id)} onContextMenu={(e) => (e.preventDefault(), onMenu(q.id, e.clientX, e.clientY))}>
                  {q.label}
                  {!draft.manual.has(q.id) && <small>自动</small>}
                </button>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/** 模型类节点的模型一侧：节点固定的关节，按部位列出（只读）；点一行等于点那个部位的槽。 */
function FixedSide({ side, rig, armed, state, onArm }: {
  side: RigSide; rig: RigChoice; armed: string | null; state: (id: string) => string; onArm: (id: string) => void;
}) {
  const rows = rig.parts.filter((q) => side.parts?.[q.id]?.length);
  return (
    <div className="rmap-side">
      <div className="rmap-side-head">
        <b>{side.label}骨架</b>
        <span className="rmap-dim">{side.names.length} 个关节 · 节点固定，只读</span>
      </div>
      <div className="rmap-fixed" style={{ height: LIST + 30 }}>
        {rows.map((q) => (
          <div key={q.id} className={`rmap-fixed-row${armed === q.id ? " on" : ""}`} onClick={() => onArm(q.id)}
            data-tip="点一下选中这个部位（等于点人形图上的槽），再在左边人物的骨架里点关节">
            <span className={`rmap-tag ${state(q.id)}`}>{q.label}</span>
            <span className="rmap-name" data-user-data>{side.parts![q.id].join(" → ")}</span>
          </div>
        ))}
        {!rows.length && <div className="rmap-empty">这个模型没有能被人物骨骼带动的关节</div>}
      </div>
    </div>
  );
}

