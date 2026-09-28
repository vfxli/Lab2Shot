import { useCallback, useState } from "react";
import { api } from "../api";
import { cardsApi, type CardNode, type CardRow, type CardsConsequences, type CardTier, type CardWaiting } from "../api/cards";
import { usable, type MessageJson } from "../api/applies";
import { MessageError, reasonOf } from "../messages/message";
import { gbText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { useSignedIn } from "../state/session";
import { Button, Switch } from "../ui/Button";
import { GpuFits } from "../ui/GpuFits";
import { GpuHours } from "./GpuHours";
import { Loading } from "../ui/Loading";
import { Sheet } from "../ui/Sheet";
import { Table, type Column } from "../ui/Table";
import { Section, useAdmin } from "./common";

/** 显卡 section. Artists never see or choose a card; this section gathers all card information for the
 * administrator and previews the effect of a change before applying it. It lists every card of this machine
 * (lab2shot/farm/cards.py view) with its load, running jobs, the GPU extensions assigned to it and why; the
 * hourly use of each card (GpuHours.tsx); the parameter tiers and the cards that run them; the GPU nodes waiting
 * for a card and whether any authorised card can run them; and the VRAM recorded for every GPU node (measured or
 * estimated). Switching a card first asks the server for the consequences (cards/consequences) and applies the change
 * only after confirmation. Permissions come from the server (applies); the page performs no role checks. */

const BUSY_GB = 2; // Memory (GB) held by other programs on a card with no Lab2Shot job above which the card is reported as in use (e.g. training or rendering).

const cardName = (c: Pick<CardRow, "index" | "model">) => `GPU ${c.index} · ${c.model}`;
const said = (m: MessageJson | string | null) => (m == null ? "" : typeof m === "string" ? m : m.text);

export function CardsSection() {
  const { problem, refreshQueue } = useAdmin();
  const state = useSignedIn();
  const read = useCallback(() => cardsApi.cards(), []);
  const { data, reload } = usePoll(read, 5000, { onError: (e) => problem(reasonOf(e)) });
  const [asking, setAsking] = useState<{ card: CardRow; on: boolean; next: string[]; answer: CardsConsequences } | null>(null);
  const [busy, setBusy] = useState("");
  const canSwitch = usable(state?.applies, "queue.authorize");

  const run = async (uuid: string, call: () => Promise<unknown>) => {
    setBusy(uuid);
    try {
      await call();
      problem(null);
    } catch (e) {
      problem(reasonOf(e));
    } finally {
      setBusy("");
    }
  };
  // the full list the server is to hold: every card as it stands now, this one switched
  const nextOf = (cards: CardRow[], card: CardRow, on: boolean) => cards.filter((c) => (c.uuid === card.uuid ? on : c.authorized)).map((c) => c.uuid);
  const ask = (card: CardRow, on: boolean) =>
    run(card.uuid, async () => {
      const next = nextOf(data?.cards ?? [], card, on);
      setAsking({ card, on, next, answer: await cardsApi.cardConsequences(next) });
    });
  // The server takes the whole list. Read again at 确认: if another card was switched meanwhile (another administrator,
  // the command line), the list asked about is stale and would switch that one back, so the consequences are asked
  // again for the list as it is now instead of sending the old one.
  const apply = () =>
    asking &&
    run(asking.card.uuid, async () => {
      const next = nextOf((await cardsApi.cards()).cards, asking.card, asking.on);
      if ([...next].sort().join() !== [...asking.next].sort().join()) {
        setAsking({ ...asking, next, answer: await cardsApi.cardConsequences(next) });
        reload();
        throw new MessageError("W-CARDS-CHANGED");
      }
      await api.admin.authorize(next);
      setAsking(null);
      reload();
      refreshQueue();
    });
  if (!data) {
    return (
      <Section title="显卡">
        <Loading what="显卡" />
      </Section>
    );
  }
  const byUuid = new Map(data.cards.map((c) => [c.uuid, c]));
  const nameOf = (uuid: string) => (byUuid.has(uuid) ? cardName(byUuid.get(uuid)!) : uuid);

  const tierColumns: Column<CardTier>[] = [
    { id: "node", label: "节点", tip: "用显卡的节点", cell: (t) => t.label },
    { id: "setting", label: "设置", tip: "节点的默认设置，或某个参数的某个选项（它自己实测了显存）", cell: (t) => (t.param ? `${t.param} = ${t.option}` : "默认") },
    { id: "vram", label: "显存", tip: "实测的显存峰值，调度时再留一点余量", cell: (t) => gbText(t.vram_gb), className: "tnum" },
    { id: "cards", label: "能跑的卡", tip: "显存够、扩展包环境支持的卡，不管接不接任务", cell: (t) => t.cards.map(nameOf).join("、") || "没有" },
    {
      id: "now",
      label: "现在",
      tip: "有没有接任务的卡能跑它：没有的话，用户那里这个设置变灰并写原因",
      cell: (t) => (
        <span className={`chip${t.available ? " q-ok" : ""}`} data-tip={said(t.message)}>
          {t.available ? "能跑" : "没有卡"}
        </span>
      ),
    },
  ];
  const waitingColumns: Column<CardWaiting>[] = [
    { id: "title", label: "任务", tip: "有显卡节点在等卡的任务", cell: (w) => <span data-user-data data-tip={w.title}>{w.title}</span> },
    { id: "node", label: "节点", tip: "在等卡的显卡节点", cell: (w) => <span data-user-data data-tip={w.node}>{w.node}</span> },
    { id: "who", label: "提交者", tip: "谁提交的", cell: (w) => <span data-user-data data-tip={w.who}>{w.who}</span> },
    { id: "vram", label: "显存", tip: "这个节点要的显存", cell: (w) => gbText(w.vram_gb), className: "tnum" },
    { id: "why", label: "在等什么", tip: "服务器说它为什么还没开始", cell: (w) => said(w.reason) || "轮到就开始" },
    {
      id: "ever",
      label: "能不能开始",
      tip: "现在接任务的卡里有没有一张显存够、环境支持它：没有的话它会一直等，直到授权一张能跑的卡",
      cell: (w) => <span className={`chip${w.runnable_ever ? " q-ok" : ""}`}>{w.runnable_ever ? "等到就能开始" : "没有卡能跑"}</span>,
    },
  ];
  const nodeColumns: Column<CardNode>[] = [
    { id: "label", label: "节点", tip: "用显卡的节点", cell: (n) => n.label },
    { id: "vram", label: "显存", tip: "节点声明的显存峰值", cell: (n) => gbText(n.vram_gb), className: "tnum" },
    {
      id: "measured",
      label: "来源",
      tip: "实测过的写明在哪张卡上测的；没测过的是估计，按估计排任务",
      cell: (n) => <span data-tip={said(n.note) || undefined}>{n.vram_measured ? `实测${n.measured_on ? ` · ${n.measured_on}` : ""}` : "估计"}</span>,
    },
  ];

  return (
    <>
      <Section
        title="显卡"
        lede="这台机器的每张显卡：打开开关的卡接队列里的任务，每张同时算一个；关掉的算完手上的任务后不再接新的。改开关之前先列出会发生什么。显存和占用是这张卡上所有程序的。"
        actions={
          <Button tip="重新读取显卡状态" tone="ghost" onClick={reload}>
            刷新
          </Button>
        }
      >
        {data.cards.length > 0 && data.cards.every((c) => !c.authorized) && (
          <div className="notice">
            {/* what to do about it is what this login can do: switch a card on, or ask someone who may */}
            还没有授权任何显卡：用到显卡的任务会一直排队。
            {canSwitch ? "打开一张卡的开关，它就开始接任务。" : "这个登录改不了显卡的开关：请管理员打开一张卡，它就开始接任务。"}
          </div>
        )}
        {data.cards.length ? (
          <div className="q-gpus">
            {data.cards.map((c) => (
              <CardTile key={c.uuid} card={c} busy={busy === c.uuid} onToggle={canSwitch ? (on) => void ask(c, on) : undefined} />
            ))}
          </div>
        ) : (
          <p className="adm-lede">这台机器上没有找到显卡。</p>
        )}
      </Section>
      <Section title="每小时使用率" lede="每张显卡最近一天每小时的平均占用（0–100%）。数是队列本来就在读的那次显卡状态带回来的，不额外花机器的力气；虚线那根是还没走完的这一小时。">
        <GpuHours hourly={data.hourly ?? []} cards={data.cards} />
      </Section>
      <Section title="参数档位" lede="按扩展包折起来，每包一行写它最吃显存的那一档和现在能不能跑；点开看这个包每个节点在默认设置和每个自己实测了显存的选项下要多少显存、哪些卡能跑。">
        <Folds
          rows={data.tiers}
          groupOf={(t) => t.runtime}
          titleOf={(t) => t.runtime_title}
          summary={(rows) => {
            const top = rows.reduce((a, b) => (b.vram_gb > a.vram_gb ? b : a));
            return (
              <>
                <span className="adm-fold-count">{rows.length} 档</span>
                <span className="tnum" data-tip={`最吃显存的一档：${top.label}${top.param ? `，${top.param} = ${top.option}` : ""}`}>最大 {gbText(top.vram_gb)}</span>
                <span className={`chip${rows.every((t) => t.available) ? " q-ok" : ""}`} data-tip={rows.every((t) => t.available) ? "每一档都有接任务的卡能跑" : `${rows.filter((t) => !t.available).length} 档现在没有卡能跑`}>
                  {rows.every((t) => t.available) ? "都能跑" : `${rows.filter((t) => t.available).length} / ${rows.length} 档能跑`}
                </span>
              </>
            );
          }}
          empty="没有用显卡的节点"
        >
          {(rows) => <Table rows={rows} columns={tierColumns} rowKey={(t) => t.id} empty="" />}
        </Folds>
      </Section>
      <Section title="等卡的显卡节点" lede="任务里在等显卡的节点，它们在等什么，以及现在接任务的卡里有没有能跑它的。">
        <Table rows={data.waiting} columns={waitingColumns} rowKey={(w) => `${w.job}-${w.node}`} empty="没有在等卡的显卡节点" />
      </Section>
      <Section title="节点显存" lede="每个用显卡的节点记下的显存：调度按它挑卡，参数档位也按它算。按扩展包折起来，每包一行写它最吃显存的节点。">
        <Folds
          rows={data.nodes}
          groupOf={(n) => n.runtime}
          titleOf={(n) => n.runtime_title}
          summary={(rows) => {
            const top = rows.reduce((a, b) => (b.vram_gb > a.vram_gb ? b : a));
            return (
              <>
                <span className="adm-fold-count">{rows.length} 个节点</span>
                <span className="tnum" data-tip={`最吃显存的节点：${top.label}`}>最大 {gbText(top.vram_gb)}</span>
                <span className="adm-fold-note">{rows.filter((n) => n.vram_measured).length} 个实测</span>
              </>
            );
          }}
          empty="没有用显卡的节点"
        >
          {(rows) => <Table rows={rows} columns={nodeColumns} rowKey={(n) => n.node} empty="" />}
        </Folds>
      </Section>
      {asking && <Consequences asking={asking} busy={busy === asking.card.uuid} onApply={() => void apply()} onClose={() => setAsking(null)} />}
    </>
  );
}

/** Rows grouped by extension, collapsed by default to keep the page short: one summary line per extension
 * (supplied by the caller, e.g. its largest tier); the table is rendered only when expanded. */
function Folds<T>({ rows, groupOf, titleOf, summary, empty, children }: {
  rows: T[]; groupOf: (r: T) => string; titleOf: (r: T) => string; summary: (rows: T[]) => React.ReactNode; empty: string;
  children: (rows: T[]) => React.ReactNode;
}) {
  if (!rows.length) return <p className="adm-lede">{empty}</p>;
  const groups = new Map<string, T[]>();
  for (const r of rows) (groups.get(groupOf(r)) ?? groups.set(groupOf(r), []).get(groupOf(r))!).push(r);
  return (
    <div className="adm-folds">
      {[...groups.entries()].sort((a, b) => titleOf(a[1][0]).localeCompare(titleOf(b[1][0]), "zh")).map(([key, items]) => (
        <details key={key} className="adm-fold">
          <summary>
            <span className="adm-fold-title">{titleOf(items[0])}</span>
            <span className="mono adm-fold-key">{key}</span>
            {summary(items)}
          </summary>
          {children(items)}
        </details>
      ))}
    </div>
  );
}

function CardTile({ card, busy, onToggle }: { card: CardRow; busy: boolean; onToggle?: (on: boolean) => void }) {
  const used = card.memory_gb ? card.load.used_gb / card.memory_gb : 0;
  return (
    <div className={`q-gpu${card.authorized ? " on" : ""}`}>
      <div className="q-gpu-head">
        <span className="q-gpu-name" data-tip={`${card.name}${card.arch ? ` · 架构 ${card.arch}` : ""}`}>
          {cardName(card)}
        </span>
        {onToggle ? (
          <Switch
            on={card.authorized}
            label={`${card.model} 接任务`}
            tip={card.authorized ? "关掉：先列出会发生什么，确认后算完手上的任务不再接新任务" : "打开：先列出会发生什么，确认后这张卡开始接队列里的任务"}
            disabled={busy}
            onChange={onToggle}
          />
        ) : (
          <span className={`chip${card.authorized ? " q-ok" : ""}`}>{card.authorized ? "接任务" : "不接任务"}</span>
        )}
      </div>
      <div className="q-gpu-meter" data-tip="显存占用（这张卡上所有程序的，不只是 Lab2Shot）">
        <i style={{ width: `${used * 100}%` }} />
      </div>
      <div className="q-gpu-stats tnum">
        显存 {gbText(card.load.used_gb)} / {gbText(card.memory_gb)} · 占用 {card.load.utilization}% · {card.load.temperature}°C
      </div>
      <div className="q-gpu-job">
        {card.running
          ? `正在算：${[card.running.who, card.running.title, card.running.node].filter(Boolean).join(" · ")}`
          : card.authorized
            ? "空闲"
            : "不接 Lab2Shot 的任务"}
      </div>
      <GpuFits fits={card.extensions} />
      <div className="q-gpu-uuid mono">{card.uuid}</div>
    </div>
  );
}

/** Confirmation before switching a card: shows the server's description of the consequences, then 打开 / 关掉 or 取消. */
function Consequences({ asking, busy, onApply, onClose }: { asking: { card: CardRow; on: boolean; answer: CardsConsequences }; busy: boolean; onApply: () => void; onClose: () => void }) {
  const { card, on, answer } = asking;
  const other = on && !card.running && card.load.used_gb > BUSY_GB;
  return (
    <Sheet title={`${on ? "打开" : "关掉"} ${cardName(card)}`} onClose={onClose}>
      <div className="cards-said">
        {other && (
          <div className="notice warn">
            这张卡上别的程序已经占用了 {gbText(card.load.used_gb)} 显存（可能在训练或渲染）：让它接 Lab2Shot 的任务，两边都可能显存不够而出错。
          </div>
        )}
        {answer.messages.map((m) => (
          <div key={m.code + m.text} className={`notice${m.level === "W" ? " warn" : ""}`} data-tip={m.code}>
            {m.text}
          </div>
        ))}
      </div>
      <div className="dialog-row cards-end">
        <Button tip="不改" tone="ghost" onClick={onClose}>
          取消
        </Button>
        <Button tip={on ? "这张卡开始接任务" : "这张卡算完手上的任务后不再接新任务"} tone="primary" disabled={busy} onClick={onApply}>
          {on ? "打开" : "关掉"}
        </Button>
      </div>
    </Sheet>
  );
}
