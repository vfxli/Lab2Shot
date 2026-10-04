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
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

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
      <Section title={t("ui.admin.nav.cards")}>
        <Loading what={t("ui.admin.nav.cards")} />
      </Section>
    );
  }
  const byUuid = new Map(data.cards.map((c) => [c.uuid, c]));
  const nameOf = (uuid: string) => (byUuid.has(uuid) ? cardName(byUuid.get(uuid)!) : uuid);

  const tierColumns: Column<CardTier>[] = [
    { id: "node", label: t("ui.admin.cards.col_node"), cell: (r) => r.subtitle },
    { id: "setting", label: t("ui.admin.cards.col_setting"), cell: (r) => (r.param ? `${r.param} = ${r.option}` : t("ui.admin.cards.default")) },
    { id: "vram", label: t("ui.admin.cards.col_vram"), cell: (r) => gbText(r.vram_gb), className: "tnum" },
    { id: "cards", label: t("ui.admin.cards.col_cards"), cell: (r) => r.cards.map(nameOf).join(t("list.sep")) || t("ui.admin.cards.none") },
    {
      id: "now",
      label: t("ui.admin.cards.col_now"),
      cell: (r) => (
        <span className={`chip${r.available ? " q-ok" : ""}`} {...tipAttrs(tipOf("error", said(r.message)))}>
          {r.available ? t("ui.admin.cards.runs") : t("ui.admin.cards.no_card")}
        </span>
      ),
    },
  ];
  const waitingColumns: Column<CardWaiting>[] = [
    { id: "title", label: t("ui.admin.cards.col_job"), cell: (w) => <span data-user-data {...tipAttrs(tipOf("truncated", w.title))}>{w.title}</span> },
    { id: "node", label: t("ui.admin.cards.col_node"), cell: (w) => <span data-user-data {...tipAttrs(tipOf("truncated", w.node))}>{w.node}</span> },
    { id: "who", label: t("ui.admin.cards.col_who"), cell: (w) => <span data-user-data {...tipAttrs(tipOf("truncated", w.who))}>{w.who}</span> },
    { id: "vram", label: t("ui.admin.cards.col_vram"), cell: (w) => gbText(w.vram_gb), className: "tnum" },
    { id: "why", label: t("ui.admin.cards.col_why"), cell: (w) => said(w.reason) || t("ui.admin.cards.next_up") },
    {
      id: "ever",
      label: t("ui.admin.cards.col_ever"),
      cell: (w) => <span className={`chip${w.runnable_ever ? " q-ok" : ""}`}>{w.runnable_ever ? t("ui.admin.cards.will_start") : t("ui.admin.cards.no_card_runs")}</span>,
    },
  ];
  const nodeColumns: Column<CardNode>[] = [
    { id: "subtitle", label: t("ui.admin.cards.col_node"), cell: (n) => n.subtitle },
    { id: "vram", label: t("ui.admin.cards.col_vram"), cell: (n) => gbText(n.vram_gb), className: "tnum" },
    {
      id: "measured",
      label: t("ui.admin.cards.col_source"),
      cell: (n) => <span {...tipAttrs(tipOf("value", said(n.note)))}>{n.vram_measured ? (n.measured_on ? t("ui.admin.cards.measured_on", { on: n.measured_on }) : t("ui.admin.cards.measured")) : t("ui.admin.cards.estimated")}</span>,
    },
  ];

  return (
    <>
      <Section
        title={t("ui.admin.nav.cards")}
        lede={t("ui.admin.cards.lede")}
        actions={
          <Button tone="ghost" onClick={reload}>
            {t("ui.admin.common.refresh")}
          </Button>
        }
      >
        {data.cards.length > 0 && data.cards.every((c) => !c.authorized) && (
          <div className="notice">
            {/* what to do about it is what this login can do: switch a card on, or ask someone who may */}
            {t("ui.admin.cards.none_on")}
            {canSwitch ? t("ui.admin.cards.switch_one") : t("ui.admin.cards.ask_admin")}
          </div>
        )}
        {data.cards.length ? (
          <div className="q-gpus">
            {data.cards.map((c) => (
              <CardTile key={c.uuid} card={c} busy={busy === c.uuid} onToggle={canSwitch ? (on) => void ask(c, on) : undefined} />
            ))}
          </div>
        ) : (
          <p className="adm-lede">{t("ui.admin.cards.no_gpus")}</p>
        )}
      </Section>
      <Section title={t("ui.admin.cards.hourly")} lede={t("ui.admin.cards.hourly_lede")}>
        <GpuHours hourly={data.hourly ?? []} cards={data.cards} />
      </Section>
      <Section title={t("ui.admin.cards.tiers")} lede={t("ui.admin.cards.tiers_lede")}>
        <Folds
          rows={data.tiers}
          groupOf={(r) => r.runtime}
          titleOf={(r) => r.runtime_title}
          summary={(rows) => {
            const top = rows.reduce((a, b) => (b.vram_gb > a.vram_gb ? b : a));
            return (
              <>
                <span className="adm-fold-count">{t("ui.admin.cards.tier_count", { n: rows.length })}</span>
                <span className="tnum" {...tipAttrs(tipOf("value", top.param ? t("ui.admin.cards.top_tier_param", { label: top.subtitle, param: top.param, option: top.option }) : t("ui.admin.cards.top_tier", { label: top.subtitle })))}>{t("ui.admin.cards.max", { gb: gbText(top.vram_gb) })}</span>
                <span className={`chip${rows.every((r) => r.available) ? " q-ok" : ""}`}>
                  {rows.every((r) => r.available) ? t("ui.admin.cards.all_run") : t("ui.admin.cards.some_run", { n: rows.filter((r) => r.available).length, all: rows.length })}
                </span>
              </>
            );
          }}
          empty={t("ui.admin.cards.no_gpu_nodes")}
        >
          {(rows) => <Table rows={rows} columns={tierColumns} rowKey={(r) => r.id} empty="" />}
        </Folds>
      </Section>
      <Section title={t("ui.admin.cards.waiting")} lede={t("ui.admin.cards.waiting_lede")}>
        <Table rows={data.waiting} columns={waitingColumns} rowKey={(w) => `${w.job}-${w.node}`} empty={t("ui.admin.cards.waiting_none")} />
      </Section>
      <Section title={t("ui.admin.cards.node_vram")} lede={t("ui.admin.cards.node_vram_lede")}>
        <Folds
          rows={data.nodes}
          groupOf={(n) => n.runtime}
          titleOf={(n) => n.runtime_title}
          summary={(rows) => {
            const top = rows.reduce((a, b) => (b.vram_gb > a.vram_gb ? b : a));
            return (
              <>
                <span className="adm-fold-count">{t("ui.admin.cards.node_count", { n: rows.length })}</span>
                <span className="tnum" {...tipAttrs(tipOf("value", t("ui.admin.cards.top_node", { label: top.subtitle })))}>{t("ui.admin.cards.max", { gb: gbText(top.vram_gb) })}</span>
                <span className="adm-fold-note">{t("ui.admin.cards.measured_count", { n: rows.filter((n) => n.vram_measured).length })}</span>
              </>
            );
          }}
          empty={t("ui.admin.cards.no_gpu_nodes")}
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
        <span className="q-gpu-name" {...tipAttrs(tipOf("value", card.arch ? t("ui.admin.cards.arch", { name: card.name, arch: card.arch }) : card.name))}>
          {cardName(card)}
        </span>
        {onToggle ? (
          <Switch
            on={card.authorized}
            label={t("ui.admin.cards.takes_label", { model: card.model })}
            disabled={busy}
            onChange={onToggle}
          />
        ) : (
          <span className={`chip${card.authorized ? " q-ok" : ""}`}>{card.authorized ? t("ui.admin.cards.takes") : t("ui.admin.cards.takes_not")}</span>
        )}
      </div>
      <div className="q-gpu-meter">
        <i style={{ width: `${used * 100}%` }} />
      </div>
      <div className="q-gpu-stats tnum">
        {t("ui.admin.cards.stats", { used: gbText(card.load.used_gb), total: gbText(card.memory_gb), util: card.load.utilization, temp: card.load.temperature })}
      </div>
      <div className="q-gpu-job">
        {card.running
          ? t("ui.admin.cards.running", { what: [card.running.who, card.running.title, card.running.node].filter(Boolean).join(" · ") })
          : card.authorized
            ? t("ui.admin.cards.idle")
            : t("ui.admin.cards.not_lab2shot")}
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
    <Sheet title={on ? t("ui.admin.cards.turn_on_title", { card: cardName(card) }) : t("ui.admin.cards.turn_off_title", { card: cardName(card) })} onClose={onClose}>
      <div className="cards-said">
        {other && (
          <div className="notice warn">
            {t("ui.admin.cards.other_programs", { gb: gbText(card.load.used_gb) })}
          </div>
        )}
        {answer.messages.map((m) => (
          <div key={m.code + m.text} className={`notice${m.level === "W" ? " warn" : ""}`}>
            {m.text}
          </div>
        ))}
      </div>
      <div className="dialog-row cards-end">
        <Button tone="ghost" onClick={onClose}>
          {t("ui.admin.common.cancel")}
        </Button>
        <Button tip={on ? undefined : tipOf("consequence", t("ui.admin.cards.turn_off_tip"))} tone="primary" disabled={busy} onClick={onApply}>
          {on ? t("ui.admin.cards.turn_on") : t("ui.admin.cards.turn_off")}
        </Button>
      </div>
    </Sheet>
  );
}
