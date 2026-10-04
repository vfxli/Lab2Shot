import { useEffect, useState } from "react";
import { adminApi, type ResourceAct, type UserResourcePage, type UserResourceTab, type UserRow } from "../api/admin";
import { Button, Chip, Segmented } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { FilterRow, Filters } from "../ui/Categories";
import { Loading } from "../ui/Loading";
import { reasonOf } from "../messages/message";
import { sizeText, stampText, whenText } from "../platform/format";
import { Table, type Column } from "../ui/Table";
import { useConfirm } from "../ui/Confirm";
import { msg } from "../messages/message";
import "./resources.css";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** Per-user resource lookup: one table for every kind of resource an account owns. The server registry
 * (lab2shot/site/resources.py) defines the kinds, the columns each shows, and which columns the 时间 and 状态 filters
 * read; this file only renders that description. Nothing here knows what a 任务, 上传 or 反馈 is, so adding a new
 * kind of user resource requires one registry entry and no page changes.
 *
 * Used in two places: the 用户 detail page (one tab per kind) and each resource's own list page, where
 * the 「按人」 filter (ByUser) selects an account and shows the identical table. */

const PAGE = 50;

/** Time filter options, as spans back from now; 0 means 全部. */
const SPANS: { value: string; days: number; label: () => string }[] = [
  { value: "all", days: 0, label: () => t("ui.admin.resources.all") },
  { value: "7", days: 7, label: () => t("ui.admin.resources.days", { n: 7 }) },
  { value: "30", days: 30, label: () => t("ui.admin.resources.days", { n: 30 }) },
  { value: "90", days: 90, label: () => t("ui.admin.resources.days", { n: 90 }) },
];

const DAY_S = 86400;

const ROW = "__row"; // Row key used by this page; never a column declared by the registry.
// Extra row fields the registry may set besides the declared columns (lab2shot/site/resources.py RESERVED): whether the
// row is no longer in use, and which of the page's actions apply to it.
const DIM = "__dim";
const ACTS = "__acts";

/** Whether a column value is a timestamp rather than an ordinary number: seconds since 1970, from 2001 onward and not far in the future. */
const isMoment = (v: number) => v > 1e9 && v < 4e9;

/** One cell: a timestamp as date and time, a boolean as a word, an empty value as a dash, and a column the registry
 * marks 「size」 as a size (platform/format.ts is the single size formatter). User-supplied text (a graph name, a
 * path, a device) is marked as user data, so only it may be truncated, with the full text shown on hover. */
function Cell({ value, says }: { value: unknown; says: string }) {
  if (value === null || value === undefined || value === "") return <span className="dim">—</span>;
  if (says === "size" && typeof value === "number") return <span className="tnum">{sizeText(value)}</span>;
  if (typeof value === "boolean") return <>{value ? t("ui.admin.resources.yes") : t("ui.admin.resources.no")}</>;
  if (typeof value === "number") {
    return isMoment(value) ? (
      <span className="tnum" {...tipAttrs(tipOf("value", stampText(value)))}>
        {whenText(value)}
      </span>
    ) : (
      <span className="tnum">{value}</span>
    );
  }
  const text = String(value);
  return (
    <span data-user-data {...tipAttrs(tipOf("truncated", text))}>
      {text}
    </span>
  );
}

export function ResourceTable({ user, kind }: { user: number; kind: string }) {
  const [page, setPage] = useState<UserResourcePage | null>(null);
  const [problem, setProblem] = useState("");
  const [q, setQ] = useState("");
  const [state, setState] = useState("");
  const [span, setSpan] = useState("all");
  const [offset, setOffset] = useState(0);
  const [again, setAgain] = useState(0); // Incremented after an action runs, to re-read the page.
  const [busy, setBusy] = useState("");
  const [ask, confirmSheet] = useConfirm();

  useEffect(() => setOffset(0), [user, kind, q, state, span]);
  useEffect(() => {
    let live = true;
    const days = SPANS.find((s) => s.value === span)?.days ?? 0;
    const since = days ? Date.now() / 1000 - days * DAY_S : 0;
    adminApi.userResource(user, kind, { offset, limit: PAGE, q, since, state }).then(
      (p) => live && (setPage(p), setProblem("")),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [user, kind, q, state, span, offset, again]);

  // Checked rows (keyed by this page's id column), for applying one action to several rows at once.
  // Which actions are available is still computed by the server; this only repeats the call per row.
  const [picked, setPicked] = useState<Set<string>>(new Set());
  useEffect(() => setPicked(new Set()), [user, kind, q, state, span, offset]);

  /** Runs one declared action on one row (lab2shot/site/resources.py Act): dangerous actions ask for confirmation first,
     * then the page is re-read. The available actions, their labels and targets are defined by the server; this file names none. */
  const run = async (act: ResourceAct, row: Record<string, unknown>, rowId: string) => {
    const what = String(row[page?.columns[1]?.key ?? ""] ?? rowId);
    if (act.danger && !(await ask({ title: act.label, say: msg("N-RESOURCE-ACT", { what: act.label, name: what }), yes: act.label, tip: tipOf("consequence", act.tip), danger: true }))) return;
    setBusy(rowId + act.id);
    try {
      await adminApi.rowAct(act, rowId);
      setProblem("");
      setAgain((n) => n + 1);
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy("");
    }
  };

  /** Applies the same action to every checked row: confirms once, then calls per row (the server checks permission and ownership for each). */
  const runMany = async (act: ResourceAct, ids: string[]) => {
    if (!ids.length) return;
    if (!(await ask({ title: act.label, say: msg("N-RESOURCE-ACTMANY", { what: act.label, count: ids.length }),
                      yes: act.label, tip: tipOf("consequence", act.tip), danger: act.danger }))) return;
    setBusy(`many:${act.id}`);
    const failed: string[] = [];
    for (const id of ids) {
      try {
        await adminApi.rowAct(act, id);
      } catch (e) {
        failed.push(t("ui.admin.resources.failed_one", { id, why: reasonOf(e as Error) }));
      }
    }
    setProblem(failed.length ? failed.join(t("ui.admin.resources.failed_sep")) : "");
    setPicked(new Set());
    setBusy("");
    setAgain((n) => n + 1);
  };

  // a page that never came is replaced by why; a refused action on a page that is there is said above it, and the
  // table (with its filters and checked rows) stays
  if (!page) return problem ? <Empty title={problem} hint={t("ui.admin.resources.retry_hint")} /> : <Loading what={t("ui.admin.user.records")} />;

  const idKeyAll = page.columns[0]?.key ?? "";
  const canPick = page.acts.length > 0;  // Row selection is needed only when actions exist.
  const idsOn = (row: Record<string, unknown>) => String(row[idKeyAll] ?? "");
  const pickable = page.rows.filter((r) => (Array.isArray(r[ACTS]) ? (r[ACTS] as string[]).length : 0) > 0).map(idsOn);
  const allOn = pickable.length > 0 && pickable.every((id) => picked.has(id));

  const columns: Column<Record<string, unknown>>[] = page.columns.map((c) => ({
    id: c.key,
    label: c.label,
    cell: (row) => <Cell value={row[c.key]} says={c.says} />,
  }));
  // 操作 column: only when the registry declares actions and this login may use some. The server has already filtered
  // them by the permissions each action's route requires, so every action received here is usable.
  if (canPick) {
    // The checkbox column comes first: as in a DCC, batch buttons act on the selected rows.
    columns.unshift({
      id: "pick",
      width: "2.5rem",
      label: "",
      cell: (row) => {
        const id = idsOn(row);
        const mine = Array.isArray(row[ACTS]) ? (row[ACTS] as string[]) : [];
        if (!mine.length) return null;  // No checkbox for a row with no applicable action.
        return (
          <input type="checkbox" checked={picked.has(id)} aria-label={t("ui.admin.resources.pick")}
            onChange={(e) => setPicked((was) => {
              const now = new Set(was);
              if (e.target.checked) now.add(id); else now.delete(id);
              return now;
            })} />
        );
      },
    });
  }
  if (page.acts.length) {
    const idKey = page.columns[0]?.key ?? "";
    columns.push({
      id: "acts",
      width: "13rem",
      label: t("ui.admin.extensions.col_action"),
      cell: (row) => {
        const rowId = String(row[idKey] ?? "");
        const mine = Array.isArray(row[ACTS]) ? (row[ACTS] as string[]) : [];
        return (
          <span className="res-acts">
            {page.acts
              .filter((a) => mine.includes(a.id))
              .map((a) => (
                <Button key={a.id} tip={tipOf("consequence", a.tip)} tone="ghost" size="sm" danger={a.danger} disabled={busy === rowId + a.id} onClick={() => void run(a, row, rowId)}>
                  {a.label}
                </Button>
              ))}
          </span>
        );
      },
    });
  }
  const last = Math.min(offset + page.rows.length, page.total);

  return (
    <div className="res">
      {problem && <div className="notice" role="alert">{problem}</div>}
      {/* Bar shown only when rows are selected: applies an action to all of them.
          The available actions are computed by the server for the current login; for example, a secondary administrator does not see 「永久删除」 because that route requires data.others. */}
      {canPick && picked.size > 0 && (
        <div className="res-many" role="status">
          <span>{t("ui.admin.resources.picked", { n: picked.size })}</span>
          {page.acts
            .filter((a) => page.rows.some((r) => picked.has(idsOn(r)) && (Array.isArray(r[ACTS]) ? (r[ACTS] as string[]) : []).includes(a.id)))
            .map((a) => (
              <Button key={a.id} tip={tipOf("consequence", a.tip)} tone="ghost" size="sm"
                danger={a.danger} disabled={busy === `many:${a.id}`}
                onClick={() => void runMany(a, page.rows.filter((r) => picked.has(idsOn(r))).map(idsOn))}>
                {a.label}
              </Button>
            ))}
          <Button tone="ghost" size="sm" onClick={() => setPicked(new Set())}>
            {t("ui.admin.resources.unpick")}
          </Button>
          {allOn ? null : (
            <Button tone="ghost" size="sm" onClick={() => setPicked(new Set(pickable))}>
              {t("ui.admin.resources.pick_page")}
            </Button>
          )}
        </div>
      )}
      <Filters>
        <FilterRow label={t("ui.admin.resources.search")}>
          <input
            className="field res-search"
            value={q}
            placeholder={t("ui.admin.resources.search_in", { what: page.label })}
            aria-label={t("ui.admin.resources.search_in", { what: page.label })}
            onChange={(e) => setQ(e.target.value)}
          />
          {page.when && (
            <Segmented
              label={t("ui.admin.resources.time")}
              value={span}
              options={SPANS.map((s) => ({ value: s.value, label: s.label() }))}
              onChange={setSpan}
              layout="res-span"
            />
          )}
        </FilterRow>
        {page.states.length > 1 && (
          <FilterRow label={t("ui.admin.resources.state")}>
            <Chip size="md" on={state === ""} count={page.total} onClick={() => setState("")}>
              {t("ui.admin.resources.all")}
            </Chip>
            {page.states.map((s) => (
              <Chip key={s} size="md" on={state === s} onClick={() => setState(s)}>
                {s}
              </Chip>
            ))}
          </FilterRow>
        )}
      </Filters>
      <Table
        rows={page.rows.map((row, i): Record<string, unknown> => ({ ...row, [ROW]: offset + i }))}
        columns={columns}
        /* The registry's first column identifies a row for display but not always uniquely (two sessions of one
           account share 方式), so the row's position in the page is used as its key. */
        rowKey={(row) => String(row[ROW])}
        /* Rows the user has deleted (in the recycle bin) are dimmed, as flagged by the registry (`__dim`), not inferred here. */
        dim={page.dim ? (row) => !!row[DIM] : undefined}
        empty={<Empty title={t("ui.admin.resources.none", { what: page.label })} hint={q || state || span !== "all" ? t("ui.admin.resources.widen") : undefined} />}
      />
      {page.total > PAGE && (
        <div className="res-pages">
          <Button tone="ghost" disabled={offset === 0} onClick={() => setOffset(Math.max(offset - PAGE, 0))}>
            {t("ui.admin.resources.prev")}
          </Button>
          <span className="dim tnum">
            {t("ui.admin.resources.range", { from: offset + 1, to: last, total: page.total })}
          </span>
          <Button tone="ghost" disabled={last >= page.total} onClick={() => setOffset(offset + PAGE)}>
            {t("ui.admin.resources.next")}
          </Button>
        </div>
      )}
      {confirmSheet}
    </div>
  );
}

/** The accounts this login may see (deleted ones left out): [] when it may see none, the server's answer decides,
 * not a role check here; null while loading. */
export function useAccounts(): UserRow[] | null {
  const [users, setUsers] = useState<UserRow[] | null>(null);
  useEffect(() => {
    let live = true;
    adminApi.users().then(
      (v) => live && setUsers(v.users.filter((u) => !u.deleted)),
      () => live && setUsers([]),
    );
    return () => {
      live = false;
    };
  }, []);
  return users;
}

/** An account as a person is named where one is picked: 中文名（用户名）, as 使用统计 names them (lab2shot/farm/usage.py).
 * Two people may share a Chinese name; the username is what tells them apart. */
const personText = (u: UserRow) => (u.name ? t("ui.admin.resources.person", { name: u.name, username: u.username }) : u.username);

/** The 「按人」 row: 全部 and one chip per account. */
export function UserChips({ users, chosen, onChoose }: { users: UserRow[]; chosen: number | null; onChoose: (id: number | null) => void }) {
  return (
    <FilterRow label={t("ui.admin.resources.by_person")}>
      <Chip size="md" on={chosen === null} onClick={() => onChoose(null)}>
        {t("ui.admin.resources.all")}
      </Chip>
      {users.map((u) => (
        <Chip key={u.id} size="md" on={chosen === u.id} onClick={() => onChoose(u.id)}>
          {personText(u)}
        </Chip>
      ))}
    </FilterRow>
  );
}

/** The 「按人」 filter: appended to every section that lists a registered resource. It selects an account and shows
 * exactly what the account's own page shows, through the same listing function. It receives only the section it
 * belongs to; the resources of that section come from the registry (each account tab carries its `section`), so
 * no page names a resource kind and adding a new kind requires no page changes.
 *
 * It fetches the account list itself and renders nothing when this login may not see accounts. */
export function ByUser({ section }: { section: string }) {
  const users = useAccounts();
  const [chosen, setChosen] = useState<number | null>(null);
  const [kinds, setKinds] = useState<UserResourceTab[]>([]);
  const [kind, setKind] = useState("");

  useEffect(() => {
    if (chosen === null) return;
    let live = true;
    adminApi.userResources(chosen).then(
      (r) => {
        if (!live) return;
        const mine = r.tabs.filter((t) => t.section === section);
        setKinds(mine);
        setKind(mine[0]?.kind ?? "");
      },
      () => live && (setKinds([]), setKind("")),
    );
    return () => {
      live = false;
    };
  }, [chosen, section]);

  if (!users?.length) return null;
  return (
    <div className="res-by-user">
      <Filters>
        <UserChips users={users} chosen={chosen} onChoose={setChosen} />
        {chosen !== null && kinds.length > 1 && (
          <FilterRow label={t("ui.admin.resources.which")}>
            {kinds.map((k) => (
              <Chip key={k.kind} size="md" on={kind === k.kind} count={k.count} onClick={() => setKind(k.kind)}>
                {k.label}
              </Chip>
            ))}
          </FilterRow>
        )}
      </Filters>
      {chosen !== null && kind && <ResourceTable key={`${chosen}-${kind}`} user={chosen} kind={kind} />}
      {chosen !== null && !kind && <Empty title={t("ui.admin.resources.none_here")} hint={t("ui.admin.resources.none_here_hint")} />}
    </div>
  );
}
