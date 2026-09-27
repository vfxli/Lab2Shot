import { useCallback, useEffect, useState } from "react";
import { adminApi, SettingsRefused, type SettingDef, type SettingsView, type SettingValue, type StatusRow } from "../api/admin";
import { NoticeCard } from "./Notice";
import { PasswordCard } from "./Auth";
import { Section, useAdmin } from "./common";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button, Segmented, Switch } from "../ui/Button";

/** 设置: every setting of the machine (one schema, lab2shot/config.py), grouped, each with its hover help; the ones
 * that take effect after a restart say so (需重启) and, once saved, wait for it (待重启). Changes are checked on the
 * page (range, numbers) and again on the server (paths, ports), and saved together. What the server can tell that
 * goes with a group (free memory, the address, the Hugging Face login) is shown with it, read only. */

type Draft = Record<string, string | boolean>; // key -> what the field holds (numbers as typed)

let kept: Draft = {}; // unsaved changes outlive switching to another section

/** What a field holds for a value (a list as its names, space-separated). */
const shownAs = (v: SettingValue): string | boolean => (typeof v === "boolean" ? v : Array.isArray(v) ? v.join(" ") : String(v));
const same = (a: SettingValue | undefined, b: SettingValue) => JSON.stringify(a) === JSON.stringify(b);

/** The field's content as the value to save, or why it can't be (the same words as the server). */
function parse(s: SettingDef, raw: string | boolean): { value?: SettingValue; problem?: string } {
  if (s.kind === "list") {
    const items = String(raw).replace(/，/g, ",").split(/[\s,]+/).filter(Boolean);
    if (!items.length) return { problem: `「${s.label}」至少要有一个` };
    const twice = items.filter((x, i) => items.indexOf(x) !== i);
    if (twice.length) return { problem: `「${s.label}」里有重复的：${[...new Set(twice)].join("、")}` };
    return { value: items };
  }
  if (s.kind === "bool" || s.kind === "choice" || s.kind === "text") return { value: typeof raw === "string" ? raw.trim() : raw };
  const text = String(raw).trim();
  const n = Number(text);
  if (!text || !Number.isFinite(n)) return { problem: `「${s.label}」要填数字` };
  if (s.kind === "int" && !Number.isInteger(n)) return { problem: `「${s.label}」要填整数` };
  if ((s.min !== null && n < s.min) || (s.max !== null && n > s.max)) return { problem: `「${s.label}」要在 ${s.min} 到 ${s.max}${s.unit ? ` ${s.unit}` : ""} 之间` };
  return { value: n };
}

export function SettingsSection() {
  const { problem, refreshOverview } = useAdmin();
  const { data: loaded } = usePoll(adminApi.settings, null, { onError: (e) => problem(reasonOf(e)) });
  const [view, setView] = useState<SettingsView | null>(null);
  const [draft, setDraftState] = useState<Draft>(kept);
  const [refused, setRefused] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => void (loaded && setView(loaded)), [loaded]);
  useEffect(() => {
    if (!saved) return;
    const t = window.setTimeout(() => setSaved(false), 6000); // "已保存" is shown briefly
    return () => window.clearTimeout(t);
  }, [saved]);

  const setDraft = useCallback((next: Draft) => {
    kept = next;
    setDraftState(next);
    setSaved(false);
  }, []);

  const byKey = Object.fromEntries((view?.settings ?? []).map((s) => [s.key, s]));
  const fieldOf = (s: SettingDef): string | boolean => (s.key in draft ? draft[s.key] : shownAs(s.value));
  const parsed = Object.fromEntries(Object.entries(draft).filter(([k]) => byKey[k]).map(([k, raw]) => [k, parse(byKey[k], raw)]));
  const problems = Object.fromEntries(Object.entries(parsed).filter(([, p]) => p.problem).map(([k, p]) => [k, p.problem!]));
  const changes = Object.fromEntries(
    Object.entries(parsed).filter(([k, p]) => p.value !== undefined && !same(p.value, byKey[k].value)).map(([k, p]) => [k, p.value!]),
  );
  const changed = Object.keys(changes).map((k) => byKey[k]);
  const dirty = changed.length > 0 || Object.keys(problems).length > 0;

  // leaving the page with changes not saved asks first
  useEffect(() => {
    if (!dirty) return;
    const warn = (e: BeforeUnloadEvent) => e.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  const edit = (s: SettingDef, raw: string | boolean) => {
    const { [s.key]: _gone, ...rest } = refused;
    setRefused(rest);
    setDraft({ ...draft, [s.key]: raw });
  };

  const save = async () => {
    if (!changed.length || Object.keys(problems).length) return;
    setSaving(true);
    try {
      setView(await adminApi.saveSettings(changes));
      setDraft({});
      setRefused({});
      setSaved(true);
      problem(null);
      refreshOverview();
    } catch (e) {
      if (e instanceof SettingsRefused) setRefused(e.errors);
      problem((e as Error).message);
    } finally {
      setSaving(false);
    }
  };

  if (!view) return <Section title="设置">{<p className="adm-lede">读取中…</p>}</Section>;
  const restartOnes = changed.filter((s) => s.restart);
  return (
    <Section
      title="设置"
      className="settings"
      lede={
        <>
          鼠标停在每一项上看它管什么、改的时候要考虑什么。标「需重启」的改了以后要重启服务才生效，其余保存后立刻生效。
          保存在服务器的 <code>{view.file}</code>，只记和默认不一样的值。
        </>
      }
    >
      <div className="set-groups">
        <NoticeCard />
        {view.groups.map((g) => (
          <div key={g.id} className="set-card" data-group={g.id}>
            <h3>{g.label}</h3>
            {view.settings
              .filter((s) => s.group === g.id)
              .map((s) => {
                const off = !!s.only_if && !(fieldOf(byKey[s.only_if]) === true);
                return (
                  <SettingRow
                    key={s.key}
                    s={s}
                    pending={view.pending.includes(s.key)}
                    field={fieldOf(s)}
                    off={off ? byKey[s.only_if].label : ""}
                    problem={problems[s.key] ?? refused[s.key] ?? ""}
                    onEdit={(raw) => edit(s, raw)}
                  />
                );
              })}
            {(view.status[g.id] ?? []).map((row) => (
              <StatusLine key={row.label} row={row} />
            ))}
          </div>
        ))}
        <PasswordCard />
      </div>
      {(dirty || saved) && (
        <div className={`set-bar glass${dirty ? "" : " done"}`}>
          {dirty ? (
            <>
              <span className="set-bar-text">
                {changed.length ? `改了 ${changed.length} 条：${changed.map((s) => s.label).join("、")}` : "有填得不对的设置"}
                {restartOnes.length > 0 && <span className="set-bar-note">其中 {restartOnes.map((s) => s.label).join("、")} 要重启服务才生效</span>}
              </span>
              <Button tip="放弃还没保存的修改" tone="ghost" onClick={() => (setDraft({}), setRefused({}))}>
                还原
              </Button>
              <Button
                tip={Object.keys(problems).length ? "先改好紫框里的项" : "服务器再检查一遍，全部通过才保存"}
                tone="primary"
                disabled={saving || !changed.length || Object.keys(problems).length > 0}
                onClick={() => void save()}
              >
                {saving ? "保存中…" : "保存"}
              </Button>
            </>
          ) : (
            <span className="set-bar-text">已保存{view.pending.length ? `：${view.pending.map((k) => byKey[k].label).join("、")} 重启服务后生效` : "，已经生效"}</span>
          )}
        </div>
      )}
    </Section>
  );
}

function SettingRow({ s, pending, field, off, problem, onEdit }: {
  s: SettingDef;
  pending: boolean; // saved, waiting for a restart
  field: string | boolean;
  off: string; // the switch it needs, when that is off
  problem: string;
  onEdit: (raw: string | boolean) => void;
}) {
  const tip = s.tip; // 服务器生成的悬停说明（config.py Setting.describe）：设置项的作用及其默认值
  const isDefault = same(parse(s, field).value, s.default);
  let control: React.ReactNode;
  if (!s.admin) control = <span className="set-static mono" data-user-data data-tip={String(s.running)}>{String(s.running)}</span>;
  else if (s.kind === "bool")
    control = <Switch on={!!field} onChange={onEdit} label={s.label} />;
  else if (s.kind === "choice")
    control = (
      <Segmented label={s.label} value={String(field)} options={s.options.map((o) => ({ value: o.value, label: o.label, tip: `${s.label}：${o.label}` }))} onChange={onEdit} />
    );
  else if (s.kind === "text" || s.kind === "list")
    control = <input className="field" value={String(field)} data-tip={tip} placeholder={`空：${s.empty}`} spellCheck={false} aria-label={s.label} onChange={(e) => onEdit(e.target.value)} />;
  else
    control = (
      <span className="set-num">
        <input className="field num" inputMode="decimal" value={String(field)} data-tip={tip} aria-label={s.label} onChange={(e) => onEdit(e.target.value)} />
        {s.unit && <span className="set-unit">{s.unit}</span>}
      </span>
    );
  return (
    <div className={`set-row${off ? " off" : ""}${problem ? " bad" : ""}`} data-key={s.key}>
      <span className="set-label" data-tip={tip}>
        {s.label}
      </span>
      <span className={`set-ctl set-${s.kind}`} data-tip={tip}>
        {control}
      </span>
      <span className="set-tags">
        {s.restart && (
          <span className="chip set-restart" data-tip={`改了要重启服务才生效：${s.why}`}>
            需重启
          </span>
        )}
        {pending && (
          <span className="chip set-pending" data-tip={`已保存 ${s.value_text}，服务现在还按 ${s.running_text} 运行：重启服务后生效`}>
            待重启
          </span>
        )}
        {s.overridden && (
          <span className="chip set-over" data-tip={`这次按${s.overridden}运行：${s.running_text}。这里保存的值，下次不带它启动时才生效`}>
            {s.overridden.startsWith("环境变量") ? "环境变量" : "启动参数"}
          </span>
        )}
        {s.source && (
          <span className="chip set-source" data-tip={s.source_tip}>
            {s.source}
          </span>
        )}
        {s.admin && !isDefault && (
          <Button tip={`恢复默认：${s.default_text}`} tone="ghost" layout="set-reset" aria-label="恢复默认" onClick={() => onEdit(shownAs(s.default))}>
            ↺
          </Button>
        )}
      </span>
      {problem && <span className="set-why bad">{problem}</span>}
      {!problem && off && <span className="set-why">打开「{off}」才起作用</span>}
      {!problem && !off && !s.admin && <span className="set-why">只能在服务器的 config/local.toml 里改</span>}
    </div>
  );
}

function StatusLine({ row }: { row: StatusRow }) {
  return (
    <div className="set-row set-status">
      <span className="set-label" data-tip={row.tip}>
        {row.label}
      </span>
      <span className="set-ctl set-static" data-user-data data-tip={`${row.value}\n${row.tip}`}>
        {row.value}
      </span>
    </div>
  );
}
