import { useCallback, useEffect, useState } from "react";
import { LabelGrid, LabelRow } from "../ui/LabelRow";
import { same as sameJson, type Json } from "../model/graphPatch";
import { adminApi, SettingsRefused, type SettingDef, type SettingValue, type StatusRow } from "../api/admin";
import type { SettingsPageEntry } from "../api";
import { shown } from "../api/applies";
import { useSignedIn } from "../state/session";
import { Section, useAdmin } from "./common";
import { ListField } from "./ListField";
import { Button, Segmented, Switch } from "../ui/Button";
import { Checks } from "./userFields";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** A settings page of the admin page's 设置 band (lab2shot/config.py PAGES): the settings of its groups, one card per
 * group, each setting with its note under it (what it does, what a change leads to); the ones that take effect after a restart say so (需重启) and, once saved,
 * wait for it (待重启). Changes are checked on the page (range, numbers, a list's rows) and again on the server
 * (paths, ports), and saved together. What the server can tell that goes with a group (free memory, the address, the
 * Hugging Face login) is shown with it, read only.
 *
 * The settings are there only for a login that reads them (settings.values). A page's parts with a right of their own
 * decide for themselves whether this login sees them: `cards`, set cards that flow with the settings' cards (the
 * 管理员通知 on 账号设置), and `children`, shown below them (the invite codes on 注册设置). */

type Field = string | boolean | string[]; // what a field holds: numbers as typed, a list as its rows (blank ones too), a multi its ticked options
type Draft = Record<string, Field>; // key -> field

const kept: Record<string, Draft> = {}; // page -> its unsaved changes, which outlive switching to another section

/** What a field holds for a value (a list as its rows, a multi as its ticked options). */
const shownAs = (v: SettingValue): Field => (typeof v === "boolean" ? v : Array.isArray(v) ? [...v] : String(v));
const same = (a: SettingValue | undefined, b: SettingValue) => sameJson(a as Json | undefined, b as Json);

/** A list's rows that repeat an earlier row (blank rows are left out, as the value leaves them out). */
function repeats(rows: string[]): Set<number> {
  const seen = new Set<string>();
  const out = new Set<number>();
  rows.forEach((raw, i) => {
    const x = raw.trim();
    if (!x) return;
    if (seen.has(x)) out.add(i);
    seen.add(x);
  });
  return out;
}

/** The field's content as the value to save, or why it can't be (the same words as the server). */
function parse(s: SettingDef, raw: Field): { value?: SettingValue; problem?: string } {
  if (s.kind === "multi") {
    const picked = raw as string[];
    return { value: s.options.map((o) => o.value).filter((v) => picked.includes(v)) }; // in the options' order, as the server keeps it
  }
  if (s.kind === "list") {
    const rows = raw as string[];
    const items = rows.map((x) => x.trim()).filter(Boolean);
    if (!items.length) return { problem: t("ui.admin.settings.list_empty", { label: s.label }) };
    const twice = [...repeats(rows)].map((i) => rows[i].trim());
    if (twice.length) return { problem: t("ui.admin.settings.list_repeats", { label: s.label, items: [...new Set(twice)].join(t("list.sep")) }) };
    return { value: items };
  }
  if (s.kind === "bool" || s.kind === "choice" || s.kind === "text") return { value: typeof raw === "string" ? raw.trim() : (raw as boolean) };
  const text = String(raw).trim();
  const n = Number(text);
  if (!text || !Number.isFinite(n)) return { problem: t("ui.admin.settings.not_number", { label: s.label }) };
  if (s.kind === "int" && !Number.isInteger(n)) return { problem: t("ui.admin.settings.not_int", { label: s.label }) };
  if ((s.min !== null && n < s.min) || (s.max !== null && n > s.max)) return { problem: t("ui.admin.settings.range", { label: s.label, min: String(s.min), max: String(s.max), unit: s.unit ? ` ${s.unit}` : "" }) };
  return { value: n };
}

export function SettingsPage({ page, cards, children }: { page: SettingsPageEntry; cards?: React.ReactNode; children?: React.ReactNode }) {
  const state = useSignedIn();
  const reads = shown(state?.applies, "settings.values");
  return (
    <Section
      title={page.label}
      className="settings"
      lede={
        reads ? (
          <>{t("ui.admin.settings.lede")}</>
        ) : undefined
      }
    >
      {reads ? (
        <SettingsForm page={page.id} cards={cards} />
      ) : (
        cards && <div className="set-groups">{cards}</div>
      )}
      {children}
    </Section>
  );
}

function SettingsForm({ page, cards }: { page: string; cards?: React.ReactNode }) {
  const { problem, refreshOverview, settings: view, settingsSaved: setView } = useAdmin();
  const [draft, setDraftState] = useState<Draft>(kept[page] ?? {});
  const [refused, setRefused] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    if (!saved) return;
    const timer = window.setTimeout(() => setSaved(false), 6000); // "已保存" is shown briefly
    return () => window.clearTimeout(timer);
  }, [saved]);

  const setDraft = useCallback(
    (next: Draft) => {
      kept[page] = next;
      setDraftState(next);
      setSaved(false);
    },
    [page],
  );

  const byKey = Object.fromEntries((view?.settings ?? []).map((s) => [s.key, s]));
  const fieldOf = (s: SettingDef): Field => (s.key in draft ? draft[s.key] : shownAs(s.value));
  const parsed = Object.fromEntries(Object.entries(draft).filter(([k]) => byKey[k]).map(([k, raw]) => [k, parse(byKey[k], raw)]));
  const problems = Object.fromEntries(Object.entries(parsed).filter(([, p]) => p.problem).map(([k, p]) => [k, p.problem!]));
  const changes = Object.fromEntries(
    Object.entries(parsed).filter(([k, p]) => p.value !== undefined && !same(p.value, byKey[k].value)).map(([k, p]) => [k, p.value!]),
  );
  const changed = Object.keys(changes).map((k) => byKey[k]);
  const dirty = changed.length > 0 || Object.keys(problems).length > 0;

  // leaving the page with changes not saved, on this settings page or another one, asks first
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (dirty || Object.entries(kept).some(([p, d]) => p !== page && Object.keys(d).length)) e.preventDefault();
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty, page]);

  const edit = (s: SettingDef, raw: Field) => {
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

  if (!view) return <p className="adm-lede">{t("ui.admin.common.reading")}</p>;
  const groups = view.pages.find((p) => p.id === page)?.groups ?? [];
  const pending = view.pending.filter((k) => groups.some((g) => g.id === byKey[k].group));
  const restartOnes = changed.filter((s) => s.restart);
  return (
    <>
      <p className="adm-lede set-file">
        {t("ui.admin.settings.file_before")}<code>{view.file}</code>{t("ui.admin.settings.file_after")}
      </p>
      <div className="set-groups">
        {groups.map((g) => (
          <LabelGrid key={g.id} className="set-card" data-group={g.id}>
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
          </LabelGrid>
        ))}
        {cards}
      </div>
      {(dirty || saved) && (
        <div className={`set-bar glass${dirty ? "" : " done"}`}>
          {dirty ? (
            <>
              <span className="set-bar-text">
                {changed.length ? t("ui.admin.settings.changed", { n: changed.length, labels: changed.map((s) => s.label).join(t("list.sep")) }) : t("ui.admin.settings.invalid")}
                {restartOnes.length > 0 && <span className="set-bar-note">{t("ui.admin.settings.restart_ones", { labels: restartOnes.map((s) => s.label).join(t("list.sep")) })}</span>}
              </span>
              <Button tip={tipOf("consequence", t("ui.admin.settings.revert_tip"))} tone="ghost" onClick={() => (setDraft({}), setRefused({}))}>
                {t("ui.admin.common.revert")}
              </Button>
              <Button
                tip={Object.keys(problems).length ? tipOf("disabled", t("ui.admin.settings.fix_first")) : undefined}
                tone="primary"
                disabled={saving || !changed.length || Object.keys(problems).length > 0}
                onClick={() => void save()}
              >
                {saving ? t("ui.admin.common.saving") : t("ui.admin.common.save")}
              </Button>
            </>
          ) : (
            <span className="set-bar-text">{pending.length ? t("ui.admin.settings.saved_pending", { labels: pending.map((k) => byKey[k].label).join(t("list.sep")) }) : t("ui.admin.settings.saved_now")}</span>
          )}
        </div>
      )}
    </>
  );
}

function SettingRow({ s, pending, field, off, problem, onEdit }: {
  s: SettingDef;
  pending: boolean; // saved, waiting for a restart
  field: Field;
  off: string; // the switch it needs, when that is off
  problem: string;
  onEdit: (raw: Field) => void;
}) {
  const isDefault = same(parse(s, field).value, s.default);
  let control: React.ReactNode;
  const fixed = !s.admin || !!s.locked; // shown, not changed here
  if (fixed) control = <span className="set-static mono" data-user-data {...tipAttrs(tipOf("truncated", String(s.running)))}>{String(s.running)}</span>;
  else if (s.kind === "bool")
    control = <Switch on={!!field} onChange={onEdit} label={s.label} />;
  else if (s.kind === "choice")
    control = (
      <Segmented label={s.label} value={String(field)} options={s.options.map((o) => ({ value: o.value, label: o.label }))} onChange={onEdit} />
    );
  else if (s.kind === "multi")
    control = (
      <Checks options={s.options.map((o) => ({ id: o.value, label: o.label }))}
        value={field as string[]} onChange={onEdit} />
    );
  else if (s.kind === "list")
    control = <ListField label={s.label} items={field as string[]} twice={repeats(field as string[])} onChange={onEdit} shown={s.item_labels} />;
  else if (s.kind === "text")
    control = <input className="field" value={String(field)} placeholder={t("ui.admin.settings.empty_means", { means: s.empty })} spellCheck={false} aria-label={s.label} onChange={(e) => onEdit(e.target.value)} />;
  else
    control = (
      <span className="set-num">
        <input className="field num" inputMode="decimal" value={String(field)} aria-label={s.label} onChange={(e) => onEdit(e.target.value)} />
        {s.unit && <span className="set-unit">{s.unit}</span>}
      </span>
    );
  return (
    // the site's one 「标签 + 控件」 row (ui/LabelRow.tsx): the label's column is the card's longest label (LabelGrid),
    // a label too long for it cut with its whole on hover, never over the control; the marks are its tail
    <LabelRow className={`set-row${s.kind === "list" ? " tall top" : ""}${off ? " off" : ""}${problem ? " bad" : ""}`} data-key={s.key}
      labelClass="set-label" label={s.label} ctlClass={`set-ctl set-${s.kind}`} note={s.note}
      below={<>
        {problem && <span className="set-why bad lrow-under">{problem}</span>}
        {!problem && off && <span className="set-why lrow-under">{t("ui.admin.settings.needs_switch", { name: off })}</span>}
        {!problem && !off && !s.admin && <span className="set-why lrow-under">{t("ui.admin.settings.file_only")}</span>}
        {!problem && !off && s.admin && s.locked && <span className="set-why lrow-under">{s.locked}</span>}
      </>}
      tail={<span className="set-tags">
        {/* 待重启 takes 需重启's place: the change it warned about is saved and waits for the restart */}
        {pending ? (
          <span className="chip set-when set-pending" {...tipAttrs(tipOf("value", t("ui.admin.settings.pending_tip", { value: s.value_text, running: s.running_text })))}>
            {t("ui.admin.settings.pending")}
          </span>
        ) : (
          s.restart && (
            <span className="chip set-when set-restart" {...tipAttrs(tipOf("consequence", t("ui.admin.settings.restart_tip", { why: s.why })))}>
              {t("ui.admin.settings.restart")}
            </span>
          )
        )}
        <span className="set-what">
          {s.overridden && (
            <span className="chip set-over" {...tipAttrs(tipOf("value", t("ui.admin.settings.overridden_tip", { by: s.overridden, running: s.running_text })))}>
              {s.overridden_kind === "env" ? t("ui.admin.settings.overridden_env") : t("ui.admin.settings.overridden_option")}
            </span>
          )}
          {s.source && (
            <span className="chip set-source" {...tipAttrs(tipOf("value", s.source_tip))}>
              {s.source}
            </span>
          )}
        </span>
        {!fixed && !isDefault && (
          <Button tip={tipOf("value", t("ui.admin.settings.default_tip", { value: s.default_text }))} tone="ghost" layout="set-reset" aria-label={t("ui.admin.common.restore_default")} onClick={() => onEdit(shownAs(s.default))}>
            ↺
          </Button>
        )}
      </span>}>
      {control}
    </LabelRow>
  );
}

function StatusLine({ row }: { row: StatusRow }) {
  return (
    <LabelRow className="set-row set-status" labelClass="set-label" label={row.label} ctlClass="set-ctl">
      <span className="set-static" data-user-data {...tipAttrs(tipOf("truncated", row.value))}>
        {row.value}
      </span>
    </LabelRow>
  );
}
