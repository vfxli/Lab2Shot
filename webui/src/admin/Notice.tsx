import { useEffect, useState } from "react";
import { LabelRow } from "../ui/LabelRow";
import { adminApi } from "../api/admin";
import { Banner, type BannerTone } from "../ui/Banner";
import { Button, Segmented, Switch } from "../ui/Button";
import { reasonOf } from "../messages/message";
import { shown, usable, why } from "../api/applies";
import { useSignedIn } from "../state/session";
import { whenText } from "../platform/format";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 管理员通知 section: a single line shown at the top of the editor and the admin page — plain text,
 * a colour indicating its severity, and an on/off switch. It is server state (lab2shot/server/notice.py, meta
 * server.notice); every page picks up changes through its existing poll of the server state, so nothing is pushed.
 *
 * Red is not offered: it is reserved for conditions that can cause a production incident. Whether this card is
 * shown is decided by the server (settings.notice through available.py), not by a role check here. */

const CHARS = 200; // Must match lab2shot/server/notice.py NOTICE_CHARS.

const tones = (): { value: BannerTone; label: string }[] => [
  { value: "info", label: t("ui.admin.notice.tone_info") },
  { value: "notice", label: t("ui.admin.notice.tone_notice") },
  { value: "warn", label: t("ui.admin.notice.tone_warn") },
];

export function NoticeCard() {
  const state = useSignedIn();
  const may = shown(state?.applies, "settings.notice");
  const [text, setText] = useState("");
  const [tone, setTone] = useState<BannerTone>("info");
  const [on, setOn] = useState(false);
  const [saved, setSaved] = useState<{ text: string; tone: string; on: boolean; updated: number; by?: string } | null>(null);
  const [problem, setProblem] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!may) return;
    let live = true;
    adminApi.notice().then(
      (n) => live && (setSaved(n), setText(n.text), setTone(n.tone as BannerTone), setOn(n.on)),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [may]);

  if (!may) return null;
  const left = CHARS - [...text].length;
  // Input the server would reject is reported here first, so a visible error never leaves the page.
  const blocked = left < 0 ? t("ui.admin.notice.too_long", { most: CHARS, over: -left }) : on && !text.trim() ? t("ui.admin.notice.empty") : "";
  // Nothing is saved before the live notice has been read: an empty form saved over it would wipe it.
  const unread = saved ? "" : problem ? t("ui.admin.notice.unread") : t("ui.admin.notice.reading");
  const dirty = !!saved && (saved.text !== text.trim() || saved.tone !== tone || saved.on !== on);

  const save = async () => {
    setSaving(true);
    try {
      const n = await adminApi.setNotice({ text: text.trim(), tone, on });
      setSaved(n);
      setText(n.text);
      setProblem("");
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="set-card lgrid" data-group="notice">
      <h3>{t("ui.admin.notice.title")}</h3>
      <p className="adm-lede">
        {t("ui.admin.notice.lede")}
      </p>
      <LabelRow className="set-row" data-key="notice.text" labelClass="set-label" label={t("ui.admin.notice.text")} ctlClass="set-ctl" tail={<span className="set-tags">
          <span className="set-what">
            <span className={`chip tnum${left < 0 ? " set-over" : ""}`} {...tipAttrs(tipOf("value", t("ui.admin.notice.left", { n: left })))}>
              {left}
            </span>
          </span>
        </span>}>
        <textarea
            className="field notice-text"
            value={text}
            rows={2}
            aria-label={t("ui.admin.notice.text_label")}
            placeholder={t("ui.admin.notice.placeholder")}
            onChange={(e) => setText(e.target.value)}
          />
      </LabelRow>
      <LabelRow className="set-row" data-key="notice.tone" labelClass="set-label" label={t("ui.admin.notice.tone")} ctlClass="set-ctl">
        <Segmented label={t("ui.admin.notice.tone_label")} value={tone} options={tones()} onChange={(v) => setTone(v)} />
      </LabelRow>
      <LabelRow className="set-row" data-key="notice.on" labelClass="set-label" label={t("ui.admin.notice.show")} ctlClass="set-ctl">
        <Switch on={on} label={t("ui.admin.notice.show_label")} tip={on ? tipOf("consequence", t("ui.admin.notice.show_tip")) : undefined} onChange={setOn} />
      </LabelRow>
      <LabelRow className="set-row set-status" labelClass="set-label" label={t("ui.admin.notice.last_saved")} ctlClass="set-ctl set-static">
        <span data-user-data>{saved?.updated ? `${saved.by ?? t("ui.admin.rights.admin")} · ${whenText(saved.updated)}` : t("ui.admin.notice.never_set")}</span>
      </LabelRow>
      <LabelRow className="set-row notice-preview" labelClass="set-label" label={t("ui.admin.notice.preview")} ctlClass="set-ctl">
        {text.trim() ? (
            <Banner tone={tone} aside={t("ui.admin.notice.title")}>
              {text.trim()}
            </Banner>
          ) : (
            <span className="dim">{t("ui.admin.notice.preview_empty")}</span>
          )}
      </LabelRow>
      {(problem || blocked) && <div className="set-why bad lrow-under">{problem || blocked}</div>}
      <LabelRow className="set-row" labelClass="set-label" label="" ctlClass="set-ctl">
        {/* Not the page's primary (filled) button: 设置 already has one, and a page has at most one. */}
          <Button
            tip={unread || blocked || why(state?.applies, "settings.notice") ? tipOf("disabled", unread || blocked || why(state?.applies, "settings.notice")) : tipOf("consequence", t("ui.admin.notice.save_tip"))}
            disabled={saving || !dirty || !!unread || !!blocked || !usable(state?.applies, "settings.notice")}
            onClick={() => void save()}
          >
            {saving ? t("ui.admin.common.saving") : dirty || !saved ? t("ui.admin.notice.save") : t("ui.admin.common.saved")}
          </Button>
      </LabelRow>
    </div>
  );
}
