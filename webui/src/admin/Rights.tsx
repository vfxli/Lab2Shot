import { useEffect, useState } from "react";
import { adminApi, type RightsSheetView, type RightsView } from "../api/admin";
import { shown, usable, why } from "../api/applies";
import type { Availability } from "../api/applies";
import { Button, Switch } from "../ui/Button";
import { Sheet } from "../ui/Sheet";
import { Loading } from "../ui/Loading";
import { reasonOf } from "../messages/message";
import { whenText } from "../platform/format";
import "./rights.css";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** 二级管理员权限：一级管理员为二级管理员开放的权限，每一项为可查看、可修改或不可见。
 *
 * 每条权限本身即对应「可查看 / 可修改 / 不可见」三档之一：部分权限仅允许查看（如查看统计），部分允许修改（如回复反馈），
 * 未勾选任何权限的后台区块不显示（可见性由服务器按权限计算，lab2shot/availability.py 中 CAPABILITY 档 =
 * 隐藏）。因此此处不另设三态开关，只提供一列勾选框，每项附带说明，表明勾选后授予的是查看还是修改。
 *
 * 排版完成的表格由服务器提供（lab2shot/roles.py sheet()）：分段、短名、说明、勾选状态和数量。本模块只负责绘制，
 * 不分组、不计数、不识别任何权限名称，也不检查角色；是否显示仅由 rights.edit 决定。 */
export function Rights({ applies }: { applies: Availability | null | undefined }) {
  const [view, setView] = useState<RightsView | null>(null);
  const [problem, setProblem] = useState("");
  const may = shown(applies, "rights.edit");

  useEffect(() => {
    if (!may) return;
    let live = true;
    adminApi.rights().then(
      (v) => live && (setView(v), setProblem("")),
      (e) => live && (setView(null), setProblem(reasonOf(e))),
    );
    return () => {
      live = false;
    };
  }, [may]);

  if (!may) return null;
  if (problem) return <div className="notice" role="alert">{t("ui.admin.rights.unread", { problem })}</div>;
  return (
    <>
      {(view?.sheets ?? [null]).map((s, i) => (
        <RightsBand key={s?.role ?? i} sheet={s} applies={applies} onChanged={setView} />
      ))}
    </>
  );
}

/** 单个角色的区块：一句现状说明和打开勾选表的按钮。 */
function RightsBand({ sheet, applies, onChanged }: {
  sheet: RightsSheetView | null;
  applies: Availability | null | undefined;
  onChanged: (v: RightsView) => void;
}) {
  const [open, setOpen] = useState(false);
  const name = sheet?.label ?? t("ui.admin.rights.deputy");
  const summary = !sheet
    ? ""
    : sheet.default
      ? t("ui.admin.rights.summary_default", { count: sheet.count, total: sheet.total })
      : t("ui.admin.rights.summary_changed", { count: sheet.count, total: sheet.total, who: sheet.updated_by || t("ui.admin.rights.admin"), when: whenText(sheet.updated) });

  return (
    <section className="rts">
      <div className="rts-head">
        <h3>{t("ui.admin.rights.title", { role: name })}</h3>
        <span className="dim rts-sum">
          {sheet ? summary : <Loading what={t("ui.admin.rights.loading")} />}
        </span>
        <Button
          tip={usable(applies, "rights.edit") ? undefined : tipOf("disabled", why(applies, "rights.edit"))}
          disabled={!usable(applies, "rights.edit") || !sheet}
          onClick={() => setOpen(true)}
        >
          {t("ui.admin.rights.assign")}
        </Button>
      </div>
      {open && sheet && <RightsSheet sheet={sheet} onClose={() => setOpen(false)} onSaved={onChanged} />}
    </section>
  );
}

/** 勾选表：按服务器给出的分段逐行列出（开关 + 短名 + 说明），底部为恢复默认、取消和保存。 */
function RightsSheet({ sheet, onClose, onSaved }: {
  sheet: RightsSheetView;
  onClose: () => void;
  onSaved: (v: RightsView) => void;
}) {
  const start = sheet.groups.flatMap((g) => g.items.filter((x) => x.on).map((x) => x.id));
  const [picked, setPicked] = useState<string[]>(start);
  const [busy, setBusy] = useState("");
  const [problem, setProblem] = useState("");
  const chosen = new Set(picked);
  const same = picked.length === start.length && start.every((c) => chosen.has(c));

  const toggle = (id: string, on: boolean) => {
    setPicked((was) => (on ? [...was, id] : was.filter((x) => x !== id)));
    setProblem("");
  };

  const run = async (what: "save" | "default") => {
    setBusy(what);
    try {
      const got = what === "save" ? await adminApi.setRights(sheet.role, picked) : await adminApi.resetRights(sheet.role);
      onSaved(got);
      onClose();
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy("");
    }
  };

  return (
    <Sheet title={t("ui.admin.rights.title", { role: sheet.label })} width={560} onClose={onClose}>
      <div className="rts-form">
        <p className="rts-note">
          {t("ui.admin.rights.note", { role: sheet.label })}
        </p>
        <div className="rts-list">
          {sheet.groups.map((g) => (
            <div className="rts-group" key={g.label}>
              <h4>{g.label}</h4>
              {g.items.map((x) => (
                <label className="rts-row" key={x.id}>
                  <Switch on={chosen.has(x.id)} mini label={x.label} onChange={(on) => toggle(x.id, on)} />
                  <span className="rts-name">{x.label}</span>
                  <span className="rts-what dim">{x.what}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
        {problem && (
          <p className="login-problem" role="alert">
            {problem}
          </p>
        )}
        <div className="dialog-row">
          <Button tip={tipOf("consequence", t("ui.admin.rights.default_tip", { n: sheet.default_count }))} tone="ghost" disabled={!!busy} onClick={() => void run("default")}>
            {t("ui.admin.common.restore_default")}
          </Button>
          <span className="rts-gap" />
          <Button tone="ghost" onClick={onClose}>
            {t("ui.admin.common.cancel")}
          </Button>
          <Button tip={same ? tipOf("disabled", t("ui.admin.common.unchanged")) : tipOf("consequence", t("ui.admin.rights.save_tip", { role: sheet.label, n: picked.length }))} tone="primary" disabled={same || !!busy} onClick={() => void run("save")}>
            {busy === "save" ? t("ui.admin.common.saving") : t("ui.admin.common.save")}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
