import { useEffect, useState } from "react";
import { adminApi, type UserChange, type UserResourceTab, type UserRow } from "../api/admin";
import { Button, Segmented, Switch } from "../ui/Button";
import { Empty } from "../ui/Empty";
import { Loading } from "../ui/Loading";
import { reasonOf } from "../messages/message";
import { ResourceTable } from "./Resources";
import { shown, usable, why } from "../api/applies";
import { LoginsSummary } from "./UserLogins";
import { UserQuota } from "./UserQuota";
import { UserRewards } from "./UserRewards";
import { DAY_S, Expiry, loginText, now } from "./Users";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 用户 / 一个账号: the account's own page — who it is and what may be done to it at the
 * top, then one tab per kind of resource it owns, with how many, and one shared table under whichever is open.
 *
 * The tabs come from the server's registry (GET /api/admin/users/{id}/resources, lab2shot/site/resources.py): this page
 * does not name a kind of resource anywhere, and shows only what this login may look at. What may be done to the
 * account comes from the same availability answer the list uses (`u.applies`), never from a role. */

export function UserDetail({ u, onBack, onEdit, onReset, onDelete, onChange }: {
  u: UserRow;
  onBack: () => void;
  onEdit: () => void;
  onReset: () => void;
  onDelete: () => void;
  onChange: (c: UserChange) => void; // 延期, 停用 / 启用, 队列优先: a change sent at once
}) {
  const [tabs, setTabs] = useState<UserResourceTab[] | null>(null);
  const [problem, setProblem] = useState("");
  const [kind, setKind] = useState("");

  useEffect(() => {
    let live = true;
    adminApi.userResources(u.id).then(
      (r) => live && (setTabs(r.tabs), setKind((k) => (r.tabs.some((t) => t.kind === k) ? k : (r.tabs[0]?.kind ?? "")))),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [u.id]);

  return (
    <section className="adm-sec usr-detail">
      <nav className="usr-crumb" aria-label={t("ui.admin.user.back")}>
        <Button tone="link" onClick={onBack}>
          {t("ui.admin.nav.users")}
        </Button>
        <span className="dim">/</span>
        <span data-user-data {...tipAttrs(tipOf("truncated", `${u.name} · ${u.username}`))}>{u.name || u.username}</span>
      </nav>
      <header className="usr-head">
        <span className="usr-face" aria-hidden>
          {[...(u.name || u.username)][0]}
        </span>
        <div className="usr-who">
          <h2 data-user-data {...tipAttrs(tipOf("truncated", u.name))}>{u.name || u.username}</h2>
          <div className="usr-meta dim">
            <span className="mono" data-user-data {...tipAttrs(tipOf("truncated", u.username))}>
              {u.username}
            </span>
            <span>{u.department_label || t("ui.admin.user.no_department")}</span>
            <span>{u.role_label}</span>
            <span {...tipAttrs(tipOf("truncated", u.state_tip))}>{u.state}</span>
            <Expiry u={u} />
            <span>{loginText(u.last_login)}</span>
          </div>
        </div>
        <UserActs u={u} onEdit={onEdit} onReset={onReset} onDelete={onDelete} onChange={onChange} />
      </header>
      <UserQuota user={u.id} applies={u.applies} />
      {usable(u.applies, "account.logins") && <LoginsSummary user={u.id} />}
      {usable(u.applies, "account.rewards") && <UserRewards user={u.id} />}
      {problem ? (
        <Empty title={problem} hint={t("ui.admin.user.reload_hint")} />
      ) : !tabs ? (
        <Loading what={t("ui.admin.user.records")} />
      ) : !tabs.length ? (
        <Empty title={t("ui.admin.user.records_hidden")} hint={t("ui.admin.user.records_hidden_hint")} />
      ) : (
        <>
          <Segmented
            label={t("ui.admin.user.records")}
            tabs
            size="md"
            value={kind}
            options={tabs.map((t) => ({
              value: t.kind,
              label: (
                <>
                  {t.label}
                  <b className="usr-tab-count tnum">{t.count}</b>
                </>
              ),
              field: `tab-${t.kind}`,
            }))}
            onChange={setKind}
            layout="usr-tabs"
          />
          {kind && <ResourceTable key={kind} user={u.id} kind={kind} />}
        </>
      )}
    </section>
  );
}

/** What may be done to one account, as the server resolved it for this login (`u.applies`). */
function UserActs({ u, onEdit, onReset, onDelete, onChange }: {
  u: UserRow;
  onEdit: () => void;
  onReset: () => void;
  onDelete: () => void;
  onChange: (c: UserChange) => void;
}) {
  const extend = (days: number) => onChange({ expires: Math.max(u.expires ?? now(), now()) + days * DAY_S });
  return (
    <div className="usr-acts">
      {usable(u.applies, "account.edit") && (
        <Button onClick={onEdit}>
          {t("ui.admin.user.edit")}
        </Button>
      )}
      {shown(u.applies, "account.expiry") && (
        <>
          <Button tip={tipOf("disabled", why(u.applies, "account.expiry"))} tone="ghost" disabled={!usable(u.applies, "account.expiry")} onClick={() => extend(30)}>
            {t("ui.admin.user.extend", { days: 30 })}
          </Button>
          <Button tip={tipOf("disabled", why(u.applies, "account.expiry"))} tone="ghost" disabled={!usable(u.applies, "account.expiry")} onClick={() => extend(90)}>
            {t("ui.admin.user.extend", { days: 90 })}
          </Button>
        </>
      )}
      {shown(u.applies, "account.enable") && (
        <Switch
          on={u.enabled}
          label={t("ui.admin.user.enabled_label", { username: u.username })}
          disabled={!usable(u.applies, "account.enable")}
          tip={why(u.applies, "account.enable") ? tipOf("disabled", why(u.applies, "account.enable")) : u.enabled ? tipOf("consequence", t("ui.admin.user.disable_tip")) : undefined}
          onChange={(enabled) => onChange({ enabled })}
        />
      )}
      {shown(u.applies, "account.queue_first") && (
        <label className="usr-flag">
          {t("ui.admin.user.queue_first")}
          <Switch
            on={u.queue_first}
            mini
            label={t("ui.admin.user.queue_first_label", { username: u.username })}
            disabled={!usable(u.applies, "account.queue_first")}
            tip={why(u.applies, "account.queue_first") ? tipOf("disabled", why(u.applies, "account.queue_first")) : tipOf("consequence", t("ui.admin.user.queue_first_tip"))}
            onChange={(queue_first) => onChange({ queue_first })}
          />
        </label>
      )}
      {usable(u.applies, "account.password") && (
        <Button tip={tipOf("consequence", t("ui.admin.user.reset_tip"))} tone="ghost" onClick={onReset}>
          {t("ui.admin.user.reset")}
        </Button>
      )}
      {shown(u.applies, "account.delete") && (
        <Button tip={tipOf("disabled", why(u.applies, "account.delete"))} tone="ghost" danger disabled={!usable(u.applies, "account.delete")} onClick={onDelete}>
          {t("ui.admin.user.delete")}
        </Button>
      )}
    </div>
  );
}
