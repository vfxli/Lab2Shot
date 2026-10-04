/** An account's login picture (GET /api/admin/users/{id}/logins): where it is online right now and the 7-/30-day
 * summary with the 异地频繁 warning. The attempts themselves are a registered resource (lab2shot/site/resources.py
 * login_log), listed by the one shared table on the account's page — this part only says what a table of rows cannot:
 * whether the pattern of logins looks like a leaked account. */

import { useEffect, useState } from "react";
import { adminApi, type UserLogins } from "../api/admin";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

const KIND_LABEL: Record<string, () => string> = { web: () => t("ui.admin.logins.web"), client: () => t("ui.admin.logins.client") };

export function LoginsSummary({ user }: { user: number }) {
  const [data, setData] = useState<UserLogins | null>(null);
  useEffect(() => {
    let alive = true;
    adminApi.userLogins(user).then(
      (d) => alive && setData(d),
      () => alive && setData(null),
    );
    return () => {
      alive = false;
    };
  }, [user]);
  if (!data) return null;
  const where = data.online.map((o) => `${KIND_LABEL[o.kind]?.() ?? o.kind} · ${o.device} · ${o.ip}`);
  // a tile says one short thing and keeps the detail on hover: UI text never wraps and is never cut
  const span = (k: "7d" | "30d") => {
    const s = data.summary[k];
    return { short: t("ui.admin.logins.count", { n: s.logins }), full: t("ui.admin.logins.full", { n: s.logins, ips: s.ips, devices: s.devices, failures: s.failures }) };
  };
  return (
    <div className="usr-logins">
      <div className="adm-tiles">
        <div className="adm-tile" {...tipAttrs(tipOf("value", where.join("\n") || undefined))}>
          <span className="adm-tile-label">{t("ui.admin.logins.online_now")}</span>
          <b>{data.online.length ? data.online.map((o) => KIND_LABEL[o.kind]?.() ?? o.kind).join(t("list.sep")) : t("ui.admin.logins.offline")}</b>
        </div>
        {(["7d", "30d"] as const).map((k) => {
          const s = span(k);
          return (
            <div key={k} className="adm-tile" {...tipAttrs(tipOf("value", s.full))}>
              <span className="adm-tile-label">{k === "7d" ? t("ui.admin.logins.days7") : t("ui.admin.logins.days30")}</span>
              <b className="tnum">{s.short}</b>
            </div>
          );
        })}
      </div>
      {data.summary.suspicious && (
        <p className="usr-suspicious" role="alert" {...tipAttrs(tipOf("error", data.threshold_tip))}>
          {t("ui.admin.logins.suspicious")}
        </p>
      )}
    </div>
  );
}
