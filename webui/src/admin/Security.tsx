import { adminApi, type SecurityView } from "../api/admin";
import { onlineTip, Section, useAdmin } from "./common";
import { whenSecondsText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { t } from "../i18n/t";
import { tipAttrs } from "../platform/tips";

/** 安全: how many are logged in (lab2shot/server/auth.py), and what looked like probing — failed logins, refused
 * routes, odd paths, floods — with the sources blocked for a while. The accounts themselves are 用户 (Users.tsx). */


export function SecuritySection() {
  const { problem, go } = useAdmin();
  const { data: security, reload: reload } = usePoll(adminApi.security, 5000, { onError: (e) => problem(reasonOf(e)) });

  const unblock = async (client: string) => {
    try {
      await adminApi.unblock(client);
      problem(null);
    } catch (e) {
      problem((e as Error).message);
    }
    reload();
  };

  return (
    <Section title={t("ui.admin.security.title")} lede={t("ui.admin.security.lede")}>
      <div className="adm-tiles">
        <button className="adm-tile" {...tipAttrs(onlineTip(security?.online))} onClick={() => go("users")}>
          <span className="adm-tile-label">{t("ui.admin.security.online")}</span>
          <b>{security ? t("ui.admin.security.online_value", { browser: security.online.browser, client: security.online.client }) : "…"}</b>
        </button>
        <div className="adm-tile">
          <span className="adm-tile-label">{t("ui.admin.security.forgot")}</span>
          <b>{t("ui.admin.security.passphrase")}</b>
        </div>
      </div>
      <Watch view={security} onUnblock={(c) => void unblock(c)} />
    </Section>
  );
}

function Watch({ view, onUnblock }: { view: SecurityView | null; onUnblock: (client: string) => void }) {
  if (!view) return <p className="adm-lede">{t("ui.admin.common.reading")}</p>;
  const l = view.limits;
  return (
    <>
      <h3 className="adm-h3">{t("ui.admin.security.suspicious")}</h3>
      <p className="adm-lede">
        {t("ui.admin.security.rules", { block_window_min: l.block_window_min, block_after: l.block_after, block_min: l.block_min, window_min: l.window_min, free: l.free, client_lock: l.client_lock, address_free: l.address_free, address_lock: l.address_lock, subject_free: l.subject_free, max_wait_s: l.max_wait_s })}
      </p>
      {Object.keys(view.counts).length > 0 && (
        <div className="sec-row">
          {Object.entries(view.counts).map(([k, n]) => (
            <span key={k} className="chip">
              {k} {n}
            </span>
          ))}
        </div>
      )}
      {view.blocked.length > 0 && (
        <table className="sec-table">
          <thead>
            <tr>
              <th>{t("ui.admin.security.col_blocked")}</th>
              <th>{t("ui.admin.security.col_account")}</th>
              <th>{t("ui.admin.security.col_reason")}</th>
              <th>{t("ui.admin.security.col_until")}</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {view.blocked.map((b) => (
              <tr key={b.client}>
                <td>{b.who}</td>
                <td className="mono">{b.user || "—"}</td>
                <td>{b.why}</td>
                <td className="tnum">{whenSecondsText(b.until)}</td>
                <td>
                  <Button tone="ghost" onClick={() => onUnblock(b.client)}>
                    {t("ui.admin.security.unblock")}
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {view.events.length ? (
        <div className="q-table-wrap">
          <table className="sec-table">
            <thead>
              <tr>
                <th>{t("ui.admin.security.col_time")}</th>
                <th>{t("ui.admin.security.col_what")}</th>
                <th>{t("ui.admin.security.col_source")}</th>
                <th>{t("ui.admin.security.col_account")}</th>
                <th>{t("ui.admin.security.col_request")}</th>
                <th>{t("ui.admin.security.col_detail")}</th>
              </tr>
            </thead>
            <tbody>
              {view.events.map((e, i) => (
                <tr key={`${e.t}-${i}`}>
                  <td className="tnum">{whenSecondsText(e.t)}</td>
                  <td>
                    {e.kind}{e.counted ? "" : t("ui.admin.security.not_counted")}
                  </td>
                  <td className="mono">
                    {e.who}
                  </td>
                  <td className="mono">
                    {e.user || "—"}
                  </td>
                  <td className="mono">
                    {e.method} {e.path}
                  </td>
                  <td>{e.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="adm-empty">{t("ui.admin.security.none")}</p>
      )}
    </>
  );
}
