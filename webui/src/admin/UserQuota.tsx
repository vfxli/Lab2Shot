import { useEffect, useState } from "react";
import { adminApi } from "../api/admin";
import type { AccountUsage } from "../api/library";
import { Button } from "../ui/Button";
import { Loading } from "../ui/Loading";
import { reasonOf } from "../messages/message";
import { gbText, sizeText } from "../platform/format";
import { shown, usable, why } from "../api/applies";
import type { Availability } from "../api/applies";
import { TRAFFIC } from "./traffic";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 这个账号占了多少资源：硬盘和流量在一块儿，都是「他占了多少」，一个回答、一块界面，不做成两处。
 * 硬盘还有上限，能按账号改。
 *
 * 显示什么由服务器算的可用性定（`account.quota` 看得到、`account.quota_set` 改得了），这里不看角色；
 * 每一块的中文名和说明也是服务器的（lab2shot/server/quota.py areas），这个文件一个都不写。 */
export function UserQuota({ user, applies }: { user: number; applies: Availability }) {
  const [view, setView] = useState<AccountUsage | null>(null);
  const [problem, setProblem] = useState("");
  const [typed, setTyped] = useState("");
  const [busy, setBusy] = useState(false);

  const may = shown(applies, "account.quota");
  useEffect(() => {
    if (!may) return; // 这个登录无权查看时不去请求（被拒绝会计入该会话）
    let live = true;
    setView(null);
    adminApi.userQuota(user).then(
      (v) => live && (setView(v), setTyped(v.own_gb === null ? "" : String(v.own_gb)), setProblem("")),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [user, may]);

  if (!may) return null;
  if (problem) return <div className="usr-quota dim">{problem}</div>;
  if (!view) return <Loading what={t("ui.admin.quota.loading")} />;

  const save = async (gb: number | null) => {
    setBusy(true);
    try {
      const got = await adminApi.setUserQuota(user, gb);
      setView(got);
      setTyped(got.own_gb === null ? "" : String(got.own_gb));
      setProblem("");
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy(false);
    }
  };

  const typedGb = Number(typed);
  const ok = typed.trim() === "" || (Number.isFinite(typedGb) && typedGb >= 1 && typedGb <= 10000);
  const changed = (typed.trim() === "" ? null : typedGb) !== view.own_gb;
  const pct = view.limit ? Math.min(100, Math.round((view.total / view.limit) * 100)) : 0;
  const canSet = usable(applies, "account.quota_set");

  return (
    <div className="usr-quota">
      <div className="usr-quota-head">
        <span className="usr-quota-what">
          {t("ui.admin.quota.disk")}
        </span>
        <span className="tnum">
          {sizeText(view.total)} / {view.limit ? gbText(view.limit / 2 ** 30) : t("ui.admin.quota.unlimited")}
        </span>
        {view.over && (
          <span className="usr-quota-over" {...tipAttrs(tipOf("error", t("ui.admin.quota.full_tip")))}>
            {t("ui.admin.quota.full")}
          </span>
        )}
      </div>
      <div className="usr-quota-bar" {...tipAttrs(tipOf("value", t("ui.admin.quota.used", { pct })))}>
        <i style={{ ["--pct" as string]: `${pct}%` }} />
      </div>
      <ul className="usr-quota-areas">
        {view.areas.map((a) => (
          <li key={a.id}>
            <span>{a.label}</span>
            <b className="tnum">{sizeText(a.bytes)}</b>
          </li>
        ))}
      </ul>
      {/* 网络流量：这个账号从服务器上取走了多少字节（lab2shot/server/traffic.py）。公网走 frp 是按流量计费的，
          所以这一格和硬盘占用并排放，看的是同一件事：这个账号占了多少资源 */}
      <div className="usr-quota-head">
        <span className="usr-quota-what">
          {t("ui.admin.quota.traffic")}
        </span>
      </div>
      <ul className="usr-quota-areas">
        {TRAFFIC.map(([id, label]) => (
          <li key={id}>
            <span>{label()}</span>
            <b className="tnum">{sizeText(view.traffic[id])}</b>
          </li>
        ))}
      </ul>
      {shown(applies, "account.quota_set") && (
        <div className="usr-quota-set">
          <label className="usr-quota-field">
            <span>{t("ui.admin.quota.quota")}</span>
            <input
              className="field mini tnum"
              inputMode="decimal"
              value={typed}
              disabled={!canSet || busy}
              placeholder={String(view.default_gb)}
              aria-label={t("ui.admin.quota.quota_label")}
              {...tipAttrs(canSet ? undefined : tipOf("disabled", why(applies, "account.quota_set")))}
              onChange={(e) => setTyped(e.target.value)}
            />
            <span className="dim">GB</span>
          </label>
          <Button
            tip={!canSet ? tipOf("disabled", why(applies, "account.quota_set")) : !ok ? tipOf("disabled", t("ui.admin.quota.bad")) : undefined}
            tone="primary"
            size="sm"
            disabled={!canSet || !ok || !changed || busy}
            onClick={() => void save(typed.trim() === "" ? null : typedGb)}
          >
            {t("ui.admin.common.save")}
          </Button>
        </div>
      )}
    </div>
  );
}
