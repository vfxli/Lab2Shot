/** An account's login picture (GET /api/admin/users/{id}/logins): where it is online right now and the 7-/30-day
 * summary with the 异地频繁 warning. The attempts themselves are a registered resource (lab2shot/resources.py
 * login_log), listed by the one shared table on the account's page — this part only says what a table of rows cannot:
 * whether the pattern of logins looks like a leaked account. */

import { useEffect, useState } from "react";
import { adminApi, type UserLogins } from "../api/admin";

const KIND_LABEL: Record<string, string> = { web: "浏览器", client: "插件" };

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
  const where = data.online.map((o) => `${KIND_LABEL[o.kind] ?? o.kind} · ${o.device} · ${o.ip}`);
  // a tile says one short thing and keeps the detail on hover: UI text never wraps and is never cut
  const span = (k: "7d" | "30d") => {
    const s = data.summary[k];
    return { short: `${s.logins} 次登录`, full: `${s.logins} 次登录 · ${s.ips} 个 IP · ${s.devices} 个设备 · ${s.failures} 次失败` };
  };
  return (
    <div className="usr-logins">
      <div className="adm-tiles">
        <div className="adm-tile" data-tip={`一个浏览器和一台电脑上的 DCC 插件 / 命令行可以同时在线，各算一处；最近一两分钟内向服务器发过请求才算在线\n${where.join("\n") || "现在不在线"}`}>
          <span className="adm-tile-label">现在在线</span>
          <b>{data.online.length ? data.online.map((o) => KIND_LABEL[o.kind] ?? o.kind).join("、") : "不在线"}</b>
        </div>
        {(["7d", "30d"] as const).map((k) => {
          const s = span(k);
          return (
            <div key={k} className="adm-tile" data-tip={`${s.full}\n${k === "7d" ? data.threshold_tip : "最近 30 天"}`}>
              <span className="adm-tile-label">{k === "7d" ? "近 7 天" : "近 30 天"}</span>
              <b className="tnum">{s.short}</b>
            </div>
          );
        })}
      </div>
      {data.summary.suspicious && (
        <p className="usr-suspicious" role="alert" data-tip={data.threshold_tip}>
          异地频繁：这个账号最近登录的来源比较分散，或者失败次数偏多，留意是不是账号泄露了
        </p>
      )}
    </div>
  );
}
