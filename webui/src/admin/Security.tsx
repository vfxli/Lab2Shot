import { adminApi, type SecurityView } from "../api/admin";
import { PASSPHRASE_TIP } from "./Auth";
import { onlineTip, Section, useAdmin } from "./common";
import { whenSecondsText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

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
    <Section title="安全" lede="谁登录着，和看起来像在试探的请求。分享到外网时常来看看。账号在「用户」里管。">
      <div className="adm-tiles">
        <button className="adm-tile" data-tip={onlineTip(security?.online)} onClick={() => go("users")}>
          <span className="adm-tile-label">在线</span>
          <b>{security ? `浏览器 ${security.online.browser} · 插件 ${security.online.client}` : "…"}</b>
        </button>
        <div className="adm-tile" data-tip={PASSPHRASE_TIP}>
          <span className="adm-tile-label">忘了密码</span>
          <b>用口令</b>
        </div>
      </div>
      <Watch view={security} onUnblock={(c) => void unblock(c)} />
    </Section>
  );
}

function Watch({ view, onUnblock }: { view: SecurityView | null; onUnblock: (client: string) => void }) {
  if (!view) return <p className="adm-lede">读取中…</p>;
  const l = view.limits;
  return (
    <>
      <h3 className="adm-h3">可疑请求</h3>
      <p className="adm-lede">
        输错密码、访问没开放的接口、带 .. 之类的路径、请求太频繁，都记在这里（最近 1000 条，重启后清空；服务日志里也有）。
        同一个登录 {l.block_window_min} 分钟内有 {l.block_after} 次，就被封 {l.block_min} 分钟。输错密码：同一来源 {l.free} 次以后越等越久，{l.client_lock} 次锁住；
        所有来源 {l.window_min} 分钟内一共 {l.global_lock} 次，谁都登录不了 {l.window_min} 分钟（在服务器上执行 uv run lab2shot admin password 马上解开）。
      </p>
      {Object.keys(view.counts).length > 0 && (
        <div className="sec-row">
          {Object.entries(view.counts).map(([k, n]) => (
            <span key={k} className="chip" data-tip="这次服务启动以来的次数">
              {k} {n}
            </span>
          ))}
        </div>
      )}
      {view.blocked.length > 0 && (
        <table className="sec-table">
          <thead>
            <tr>
              <th>暂时封住</th>
              <th>账号</th>
              <th>原因</th>
              <th>到</th>
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
                  <Button tip="马上解开这个来源" tone="ghost" onClick={() => onUnblock(b.client)}>
                    解开
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
                <th>时间</th>
                <th>什么</th>
                <th>来源</th>
                <th>账号</th>
                <th>请求</th>
                <th>说明</th>
              </tr>
            </thead>
            <tbody>
              {view.events.map((e, i) => (
                <tr key={`${e.t}-${i}`}>
                  <td className="tnum">{whenSecondsText(e.t)}</td>
                  <td data-tip={e.counted ? "算进封禁：同一个登录 10 分钟里攒够 40 条就封 30 分钟"
                    : "不算进封禁：我们自己的页面打到这台服务器还没有的接口上，是版本对不上，不是探测"}>
                    {e.kind}{e.counted ? "" : " · 不计"}
                  </td>
                  <td className="mono" data-tip={e.client.startsWith("s:") ? "带着登录凭证的请求（按凭证区分）" : "没有凭证：按地址区分；经过内网穿透时大家的地址可能一样"}>
                    {e.who}
                  </td>
                  <td className="mono" data-tip={e.user ? "请求带的登录属于这个账号" : "没有登录"}>
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
        <p className="adm-empty">这次服务启动以来没有可疑请求。</p>
      )}
    </>
  );
}
