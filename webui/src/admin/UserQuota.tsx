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
    if (!may) return; // not this login's to see: not asked (the refusal would count against the session)
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
  if (!view) return <Loading what="这个账号占了多少" />;

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
        <span className="usr-quota-what" data-tip="这个账号所有任务的文件夹（上传的素材、输出）、还没有任务用到的上传和存在服务器上的模板加起来；缓存不算">
          磁盘占用
        </span>
        <span className="tnum" data-tip={`占了 ${sizeText(view.total)}，上限 ${view.limit ? gbText(view.limit / 2 ** 30) : "不限"}`}>
          {sizeText(view.total)} / {view.limit ? gbText(view.limit / 2 ** 30) : "不限"}
        </span>
        {view.over && (
          <span className="usr-quota-over" data-tip="已经到上限：这个账号不能再上传，要先清理或者调大配额">
            已满
          </span>
        )}
      </div>
      <div className="usr-quota-bar" data-tip={`占了 ${pct}%`}>
        <i style={{ ["--pct" as string]: `${pct}%` }} />
      </div>
      <ul className="usr-quota-areas">
        {view.areas.map((a) => (
          <li key={a.id} data-tip={a.note}>
            <span>{a.label}</span>
            <b className="tnum">{sizeText(a.bytes)}</b>
          </li>
        ))}
      </ul>
      {/* 网络流量：这个账号从服务器上取走了多少字节（lab2shot/server/traffic.py）。公网走 frp 是按流量计费的，
          所以这一格和硬盘占用并排放，看的是同一件事：这个账号占了多少资源 */}
      <div className="usr-quota-head">
        <span className="usr-quota-what" data-tip="这个账号从服务器上取走的数据：取帧、三维数据、交付物下载、接口的回答都算。服务被强行停掉时，最后不到一分钟的量会丢">
          网络流量
        </span>
      </div>
      <ul className="usr-quota-areas">
        {TRAFFIC.map(([id, label, , tip]) => (
          <li key={id} data-tip={tip}>
            <span>{label}</span>
            <b className="tnum">{sizeText(view.traffic[id])}</b>
          </li>
        ))}
      </ul>
      {shown(applies, "account.quota_set") && (
        <div className="usr-quota-set">
          <label className="usr-quota-field">
            <span data-tip={`这个账号的上限，GB；留空跟着设置里的默认（${view.default_gb} GB）`}>配额</span>
            <input
              className="field mini tnum"
              inputMode="decimal"
              value={typed}
              disabled={!canSet || busy}
              placeholder={String(view.default_gb)}
              aria-label="磁盘配额，GB"
              data-tip={canSet ? `留空：跟着设置里的默认（${view.default_gb} GB）；1 到 10000 GB` : why(applies, "account.quota_set")}
              onChange={(e) => setTyped(e.target.value)}
            />
            <span className="dim">GB</span>
          </label>
          <Button
            tip={!canSet ? why(applies, "account.quota_set") : !ok ? "填 1 到 10000 之间的数字，或者留空跟着默认" : "保存这个账号的配额：马上生效"}
            tone="primary"
            size="sm"
            disabled={!canSet || !ok || !changed || busy}
            onClick={() => void save(typed.trim() === "" ? null : typedGb)}
          >
            保存
          </Button>
        </div>
      )}
    </div>
  );
}
