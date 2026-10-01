import { useEffect, useState } from "react";
import { api } from "../api";
import type { StorageUsage } from "../api/library";
import { Empty } from "./Empty";
import { Loading } from "./Loading";
import { reasonOf } from "../messages/message";
import { gbText, sizeText } from "../platform/format";
import "./storage.css";

/** 我的占用：队列窗口中的一段，只显示总量：已用、上限、剩余。
 *
 * 不分项列出明细，也不提供「清理」按钮：腾出空间的操作均在任务上。占用 = 自己所有任务的文件夹 + 还没有任务用到的上传
 * + 存在服务器上的模板（缓存不算）；在队列中删除一条任务时，它的文件夹整个删除（server/quota.py drop_for_job）；需要一次性清理时使用
 * 队列上方的「删除全部」。
 *
 * 此处不显示流量：流量只在后台「用户」页显示（admin/traffic.tsx、admin/UserQuota.tsx）。这并非在此处按角色拦截，
 * 而是服务器的 `/api/my/storage` 根本不返回 `traffic` 键（lab2shot/server/quota.py my_storage），网页组件中不写角色判断。
 *
 * 本组件自行请求 `/api/my/storage`（窗口打开时，以及删除任务后 `again` 变化时），不随队列每 1.5 秒的轮询请求：
 * 占用在每次计算时都会变化，放入轮询会使本应返回 304 的响应每次都完整重发（api/library.ts StorageGate）。 */
export function StoragePanel({ again }: { again: number }) {
  const [usage, setUsage] = useState<StorageUsage | null>(null);
  const [problem, setProblem] = useState("");

  useEffect(() => {
    let live = true;
    api.storage().then(
      (v) => live && (setUsage(v), setProblem("")),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [again]);

  if (problem && !usage) return <Empty title={problem} hint="刷新页面再试一次。" />;
  if (!usage) return <Loading what="自己的磁盘占用" />;

  const pct = usage.limit ? Math.min(100, Math.round((usage.total / usage.limit) * 100)) : 0;
  return (
    <div className="sto">
      <div className="sto-head">
        <span className="tnum sto-total">
          {sizeText(usage.total)} / {usage.limit ? gbText(usage.limit / 2 ** 30) : "不限"}
        </span>
        <span className={`sto-bar${usage.over ? " over" : ""}`} data-tip={`占了 ${pct}%`}>
          <i style={{ ["--pct" as string]: `${pct}%` }} />
        </span>
        {usage.over ? (
          <span className="chip sto-over" data-tip="已经到上限：在队列里删掉用不到的任务（每一行的「删除」，或者上面的「删除全部」），删任务就腾出它占的空间，删完接着用；还不够找管理员调大配额。满着的时候点「计算」「打包」会被拦下">
            已满
          </span>
        ) : (
          <span className="chip sto-left tnum" data-tip="离上限还有这么多；满了就不能再算、再传、再存模板">
            还剩 {usage.limit ? sizeText(usage.left) : "不限"}
          </span>
        )}
      </div>
      {problem && (
        <p className="login-problem" role="alert">
          {problem}
        </p>
      )}
    </div>
  );
}
