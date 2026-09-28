import type { UsageStats, UsageTemplate } from "../api";
import { fullTimeText } from "../platform/format";
import { lastText, times } from "./Usage";

/** 使用统计 按模板 (lab2shot/farm/usage.py templates): which template card each job of the range was opened from,
 * counted by when the job was submitted. A template is shown by its name now; a deleted one by the name it had at its
 * last use, marked 已删除. The graphs built by hand are one row of their own, last. */
export function TemplateView({ data }: { data: UsageStats }) {
  const rows = data.templates;
  const cards = rows.filter((t) => t.id);
  const all = rows.reduce((s, t) => s + t.count, 0);
  const fromCards = cards.reduce((s, t) => s + t.count, 0);
  const share = all ? Math.round((100 * fromCards) / all) : 0;

  return (
    <>
      <div className="u-tiles">
        <div className="u-tile" data-tip="这段时间提交过任务的模板，删掉的也算">
          <span>用过的模板</span>
          <b>{cards.length} 张</b>
        </div>
        <div className="u-tile" data-tip="这段时间提交的任务，不管结果如何">
          <span>任务</span>
          <b>{times(all)}</b>
        </div>
        <div className="u-tile" data-tip="这些任务里，节点图是从模板打开的占多少">
          <span>从模板打开的</span>
          <b>{all ? `${share}%` : "—"}</b>
        </div>
      </div>
      {rows.length ? (
        <div className="q-table-wrap">
          <table className="q-table u-table u-templates">
            <thead>
              <tr>
                <th data-tip="节点图是从哪张模板打开的：按模板现在的名字；删掉的按最后一次用时的名字">模板</th>
                <th data-tip="提交了几次任务，不管结果如何">次数</th>
                <th data-tip="算完、没有出错的次数">成功</th>
                <th data-tip="出错结束的次数；部分完成和取消的只算在次数里">失败</th>
                <th data-tip="这段时间用过它的账号数">人数</th>
                <th data-tip="这段时间里最近一次提交">最近一次</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <TemplateRow key={t.id || "hand"} t={t} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="help-muted">这段时间没有提交过任务</p>
      )}
    </>
  );
}

function TemplateRow({ t }: { t: UsageTemplate }) {
  const zero = (n: number) => (n ? "tnum" : "tnum u-zero");
  return (
    <tr className={t.id ? undefined : "u-core"}>
      <td>
        {t.id ? (
          <span className={t.deleted ? "u-zero" : undefined} data-user-data data-tip={`${t.name}\n${t.id}`}>
            {t.name}
          </span>
        ) : (
          <span className="u-zero" data-tip="节点图不是从模板卡片打开的：自己一个个节点搭起来的">
            {t.name}
          </span>
        )}
        {t.deleted && (
          <span className="chip u-chip" data-tip="这张模板已经删掉了，按它最后一次被用时的名字列出">
            已删除
          </span>
        )}
      </td>
      <td className="tnum">{times(t.count)}</td>
      <td className={zero(t.done)}>{times(t.done)}</td>
      <td className={zero(t.failed)}>{times(t.failed)}</td>
      <td className="tnum">{t.users} 人</td>
      <td className="tnum" data-tip={fullTimeText(t.last)}>
        {lastText(t.last)}
      </td>
    </tr>
  );
}
