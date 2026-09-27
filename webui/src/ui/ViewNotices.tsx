import "./viewNotices.css";

/** 视图的通知区：集中显示与当前画面相关的信息。视图的通知只在此处显示，不放在节点旁，
 * 以免被忽略。
 *
 * 规则：
 * - 位置固定：始终位于舞台左上角、工具栏下方，条数变化时不跳动，便于用户记住查看位置；
 * - 顺序固定（`ORDER`）：同一条通知始终位于同一格，不因其他通知出现而换位；
 * - 每条为一句短语，原因写在悬停提示中（界面文字不换行、不截断）；
 * - 节点的问题不在此处显示，属于节点的通知。此处只说明当前画面与数据本身的差异或尚缺的内容。
 *
 * 只有一种通知会自动消失（`fades`：视图刚自动执行的操作），其余均持续显示；
 * 「当前显示的不是数据本身」这类说明不得自动消失。
 *
 * 常见种类：
 *   proxy    点云以代理方式显示，并非全部点（「显示了 30 万 / 共 100 万点」）
 *   partial  边算边看：已计算到第几帧；以及「该帧尚未到达，当前显示的是第几帧」
 *   did      视图刚自动执行的操作，4 秒后消失（三维视图切换视角、曲线无法绘制时使用） */
import type { NoticeKind } from "../model/viewNotices";
export type { NoticeKind };

export interface Notice {
  kind: NoticeKind;
  key: string;   // 同一条通知内容变化时仍视为同一条（React 的 key）
  text: string;  // 一句短语
  tip: string;   // 原因：悬停查看
}

// 固定顺序：先说明「当前显示的不是数据本身」，再说明「数据尚未齐全」，最后是「视图刚自动执行的操作」
const ORDER: NoticeKind[] = ["local", "proxy", "camera", "error", "partial", "hint", "did"];

export function ViewNotices({ notices }: { notices: Notice[] }) {
  if (!notices.length) return null;
  const sorted = [...notices].sort((a, b) => ORDER.indexOf(a.kind) - ORDER.indexOf(b.kind));
  return (
    <div className="view-notices" role="status">
      {sorted.map((n) => (
        <div key={n.key} className={`view-notice notice-${n.kind}`} data-tip={n.tip}>
          {n.text}
        </div>
      ))}
    </div>
  );
}
