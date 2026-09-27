import type { Traffic } from "../api/library";
import type { UserRow } from "../api/admin";
import { sizeText } from "../platform/format";

/** 流量的三个数值，集中定义：
 * 「用户」表中每列一个（admin/Users.tsx），用户页上并列三个（admin/UserQuota.tsx），
 * 编辑器队列窗口中自己的那一行（ui/Storage.tsx），三处表示同一含义，名称与说明只在此处定义。
 *
 * 统计的是实际发出的字节：取帧、三维数据、交付物下载、接口回答均计入（lab2shot/server/traffic.py）。 */
export const TRAFFIC = [
  ["today", "今天", "今日流量", "今天到现在为止，这个账号从服务器上取走的数据"],
  ["week", "近 7 天", "7 天流量", "最近 7 天（含今天）取走的数据"],
  ["total", "总计", "总流量", "从建号到现在取走的数据合计；服务被强行停掉时，最后不到一分钟的量会丢"],
] as const;

/** 在一格中完整呈现三个数值（表格每列只显示一个数字，悬停提示给出另外两个）。 */
export const trafficTip = (t: Traffic): string =>
  `取帧、三维数据、交付物下载、接口的回答都算\n今天 ${sizeText(t.today)} · 近 7 天 ${sizeText(t.week)} · 总计 ${sizeText(t.total)}`;

/** 「用户」表中的流量三列（在 admin/Users.tsx 的 columns 中展开）：每列一个数字，悬停提示中包含全部三个数值。 */
export const trafficColumns = () =>
  TRAFFIC.map(([k, , label, tip]) => ({
    id: `traffic_${k}`,
    label,
    tip,
    className: "tnum",
    cell: (u: UserRow) => <span data-tip={trafficTip(u.traffic)}>{sizeText(u.traffic[k])}</span>,
  }));
