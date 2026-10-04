import type { Traffic } from "../api/library";
import type { UserRow } from "../api/admin";
import { sizeText } from "../platform/format";
import { t as tr } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 流量的三个数值，集中定义：
 * 「用户」表中每列一个（admin/Users.tsx），用户页上并列三个（admin/UserQuota.tsx），两处表示同一含义，
 * 名称只在此处定义。
 *
 * 统计的是实际发出的字节：取帧、三维数据、「输出」的下载、接口回答均计入（lab2shot/server/traffic.py）。 */
export const TRAFFIC = [
  ["today", () => tr("ui.admin.traffic.today"), () => tr("ui.admin.traffic.today_column")],
  ["week", () => tr("ui.admin.traffic.week"), () => tr("ui.admin.traffic.week_column")],
  ["total", () => tr("ui.admin.traffic.total"), () => tr("ui.admin.traffic.total_column")],
] as const;

/** 在一格中完整呈现三个数值（表格每列只显示一个数字，悬停提示给出另外两个）。 */
const trafficTip = (t: Traffic): string =>
  tr("ui.admin.traffic.tip", { today: sizeText(t.today), week: sizeText(t.week), total: sizeText(t.total) });

/** 「用户」表中的流量三列（在 admin/Users.tsx 的 columns 中展开）：每列一个数字，悬停提示中包含全部三个数值。 */
export const trafficColumns = () =>
  TRAFFIC.map(([k, , label]) => ({
    id: `traffic_${k}`,
    label: label(),
    className: "tnum",
    cell: (u: UserRow) => <span {...tipAttrs(tipOf("value", trafficTip(u.traffic)))}>{sizeText(u.traffic[k])}</span>,
  }));
