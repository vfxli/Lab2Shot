// 我的模板与个人磁盘占用 (lab2shot/server/library.py, quota.py)。
// 编辑器与账号菜单读取此处；后台读取相同内容经由 api/admin.ts，两者使用同一套存储与同一份回答。

import { json } from "../platform/http";
import type { GraphJSON } from "./catalog";

/** 保存在服务器上的一张节点图（列表中不包含节点图本身，打开时才获取）。
 * 名称与简介为使用者填写的文字：绘制时一律按纯文本处理，并带有 data-user-data。 */
export interface SavedGraph {
  id: string; // user~<用户名>~<文件名>（lab2shot/library.py card_id）
  stem: string; // the file's name
  name: string;
  intro: string;
  bytes: number;
  updated: number;
  deleted: number | null; // 位于回收站中（work/users/<用户名>/templates/_bin/，管理员可恢复）
  deleted_by: string; // user / admin
}

/** 磁盘占用中的一项：占用量、内容说明、使用者自行清理可释放的空间（0 表示该项不允许自行清理）。 */
export interface StorageArea {
  id: string;
  label: string;
  bytes: number;
  note: string;
}

/** 账号使用的网络流量，单位为字节（lab2shot/server/traffic.py）：今天、近 7 天（含今天）、总计。
 * 仅后台管理员接口提供此数据：服务器只在 /api/admin/users… 相关接口中返回（lab2shot/server/quota.py my_storage），
 * 前台无法获取，因此前台没有任何组件读取该数据。 */
export interface Traffic {
  today: number;
  week: number;
  total: number;
}

/** 判断是否仍可写入的三个数（lab2shot/server/quota.py gate）：队列轮询中只包含这些。
 * 四项明细与流量不在其中：它们在计算过程中持续变化，若纳入每 1.5–30 秒一次的轮询，本应返回 304 的回答每次都会完整重发。
 * 明细位于 StorageUsage，由「我的占用」自行查询一次。 */
export interface StorageGate {
  total: number;
  limit: number; // 字节（0：不限）
  over: boolean;
}

/** 账号占用的资源（lab2shot/server/quota.py usage）：硬盘四项与上限，不含流量：
 * 前台（队列窗口中的「我的占用」、删除任务后的更新）获得的即为此数据。 */
export interface StorageUsage extends StorageGate {
  user: number;
  own_gb: number | null; // 该账号自身的上限（null：使用设置中的默认值）
  default_gb: number;
  left: number;
  areas: StorageArea[];
}

/** 后台按账号查看的数据：硬盘与流量（GET / PUT /api/admin/users/{id}/quota，权限 users.manage_normal）。
 * 只有该接口包含流量；`GET /api/my/storage` 不返回流量，因此前台使用上方的 `StorageUsage`。 */
export interface AccountUsage extends StorageUsage {
  traffic: Traffic;
}

export interface MyTemplates {
  mine: SavedGraph[];
  bin: SavedGraph[];
  usage: StorageUsage;
}

export const libraryApi = {
  myTemplates: () => json<MyTemplates>("GET", "/api/my/templates"),
  saveTemplate: (t: { name: string; intro?: string; graph: GraphJSON; id?: string }) =>
    json<MyTemplates>("POST", "/api/my/templates", { intro: "", id: "", ...t }),
  openTemplate: (id: string) => json<SavedGraph & { graph: GraphJSON }>("GET", `/api/my/templates/${encodeURIComponent(id)}`),
  binTemplate: (id: string) => json<MyTemplates>("DELETE", `/api/my/templates/${encodeURIComponent(id)}`),
  restoreTemplate: (id: string) => json<MyTemplates>("POST", `/api/my/templates/${encodeURIComponent(id)}/restore`, {}),
  storage: () => json<StorageUsage>("GET", "/api/my/storage"),
};
