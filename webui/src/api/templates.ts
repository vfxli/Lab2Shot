// 模板和节点菜单的管理（lab2shot/server/templates.py、server/categories.py），做在前台的模板弹窗和节点菜单里
// （分类不写死在代码里，管理员在面板上分）。模板全是文件（lab2shot/site/library.py：templates/ 项目预设、
// adapters/<包>/templates/ 接入层自带、work/users/<用户名>/templates/ 用户自己的）；两棵分类树和节点归属也是文件
// （lab2shot/categories.py）。由 api/admin.ts 转发导出。

import { json } from "../platform/http";
import type { MenuInfo, TreeCategory } from "./catalog";

export type AdminCategory = TreeCategory;

/** A display text per language as written ({zh, en}; an unwritten one missing or empty), or one string meaning the
 * language of the request (the server leaves the other languages as they are). */
export type LangWords = Partial<Record<"zh" | "en", string>> | string;
/** Every category's name in every language, as written (the admin answers' `words`: the editor's field). */
export type CategoryWords = Record<string, { label: Partial<Record<"zh" | "en", string>> }>;

/** 一个分类的新建或增改：`parent` 空是一级分类；节点菜单的一级分类还要说明在哪个区（`section`）。 */
export interface CategoryEdit {
  id: string;
  parent?: string; // 空：一级分类
  label: LangWords; // {lang: text} for each language written, or a string: the page's language (the server keeps the others)
  color?: string;
  rank?: number;
  section?: string; // 节点菜单：tools / deliver；模板的树不用
}

const enc = encodeURIComponent;

export const templatesApi = {
  switchTemplate: (id: string, enabled: boolean) => json<{ id: string; enabled: boolean }>("PUT", `/api/admin/templates/${enc(id)}`, { enabled }),
  // 把一张卡归到一个二级 / 一级分类（拖过去）；"" 进「未分类」。写进它自己文件的 meta.deliverable
  placeTemplate: (id: string, where: string) => json<{ id: string; where: string }>("PUT", `/api/admin/templates/${enc(id)}/place`, { where }),
  // 改名字和简介（卡片菜单「编辑」）：写进它自己的文件
  editTemplate: (id: string, name: string, intro: string) => json<{ id: string; name: string; intro: string }>("PUT", `/api/admin/templates/${enc(id)}/text`, { name, intro }),
  // 复制成一张新的预设卡（名字后加「副本」，归同一分类）
  copyTemplate: (id: string, name = "") => json<{ id: string; name: string }>("POST", `/api/admin/templates/${enc(id)}/copy`, { name }),
  // 删掉一个项目预设（templates/ 下的那个文件）；接入层自带的删不了，只能关闭
  deleteTemplate: (id: string) => json<{ id: string }>("DELETE", `/api/admin/templates/${enc(id)}`),
  // 「保存为预设模板」（文件菜单）：写成 templates/ 下的一个文件。和已有项目预设同名时服务器回 409 E-TEMPLATES-SAMENAME，
  // 问过管理员之后带 `replace` 再发：覆盖那个文件（文件名、创建时间、作者不变）
  createTemplate: (entry: { name: string; intro?: string; deliverable?: string; graph: unknown; replace?: boolean }) =>
    json<{ id: string; name: string; replaced: boolean }>("POST", "/api/admin/templates", { intro: "", deliverable: "", replace: false, ...entry }),
  // 模板面板的树（templates/_categories.json）
  categories: () => json<{ categories: TreeCategory[]; problem: string; words: CategoryWords }>("GET", "/api/admin/categories"),
  saveCategory: (c: CategoryEdit) => json<{ categories: TreeCategory[] }>("PUT", "/api/admin/categories", { parent: "", color: "", rank: 0, ...c }),
  // 删除本身总会完成；`problem` 列出文件没能改写的卡片（它们显示为「未分类」）
  removeCategory: (id: string) => json<{ categories: TreeCategory[]; problem: string }>("DELETE", `/api/admin/categories/${enc(id)}`),
  // 一次拖动 = 一次写入、一条留底：一级分类（parent 空）或一个分类下的二级分类的新次序
  orderCategories: (ids: string[], parent = "") => json<{ categories: TreeCategory[] }>("PUT", "/api/admin/categories/order", { ids, parent }),
  // 节点菜单的树（menu/categories.json）和节点归属（menu/nodes.json）：每条都交回整个菜单
  // the node menu's categories in every language (the rename sheet's two fields)
  menuCategories: () => json<MenuInfo & { words: CategoryWords }>("GET", "/api/admin/menu/categories"),
  saveMenuCategory: (c: CategoryEdit) => json<MenuInfo>("PUT", "/api/admin/menu/categories", { parent: "", color: "", rank: 0, section: "", ...c }),
  removeMenuCategory: (id: string) => json<MenuInfo>("DELETE", `/api/admin/menu/categories/${enc(id)}`),
  orderMenuCategories: (ids: string[], parent = "") => json<MenuInfo>("PUT", "/api/admin/menu/categories/order", { ids, parent }),
  placeNode: (typeId: string, where: string) => json<MenuInfo>("PUT", `/api/admin/menu/nodes/${enc(typeId)}/place`, { where }),
  // 一个节点的副标题和说明，每种语言实际写了的（节点菜单里节点的「编辑」）；改：写进各语言的目录文件
  nodeText: (typeId: string) => json<NodeWords>("GET", `/api/admin/menu/nodes/${enc(typeId)}/text`),
  editNodeText: (typeId: string, subtitle: LangWords, description: LangWords) => json<NodeWords>("PUT", `/api/admin/menu/nodes/${enc(typeId)}/text`, { subtitle, description }),
};

/** A node type's subtitle and description in every language, as written (GET/PUT /api/admin/menu/nodes/{type}/text). */
export interface NodeWords {
  id: string;
  subtitle: Partial<Record<"zh" | "en", string>>;
  description: Partial<Record<"zh" | "en", string>>;
}
