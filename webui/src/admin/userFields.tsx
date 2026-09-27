/** 账号表单的零件：一行（标签 + 控件 + 说明）、部门、角色、标签勾选。
 *
 * admin/Users.tsx 管的是「用户这一页长什么样」（列表、详情、每一行的动作），这里管的是「一张账号表单由哪些
 * 格子组成」：新建、修改、配额、权限几个弹层都用它们，改一处四处都跟着变。 */

import type { UsersView } from "../api/admin";
import { Segmented } from "../ui/Button";
import "./users.css";

export function Row({ label, tip, children, why }: { label: string; tip: string; children: React.ReactNode; why?: string }) {
  return (
    <>
      <div className="who-row">
        <span className="who-label" data-tip={tip}>
          {label}
        </span>
        <span className="usr-ctl">{children}</span>
      </div>
      {why && <div className="usr-why">{why}</div>}
    </>
  );
}

export function Departments({ list, value, onPick }: { list: string[]; value: string; onPick: (d: string) => void }) {
  return (
    <Segmented label="部门" value={value} options={list.map((d) => ({ value: d, label: d, tip: `部门 ${d}` }))} onChange={onPick} />
  );
}

export function Roles({ list, value, onPick }: { list: UsersView["roles"]; value: string; onPick: (r: string) => void }) {
  return (
    <Segmented label="角色" value={value} options={list.map((r) => ({ value: r.id, label: r.label, tip: r.tip }))} onChange={onPick} />
  );
}

export function Tags({ view, value, onChange }: { view: UsersView; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <span className="usr-tags">
      {Object.entries(view.tags)
        .filter(([, t]) => !t.implied)
        .map(([id, t]) => (
          <label key={id} className="usr-tag" data-tip={t.tip}>
            <input type="checkbox" checked={value.includes(id)} onChange={(e) => onChange(e.target.checked ? [...value, id] : value.filter((x) => x !== id))} />
            {t.label}
          </label>
        ))}
    </span>
  );
}

export const TAGS_TIP = "能用哪些节点：只看得到带这些标签的节点和模板，别的在他那里就像不存在。基础节点（Lab2Shot 自己的和读写文件格式的）每个人都能用";
