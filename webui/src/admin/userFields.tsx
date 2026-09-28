/** 账号表单的零件：一行（标签 + 控件 + 说明）、角色、勾选框（标签）。
 *
 * admin/Users.tsx 管的是「用户这一页长什么样」（列表、详情、每一行的动作），这里管的是「一张账号表单由哪些
 * 格子组成」：新建、修改用户的弹层（UserDialogs.tsx）都用它们，设置页的多选（Settings.tsx）和邀请码
 * （Invites.tsx）也借用其中的勾选框和一行，改一处各处都跟着变。 */

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

export function Roles({ list, value, onPick }: { list: UsersView["roles"]; value: string; onPick: (r: string) => void }) {
  return (
    <Segmented label="角色" value={value} options={list.map((r) => ({ value: r.id, label: r.label, tip: r.tip }))} onChange={onPick} />
  );
}

/** Any of a few options, each a box to tick (an account's tags; a setting of kind multi, admin/Settings.tsx). */
export function Checks({ options, value, onChange }: { options: { id: string; label: string; tip: string }[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <span className="usr-tags">
      {options.map((o) => (
        <label key={o.id} className="usr-tag" data-tip={o.tip}>
          <input type="checkbox" checked={value.includes(o.id)} onChange={(e) => onChange(e.target.checked ? [...value, o.id] : value.filter((x) => x !== o.id))} />
          {o.label}
        </label>
      ))}
    </span>
  );
}

export function Tags({ view, value, onChange }: { view: UsersView; value: string[]; onChange: (v: string[]) => void }) {
  const options = Object.entries(view.tags).filter(([, t]) => !t.implied).map(([id, t]) => ({ id, label: t.label, tip: t.tip }));
  return <Checks options={options} value={value} onChange={onChange} />;
}

export const TAGS_TIP = "能用哪些节点：只看得到带这些标签的节点和模板，别的在他那里就像不存在。基础节点（Lab2Shot 自己的和读写文件格式的）每个人都能用";
