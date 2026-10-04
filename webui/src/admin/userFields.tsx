/** 账号表单的零件：一行（标签 + 控件 + 说明）、角色、勾选框（标签）。
 *
 * admin/Users.tsx 管的是「用户这一页长什么样」（列表、详情、每一行的动作），这里管的是「一张账号表单由哪些
 * 格子组成」：新建、修改用户的弹层（UserDialogs.tsx）都用它们，设置页的多选（Settings.tsx）和邀请码
 * （Invites.tsx）也借用其中的勾选框和一行，改一处各处都跟着变。 */

import type { UsersView } from "../api/admin";
import { Segmented } from "../ui/Button";
import "./users.css";
import { t } from "../i18n/t";
import { tipAttrs, type Tip } from "../platform/tips";
import { LabelRow } from "../ui/LabelRow";

/** One row of an account form: the site's 「标签 + 控件」 row (ui/LabelRow.tsx) in the form's LabelGrid (.usr-form), its
 * note under the control. */
export function Row({ label, tip, children, why }: { label: string; tip?: Tip | null; children: React.ReactNode; why?: string }) {
  return (
    <LabelRow label={label} labelClass="who-label" labelTip={tip} ctlClass="usr-ctl" below={why && <div className="usr-why lrow-under">{why}</div>}>
      {children}
    </LabelRow>
  );
}

export function Roles({ list, value, onPick }: { list: UsersView["roles"]; value: string; onPick: (r: string) => void }) {
  return (
    <Segmented label={t("ui.admin.users.role")} value={value} options={list.map((r) => ({ value: r.id, label: r.label }))} onChange={onPick} />
  );
}

/** 几个选项任选其一或多个，每项一个勾选框（账号的标签；多选类设置，admin/Settings.tsx）。 */
export function Checks({ options, value, onChange }: { options: { id: string; label: string; tip?: Tip | null }[]; value: string[]; onChange: (v: string[]) => void }) {
  return (
    <span className="usr-tags">
      {options.map((o) => (
        <label key={o.id} className="usr-tag" {...tipAttrs(o.tip)}>
          <input type="checkbox" checked={value.includes(o.id)} onChange={(e) => onChange(e.target.checked ? [...value, o.id] : value.filter((x) => x !== o.id))} />
          {o.label}
        </label>
      ))}
    </span>
  );
}

export function Tags({ view, value, onChange }: { view: UsersView; value: string[]; onChange: (v: string[]) => void }) {
  const options = Object.entries(view.tags).filter(([, t]) => !t.implied).map(([id, t]) => ({ id, label: t.label }));
  return <Checks options={options} value={value} onChange={onChange} />;
}
