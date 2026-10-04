import { Select } from "./Select";
import { t } from "../i18n/t";
import type { Department } from "../api/accounts";

/** 环节 (setting people.departments, lab2shot/accounts.py): the one picker of an account's production stage, on the
 * admin page's account forms and on the registration page alike. The list is long (every stage of a VFX pipeline),
 * so it is a drop-down, not a row of buttons; nothing is picked until someone picks. Each shows as the server says it
 * (accounts.department_label: a factory one in the page's language, one an administrator added as written); its value
 * is what is stored. */
export function StageSelect({ list, value, onPick, bad, large }: { list: Department[]; value: string; onPick: (stage: string) => void; bad?: boolean; large?: boolean }) {
  return (
    <Select
      label={t("ui.display.department")}
      className={[large && "lg", bad && "bad"].filter(Boolean).join(" ")}
      value={value}
      options={[...(value ? [] : [{ value: "", label: t("ui.display.pick_department"), off: true }]), ...list.map((d) => ({ value: d.value, label: d.label }))]}
      onPick={onPick}
      data-field="stage"
    />
  );
}
