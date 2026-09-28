import { Select } from "./Select";

/** 环节 (setting people.departments, lab2shot/accounts.py): the one picker of an account's production stage, on the
 * admin page's account forms and on the registration page alike. The list is long (every stage of a VFX pipeline),
 * so it is a drop-down, not a row of buttons; nothing is picked until someone picks. */
export function StageSelect({ list, value, onPick, bad, large }: { list: string[]; value: string; onPick: (stage: string) => void; bad?: boolean; large?: boolean }) {
  return (
    <Select
      label="环节"
      tip="你在制作里属于哪个环节"
      className={[large && "lg", bad && "bad"].filter(Boolean).join(" ")}
      value={value}
      options={[...(value ? [] : [{ value: "", label: "选一个环节", tip: "还没选", off: true }]), ...list.map((d) => ({ value: d, label: d, tip: `环节：${d}` }))]}
      onPick={onPick}
      data-field="stage"
    />
  );
}
