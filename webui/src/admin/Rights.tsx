import { useEffect, useState } from "react";
import { adminApi, type RightsSheetView, type RightsView } from "../api/admin";
import { shown, usable, why } from "../api/applies";
import type { Availability } from "../api/applies";
import { Button, Switch } from "../ui/Button";
import { Sheet } from "../ui/Sheet";
import { Loading } from "../ui/Loading";
import { msg, reasonOf } from "../messages/message";
import { MessageText } from "../ui/MessageText";
import { whenText } from "../platform/format";
import "./rights.css";

/** 二级管理员权限：一级管理员为二级管理员开放的权限，每一项为可查看、可修改或不可见。
 *
 * 每条权限本身即对应「可查看 / 可修改 / 不可见」三档之一：部分权限仅允许查看（如查看统计），部分允许修改（如素材与真值），
 * 未勾选任何权限的后台区块不显示（可见性由服务器按权限计算，lab2shot/availability.py 中 CAPABILITY 档 =
 * 隐藏）。因此此处不另设三态开关，只提供一列勾选框，每项附带说明，表明勾选后授予的是查看还是修改。
 *
 * 排版完成的表格由服务器提供（lab2shot/roles.py sheet()）：分段、短名、说明、勾选状态和数量。本模块只负责绘制，
 * 不分组、不计数、不识别任何权限名称，也不检查角色；是否显示仅由 rights.edit 决定。 */
export function Rights({ applies }: { applies: Availability | null | undefined }) {
  const [view, setView] = useState<RightsView | null>(null);
  const may = shown(applies, "rights.edit");

  useEffect(() => {
    if (!may) return;
    let live = true;
    adminApi.rights().then(
      (v) => live && setView(v),
      () => live && setView(null),
    );
    return () => {
      live = false;
    };
  }, [may]);

  if (!may) return null;
  return (
    <>
      {(view?.sheets ?? [null]).map((s, i) => (
        <RightsBand key={s?.role ?? i} sheet={s} applies={applies} onChanged={setView} />
      ))}
    </>
  );
}

/** 单个角色的区块：一句现状说明和打开勾选表的按钮。 */
function RightsBand({ sheet, applies, onChanged }: {
  sheet: RightsSheetView | null;
  applies: Availability | null | undefined;
  onChanged: (v: RightsView) => void;
}) {
  const [open, setOpen] = useState(false);
  const name = sheet?.label ?? "二级管理员";
  const summary = !sheet
    ? ""
    : `现在有 ${sheet.count} / ${sheet.total} 条` +
      (sheet.default ? "：还是默认那一份" : `：${sheet.updated_by || "管理员"} ${whenText(sheet.updated)}改的`);

  return (
    <section className="rts">
      <div className="rts-head">
        <h3>{name}权限</h3>
        <span className="dim rts-sum" data-tip={sheet?.tip ?? "勾上哪些就能做哪些；一条都没勾的那一块，他的后台里不出现"}>
          {sheet ? summary : <Loading what="权限" />}
        </span>
        <Button
          tip={usable(applies, "rights.edit") ? `分配${name}能做哪些事：勾上就有，去掉就没了，改完下一个请求就生效` : why(applies, "rights.edit")}
          disabled={!usable(applies, "rights.edit") || !sheet}
          onClick={() => setOpen(true)}
        >
          分配权限
        </Button>
      </div>
      {open && sheet && <RightsSheet sheet={sheet} onClose={() => setOpen(false)} onSaved={onChanged} />}
    </section>
  );
}

/** 勾选表：按服务器给出的分段逐行列出（开关 + 短名 + 说明），底部为恢复默认、取消和保存。 */
function RightsSheet({ sheet, onClose, onSaved }: {
  sheet: RightsSheetView;
  onClose: () => void;
  onSaved: (v: RightsView) => void;
}) {
  const start = sheet.groups.flatMap((g) => g.items.filter((x) => x.on).map((x) => x.id));
  const [picked, setPicked] = useState<string[]>(start);
  const [busy, setBusy] = useState("");
  const [problem, setProblem] = useState("");
  const chosen = new Set(picked);
  const same = picked.length === start.length && start.every((c) => chosen.has(c));
  // 授予该权限等同于使该角色成为管理员（其说明中写明可分配权限），因此勾选时须给出明确提示。
  // 提示文字位于消息目录（N-RIGHTS-ADMINS），此处不重复定义，也不依赖其代号。
  const risky = sheet.groups.flatMap((g) => g.items).find((x) => x.what.includes("分配二级管理员的权限"));

  const toggle = (id: string, on: boolean) => {
    setPicked((was) => (on ? [...was, id] : was.filter((x) => x !== id)));
    setProblem("");
  };

  const run = async (what: "save" | "default") => {
    setBusy(what);
    try {
      const got = what === "save" ? await adminApi.setRights(sheet.role, picked) : await adminApi.resetRights(sheet.role);
      onSaved(got);
      onClose();
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setBusy("");
    }
  };

  return (
    <Sheet title={`${sheet.label}权限`} width={560} onClose={onClose}>
      <div className="rts-form">
        <p className="rts-note">
          勾上哪些，{sheet.label}就能做哪些。每条自己写着是「看」还是「改」；一条都没勾的那一块，他的后台里根本不出现。改完下一个请求就生效，他不用重新登录。
        </p>
        <div className="rts-list">
          {sheet.groups.map((g) => (
            <div className="rts-group" key={g.label}>
              <h4>{g.label}</h4>
              {g.items.map((x) => (
                <label className="rts-row" key={x.id} data-tip={x.what}>
                  <Switch on={chosen.has(x.id)} mini label={x.label} tip={x.what} onChange={(on) => toggle(x.id, on)} />
                  <span className="rts-name">{x.label}</span>
                  <span className="rts-what dim">{x.what}</span>
                </label>
              ))}
            </div>
          ))}
        </div>
        {risky && chosen.has(risky.id) && (
          <p className="rts-warn" role="status">
            <MessageText message={msg("N-RIGHTS-ADMINS")} />
          </p>
        )}
        {problem && (
          <p className="login-problem" role="alert">
            {problem}
          </p>
        )}
        <div className="dialog-row">
          <Button tip={`恢复成代码里那一份默认的 ${sheet.default_count} 条`} tone="ghost" disabled={!!busy} onClick={() => void run("default")}>
            恢复默认
          </Button>
          <span className="rts-gap" />
          <Button tip="不改了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={same ? "没有改动" : `保存：${sheet.label}马上就是这 ${picked.length} 条`} tone="primary" disabled={same || !!busy} onClick={() => void run("save")}>
            {busy === "save" ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
