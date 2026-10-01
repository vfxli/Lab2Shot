/** 用户's dialogs: 新建用户, 修改, 重设密码, 删除, 永久删除 (what each does is the server's: lab2shot/accounts.py). */

import { useState } from "react";
import { Sheet } from "../ui/Sheet";
import { adminApi, type UserChange, type UserRow, type UsersView } from "../api/admin";
import { nameProblem, passwordProblem, tidyName, usernameProblem } from "../platform/accountRules";
import { shown, usable, why, type Availability } from "../api/applies";
import { Button } from "../ui/Button";
import { DAY_S, dateValue, endOf, now, roleTip } from "./Users";
import { Roles, Row, TAGS_TIP, Tags } from "./userFields";
import { StageSelect } from "../ui/StageSelect";

export function NewUserDialog({ view, a, onClose, onMade }: { view: UsersView; a: Availability | null; onClose: () => void; onMade: (v: UsersView, made: { username: string; password: string }) => void }) {
  const [pick, setPick] = useState(view.default_role);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [name, setName] = useState("");
  const [dept, setDept] = useState("");
  const [expires, setExpires] = useState(dateValue(now() + 30 * DAY_S));
  const [allowed, setAllowed] = useState<string[]>(view.allowed_new);
  const [touched, setTouched] = useState(false);
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const wrong = {
    username: username ? usernameProblem(username) : "要填用户名",
    password: password ? passwordProblem(password, again).replace("新密码", "密码") : "要填初始密码",
    again: password && again !== password ? "两次输入的密码不一样" : "",
    name: nameProblem(name),
    dept: dept ? "" : "要选环节",
    expires: !expires ? "要选到期时间" : endOf(expires) <= now() ? "到期时间要在以后" : "",
  };
  const bad = Object.values(wrong).some(Boolean);

  const submit = async () => {
    setTouched(true);
    if (bad || busy) return;
    setBusy(true);
    try {
      const r = await adminApi.createUser({
        username,
        password,
        name: tidyName(name),
        department: dept,
        expires: endOf(expires),
        ...(usable(a, "users.tags") ? { tags: allowed } : {}), // otherwise the server gives a new account the default
        ...(usable(a, "users.role") ? { role: pick } : {}),
      });
      onMade(r, { username: r.user.username, password });
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  const show = (k: keyof typeof wrong) => (touched ? wrong[k] : "");

  return (
    <Sheet title="新建用户" width={560} onClose={onClose}>
      <div className="usr-form">
        <Row label="用户名" tip="登录用的名字：小写英文字母开头，3 到 32 个字符，可以用小写字母、数字和 _ . -" why={show("username")}>
          <input className={`field${show("username") ? " bad" : ""}`} value={username} autoFocus spellCheck={false} placeholder="如 zhangsan" data-tip="小写英文字母开头，只用小写字母、数字和 _ . -"
            onChange={(e) => (setUsername(e.target.value.toLowerCase().trim()), setRefused(""))} />
        </Row>
        <Row label="初始密码" tip="至少 8 个字符；建好后告诉他，他登录后可以自己改" why={show("password")}>
          <input className={`field${show("password") ? " bad" : ""}`} type="password" autoComplete="new-password" value={password} data-tip="至少 8 个字符" onChange={(e) => (setPassword(e.target.value), setRefused(""))} />
        </Row>
        <Row label="再输一次" tip="再输一次初始密码，防止输错" why={show("again")}>
          <input className={`field${show("again") ? " bad" : ""}`} type="password" autoComplete="new-password" value={again} data-tip="和上面的一样" onChange={(e) => setAgain(e.target.value)} />
        </Row>
        <Row label="中文名" tip="他的中文名，2 到 6 个字；少数民族名字的几部分用 · 隔开。统计和队列里都显示它" why={show("name")}>
          <input className={`field${show("name") ? " bad" : ""}`} value={name} placeholder="如 张三" data-tip="2 到 6 个中文字" onChange={(e) => (setName(e.target.value), setRefused(""))} />
        </Row>
        <Row label="环节" tip="他在制作里属于哪个环节，使用统计按它分环节；环节表在「账号设置」的「环节」里改" why={show("dept")}>
          <StageSelect list={view.departments} value={dept} bad={!!show("dept")} onPick={(d) => (setDept(d), setRefused(""))} />
        </Row>
        <Row label="到期" tip="这天过完就登录不了，已经登录的也马上退出；以后可以延期" why={show("expires")}>
          <input className={`field${show("expires") ? " bad" : ""}`} type="date" value={expires} min={dateValue(now())} data-tip="默认 30 天以后" onChange={(e) => setExpires(e.target.value)} />
        </Row>
        {usable(a, "users.role") && (
          <Row label="角色" tip={roleTip(view.roles)}>
            <Roles list={view.roles} value={pick} onPick={setPick} />
          </Row>
        )}
        {usable(a, "users.tags") && (
          <Row label="可用" tip={TAGS_TIP}>
            <Tags view={view} value={allowed} onChange={setAllowed} />
          </Row>
        )}
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不建了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={bad ? "先把上面紫框里的填好" : "建好账号，再把用户名和密码告诉他"} tone="primary" disabled={busy || (touched && bad)} onClick={() => void submit()}>
            {busy ? "新建中…" : "新建"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function EditDialog({ view, u, onClose, onDone }: { view: UsersView; u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [name, setName] = useState(u.name);
  const [dept, setDept] = useState(u.department);
  const [expires, setExpires] = useState(u.expires ? dateValue(u.expires) : "");
  const [allowed, setAllowed] = useState<string[]>(u.tags);
  const [pick, setPick] = useState(u.role);
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const wrong = {
    name: nameProblem(name),
    expires: !usable(u.applies, "account.expiry") ? "" : !expires ? "要选到期时间" : "",
  };
  const bad = Object.values(wrong).some(Boolean);

  const submit = async () => {
    if (bad || busy) return;
    setBusy(true);
    const change: UserChange = { name: tidyName(name), department: dept };
    if (usable(u.applies, "account.expiry")) change.expires = endOf(expires);
    if (usable(u.applies, "account.tags")) change.tags = allowed;
    if (usable(u.applies, "account.role")) change.role = pick;
    try {
      onDone(await adminApi.changeUser(u.id, change));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={`改 ${u.username}`} width={560} onClose={onClose}>
      <div className="usr-form">
        <Row label="中文名" tip="统计和队列里显示的中文名，2 到 6 个字" why={wrong.name}>
          <input className={`field${wrong.name ? " bad" : ""}`} value={name} autoFocus data-tip="2 到 6 个中文字" onChange={(e) => (setName(e.target.value), setRefused(""))} />
        </Row>
        <Row label="环节" tip="他在制作里属于哪个环节；使用统计按它分环节，以前的任务也跟着算到新环节">
          <StageSelect list={view.departments} value={dept} onPick={(d) => (setDept(d), setRefused(""))} />
        </Row>
        {shown(u.applies, "account.role") && (
          <Row label="角色" tip={roleTip(view.roles)}>
            {usable(u.applies, "account.role") ? <Roles list={view.roles} value={pick} onPick={setPick} /> : <span className="usr-static" data-tip={why(u.applies, "account.role")}>{u.role_label}</span>}
          </Row>
        )}
        {shown(u.applies, "account.expiry") && (
          <Row label="到期" tip="这天过完就登录不了，已经登录的也马上退出；改到以前的日期就是马上到期" why={wrong.expires}>
            {usable(u.applies, "account.expiry") ? (
              <input className={`field${wrong.expires ? " bad" : ""}`} type="date" value={expires} data-tip="这天过完就到期" onChange={(e) => (setExpires(e.target.value), setRefused(""))} />
            ) : (
              <span className="usr-static" data-tip={why(u.applies, "account.expiry")}>不过期</span>
            )}
          </Row>
        )}
        {shown(u.applies, "account.tags") && (
          <Row label="可用" tip={TAGS_TIP}>
            {usable(u.applies, "account.tags") ? <Tags view={view} value={allowed} onChange={setAllowed} /> : <span className="usr-static" data-tip={why(u.applies, "account.tags")}>全部</span>}
          </Row>
        )}
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不改了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={bad ? "先把上面紫框里的填好" : "保存；停用或到期马上生效"} tone="primary" disabled={bad || busy} onClick={() => void submit()}>
            {busy ? "保存中…" : "保存"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function ResetDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView, made: { username: string; password: string }) => void }) {
  const [next, setNext] = useState("");
  const [again, setAgain] = useState("");
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const rule = passwordProblem(next, again);
  const ready = !!next && next === again && !rule;

  const submit = async () => {
    if (!ready || busy) return;
    setBusy(true);
    try {
      onDone(await adminApi.resetPassword(u.id, next), { username: u.username, password: next });
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Sheet title={`给 ${u.username} 设新密码`} width={480} onClose={onClose}>
      <div className="usr-form">
        <p className="tpl-desc">他所有的登录马上退出，要用新密码重新登录（给自己设的，这个浏览器接着用）。把新密码告诉他。</p>
        <Row label="新密码" tip="至少 8 个字符">
          <input className="field" type="password" autoComplete="new-password" value={next} autoFocus data-tip="至少 8 个字符" onChange={(e) => (setNext(e.target.value), setRefused(""))} />
        </Row>
        <Row label="再输一次" tip="再输一次新密码，防止输错" why={rule || refused}>
          <input className="field" type="password" autoComplete="new-password" value={again} data-tip="和上面的一样" onChange={(e) => (setAgain(e.target.value), setRefused(""))} />
        </Row>
        <div className="dialog-row usr-end">
          <Button tip="不改了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip={ready ? "设成新密码" : "先填两次一样的新密码"} tone="primary" disabled={!ready || busy} onClick={() => void submit()}>
            {busy ? "设置中…" : "设新密码"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

/** 永久删除 takes two steps: 删除 first moves the account to 已删除, and 永久删除 is done from there, so no single
 * slip is irreversible. What is kept and what is removed is decided by the server alone (lab2shot/accounts.py purge);
 * this dialog only says it. */
export function PurgeDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      onDone(await adminApi.purgeUser(u.id));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={`永久删除 ${u.username}`} width={480} onClose={onClose}>
      <div className="usr-form">
        <p className="tpl-desc">
          永久删除 {u.name}（{u.username}）：账号从列表里彻底消失，用户名可以给新人重用；他存在服务器上的节点图一起删掉。
          任务记录、反馈和登录记录留着，统计里显示成「已删除的用户」。不能撤销。
        </p>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不删了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip="彻底删掉这个账号，不能撤销" tone="primary" danger disabled={busy} onClick={() => void submit()}>
            {busy ? "删除中…" : "永久删除"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}

export function DeleteDialog({ u, onClose, onDone }: { u: UserRow; onClose: () => void; onDone: (v: UsersView) => void }) {
  const [refused, setRefused] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    setBusy(true);
    try {
      onDone(await adminApi.deleteUser(u.id));
    } catch (e) {
      setRefused((e as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Sheet title={`删除 ${u.username}`} width={480} onClose={onClose}>
      <div className="usr-form">
        <p className="tpl-desc">
          删除 {u.name}（{u.username}）：登录马上失效，排队和计算中的任务停下，结果和上传的引用删掉；任务记录留着，统计里算作「已删除的用户」。不能撤销；只想暂时不让他用，点「停用」就好。
        </p>
        {refused && <div className="usr-why">{refused}</div>}
        <div className="dialog-row usr-end">
          <Button tip="不删了" tone="ghost" onClick={onClose}>
            取消
          </Button>
          <Button tip="删除这个账号，不能撤销" tone="primary" danger disabled={busy} onClick={() => void submit()}>
            {busy ? "删除中…" : "删除"}
          </Button>
        </div>
      </div>
    </Sheet>
  );
}
