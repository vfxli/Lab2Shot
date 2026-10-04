import { useEffect, useRef, useState } from "react";
import { messageOf, type Message } from "./messages/message";
import { deviceId } from "./platform/client";
import { fromGate, json } from "./platform/http";
import { MIN_CHARS, nameProblem, passwordProblem, tidyName, usernameProblem } from "./platform/accountRules";
import { t } from "./i18n/t";
import { workerAsks } from "./platform/work";
import { Button } from "./ui/Button";
import { StageSelect } from "./ui/StageSelect";
import type { Department } from "./api/accounts";
import { readTerms, TermsBox, type TermsView } from "./terms";
import { tipOf } from "./platform/tips";

/** 注册: the login page's form for making one's own account (lab2shot/site/registration.py, server/register.py), there
 * only while the administrator has 开放注册 on. The same fields and rules as the admin page's 新建用户 (username,
 * Chinese name, 环节, the password twice; the invite code when 邀请码验证 is on), checked here first and again by the
 * server; the role is always 普通用户, and the new account is logged in at once. The box 「我已阅读并同意《用户协议》和
 * 《隐私政策》」 must be ticked (terms.tsx): the registration carries the version of the texts it showed, and the server
 * refuses one without it, or with a version that is no longer current (then the texts are read again).
 *
 * What only a person filling the form in gets past, all checked by the server: a proof of work (the page asks for a
 * challenge as the form opens and a worker answers it while the person types, platform/pow.ts: a second or two), a
 * decoy field no person sees (its name comes with the challenge), and a minimum time between the challenge and the
 * registration. A challenge is used up by the registration that carries its answer, so after any refusal the page
 * asks for a new one and answers it again. Part of the login page (the gate), so it imports nothing of the page
 * behind it. */

export interface RegisterInfo {
  open: boolean;
  invite?: boolean;
  stages?: Department[]; // value + how it shows (lab2shot/accounts.py departments_shown)
  paused?: boolean;
  min_fill_s?: number; // the least time between a challenge and the registration (server/register.py MIN_FILL_S)
}

interface Challenge {
  token: string;
  nonce: string;
  bits: number;
  trap: string;
}

const solveIt = workerAsks<{ answer: string }>(() => new Worker(new URL("./platform/powWorker.ts", import.meta.url), { type: "module" }));

/** A challenge, and its answer as it is being worked out. */
interface Proof {
  challenge: Challenge;
  answer: Promise<string>;
  at: number; // when it came (the server counts its minimum time to fill the form from then)
}

async function prove(): Promise<Proof> {
  const challenge = await fromGate(() => json<Challenge>("POST", "/api/auth/register/challenge"));
  return { challenge, answer: solveIt({ nonce: challenge.nonce, bits: challenge.bits }).then((a) => a.answer), at: Date.now() };
}

/** Until a person could have filled the form since the challenge came: sooner the server refuses it
 * (E-REGISTER-TOOFAST). Only after a refusal can a person be that quick, with the form already filled in. */
const filled = (p: Proof, minS: number) => new Promise((ok) => window.setTimeout(ok, Math.max(0, p.at + minS * 1000 + 300 - Date.now())));

export function RegisterForm({ info, onIn, onBack }: { info: RegisterInfo; onIn: (s: { user?: { id: number } | null; terms?: number | null }) => void; onBack: () => void }) {
  const [username, setUsername] = useState("");
  const [name, setName] = useState("");
  const [stage, setStage] = useState("");
  const [password, setPassword] = useState("");
  const [again, setAgain] = useState("");
  const [invite, setInvite] = useState("");
  const [trap, setTrap] = useState("");
  const [terms, setTerms] = useState<TermsView | null>(null);
  const [agreed, setAgreed] = useState(false);
  const [touched, setTouched] = useState(false);
  const [problem, setProblem] = useState<Message | null>(null);
  const [busy, setBusy] = useState<"" | "proving" | "sending">("");
  const proof = useRef<Promise<Proof> | null>(null);
  const field = useRef<HTMLInputElement>(null);

  const [trapName, setTrapName] = useState("website");
  const fresh = () => {
    const next = prove();
    next.then((p) => setTrapName(p.challenge.trap)).catch(() => undefined); // a failure is reported when the form is sent
    proof.current = next;
  };
  const readAgain = () => void readTerms().then((v) => (setTerms(v), setAgreed(false)), (e) => setProblem(messageOf(e)));
  useEffect(() => {
    field.current?.focus();
    fresh();
    readAgain();
  }, []);

  const wrong = {
    username: username ? usernameProblem(username) : t("ui.register.need_username"),
    name: nameProblem(name),
    stage: stage ? "" : t("ui.register.need_stage"),
    password: password ? passwordProblem(password, "", true) : t("ui.register.need_password"),
    again: password && again !== password ? t("ui.register.again_differs") : "",
    invite: info.invite && !invite.trim() ? t("ui.register.need_invite") : "",
    terms: agreed ? "" : t("ui.register.need_terms"),
  };
  const bad = Object.values(wrong).some(Boolean);
  const show = (k: keyof typeof wrong) => (touched ? wrong[k] : "");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setTouched(true);
    if (bad || busy) return;
    setProblem(null);
    setBusy("proving");
    try {
      const p = await (proof.current ?? prove());
      const { challenge } = p;
      const solved = await p.answer;
      await filled(p, info.min_fill_s ?? 0);
      setBusy("sending");
      const s = await fromGate(() =>
        json<{ user?: { id: number } | null; terms?: number | null }>("POST", "/api/auth/register", {
          username,
          name: tidyName(name),
          department: stage,
          password,
          again,
          invite: invite.trim(),
          challenge: challenge.token,
          answer: solved,
          trap,
          terms: agreed && terms ? terms.version : 0,
          device_id: deviceId(),
        }),
      );
      onIn(s);
    } catch (err) {
      const said = messageOf(err);
      setProblem(said);
      fresh(); // the challenge was used up (or never came): the next try needs a new one
      if (said.code === "E-TERMS-CHANGED") readAgain(); // the new texts, to be read and ticked again
    } finally {
      setBusy("");
    }
  };

  return (
    <form className="login-card glass clear register-card" aria-label={t("ui.register.title")} onSubmit={(e) => void submit(e)} noValidate>
      <h1>{t("ui.register.title")}</h1>
      <p className="login-lede">{info.invite ? t("ui.register.lede_invited") : t("ui.register.lede")}</p>
      <label className="login-field">
        <span>{t("ui.register.username")}</span>
        <input ref={field} className={`field lg${show("username") ? " bad" : ""}`} name="username" autoComplete="username" spellCheck={false}
          placeholder={t("ui.register.username_placeholder")} value={username}
          onChange={(e) => (setUsername(e.target.value.toLowerCase().trim()), setProblem(null))} />
        {show("username") && <em className="login-why">{show("username")}</em>}
      </label>
      <label className="login-field">
        <span>{t("ui.register.name")}</span>
        <input className={`field lg${show("name") ? " bad" : ""}`} name="realname" autoComplete="name" placeholder={t("ui.register.name_placeholder")} value={name}
          onChange={(e) => (setName(e.target.value), setProblem(null))} />
        {show("name") && <em className="login-why">{show("name")}</em>}
      </label>
      <div className="login-field">
        <span>{t("ui.register.stage")}</span>
        <StageSelect large list={info.stages ?? []} value={stage} bad={!!show("stage")} onPick={(d) => (setStage(d), setProblem(null))} />
        {show("stage") && <em className="login-why">{show("stage")}</em>}
      </div>
      <label className="login-field">
        <span>{t("ui.register.password")}</span>
        <input className={`field lg${show("password") ? " bad" : ""}`} type="password" name="password" autoComplete="new-password" placeholder={t("ui.register.password_placeholder", { count: MIN_CHARS })} value={password}
          onChange={(e) => (setPassword(e.target.value), setProblem(null))} />
        {show("password") && <em className="login-why">{show("password")}</em>}
      </label>
      <label className="login-field">
        <span>{t("ui.register.again")}</span>
        <input className={`field lg${show("again") ? " bad" : ""}`} type="password" name="again" autoComplete="new-password" value={again}
          onChange={(e) => (setAgain(e.target.value), setProblem(null))} />
        {show("again") && <em className="login-why">{show("again")}</em>}
      </label>
      {info.invite && (
        <label className="login-field">
          <span>{t("ui.register.invite")}</span>
          <input className={`field lg${show("invite") ? " bad" : ""}`} name="invite" autoComplete="off" spellCheck={false} value={invite}
            onChange={(e) => (setInvite(e.target.value), setProblem(null))} />
          {show("invite") && <em className="login-why">{show("invite")}</em>}
        </label>
      )}
      <TermsBox view={terms} on={agreed} bad={!!show("terms")} onChange={(v) => (setAgreed(v), setProblem(null))} />
      {show("terms") && <em className="login-why">{show("terms")}</em>}
      {/* the decoy: no person sees or reaches it (styles/login.css), a form-filling bot fills it */}
      <label className="login-trap" aria-hidden="true">
        {trapName}
        <input tabIndex={-1} name={trapName} autoComplete="off" value={trap} onChange={(e) => setTrap(e.target.value)} />
      </label>
      {problem && (
        <p className="login-problem" role="alert" data-code={problem.code}>
          {problem.text}
        </p>
      )}
      <div className="login-stack">
        <Button tip={bad && touched ? tipOf("disabled", t("ui.register.fix_marked")) : undefined} tone="primary" size="lg" layout="login-go" type="submit" disabled={!!busy || (touched && bad)}>
          {busy === "proving" ? t("ui.register.proving") : busy === "sending" ? t("ui.register.sending") : t("ui.register.title")}
        </Button>
        <Button tone="ghost" type="button" onClick={onBack}>
          {t("ui.register.back")}
        </Button>
      </div>
    </form>
  );
}

/** Whether this server lets people register now (the login page shows 「注册」 only then). */
export async function registerInfo(): Promise<RegisterInfo> {
  try {
    return await fromGate(() => json<RegisterInfo>("GET", "/api/auth/register", undefined, { cache: "no-store" }));
  } catch {
    return { open: false };
  }
}
