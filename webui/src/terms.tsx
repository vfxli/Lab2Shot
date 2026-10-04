import { Fragment, useEffect, useState } from "react";
import { messageOf, type Message } from "./messages/message";
import { fromGate, json } from "./platform/http";
import { Button } from "./ui/Button";
import { Sheet } from "./ui/Sheet";
import { t } from "./i18n/t";
import { tipOf } from "./platform/tips";

/** 用户协议 and 隐私政策 (lab2shot/terms, server/terms.py) on the login page: the box to tick when registering
 * (register.tsx), and the card an account that has not agreed to the current version meets before anything else
 * (gate.tsx: one an administrator made, or anyone after the texts changed). Either opens each text in a sheet of its
 * own. The server refuses a registration without the box ticked, and everything but logging in and out to an account
 * that owes its agreement: the page only asks. Part of the login page (the gate), so it imports nothing of the page
 * behind it; the admin page's editor (admin/Terms.tsx) shows its preview with TermsSheet. */

export interface TermsDoc {
  id: string;
  title: string;
  text: string;
}

export interface TermsView {
  version: number;
  at: number;
  documents: TermsDoc[];
}

export const readTerms = () => fromGate(() => json<TermsView>("GET", "/api/terms", undefined, { cache: "no-store" }));

/** A line's **strong** parts. */
function inline(line: string): React.ReactNode {
  const parts = line.split("**");
  return parts.map((p, i) => (i % 2 ? <strong key={i}>{p}</strong> : <Fragment key={i}>{p}</Fragment>));
}

/** A text as written (a few marks of Markdown: `# ` its title, `## ` a heading, `- ` an item, `**…**` strong, a
 * blank line between paragraphs), as React elements: plain text, never HTML. */
export function TermsText({ text, skipTitle }: { text: string; skipTitle?: boolean }) {
  const blocks: React.ReactNode[] = [];
  let items: string[] = [];
  let para: string[] = [];
  const flush = () => {
    if (para.length) blocks.push(<p key={blocks.length}>{inline(para.join(""))}</p>);
    if (items.length)
      blocks.push(
        <ul key={blocks.length}>
          {items.map((x, i) => (
            <li key={i}>{inline(x)}</li>
          ))}
        </ul>,
      );
    para = [];
    items = [];
  };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) flush();
    else if (line.startsWith("# ")) {
      flush();
      if (!skipTitle) blocks.push(<h1 key={blocks.length}>{line.slice(2)}</h1>);
    } else if (line.startsWith("## ")) {
      flush();
      blocks.push(<h2 key={blocks.length}>{line.slice(3)}</h2>);
    } else if (line.startsWith("- ")) {
      if (para.length) flush();
      items.push(line.slice(2));
    } else {
      if (items.length) flush();
      para.push(line);
    }
  }
  flush();
  // a document read through, wrapped: not data-user-data, whose one rule cuts a value to one line
  return <div className="terms-text">{blocks}</div>;
}

/** One text in a sheet over the page. */
export function TermsSheet({ doc, onClose }: { doc: TermsDoc; onClose: () => void }) {
  return (
    <Sheet title={doc.title} width={680} onClose={onClose}>
      <TermsText text={doc.text} skipTitle />
    </Sheet>
  );
}

/** 「我已阅读并同意《用户协议》和《隐私政策》」: the box, and each title a link that opens its text. */
export function TermsBox({ view, on, bad, onChange }: { view: TermsView | null; on: boolean; bad?: boolean; onChange: (on: boolean) => void }) {
  const [open, setOpen] = useState<TermsDoc | null>(null);
  const link = (id: string, title: string) => { // title: the document's name, already in the page's language
    const doc = view?.documents.find((d) => d.id === id);
    return (
      <Button tone="link" disabled={!doc} onClick={() => doc && setOpen(doc)}>
        {t("ui.terms.quoted", { title })}
      </Button>
    );
  };
  return (
    <div className={`terms-box${bad ? " bad" : ""}`}>
      <label>
        <input type="checkbox" name="terms" checked={on} disabled={!view} onChange={(e) => onChange(e.target.checked)} />
        <span>{t("ui.terms.agree_box")}</span>
      </label>
      {link("agreement", t("ui.terms.agreement"))}
      <span>{t("ui.terms.and")}</span>
      {link("privacy", t("ui.terms.privacy"))}
      {open && <TermsSheet doc={open} onClose={() => setOpen(null)} />}
    </div>
  );
}

/** The card an account that owes its agreement meets (gate.tsx): the box whose titles open the texts, 同意并继续; 退出登录 to leave
 * without agreeing. `over`: shown over a page already open (the texts changed while it was). */
export function TermsCard({ over, onAgreed, onLeave }: { over: boolean; onAgreed: () => void; onLeave: () => void }) {
  const [view, setView] = useState<TermsView | null>(null);
  const [on, setOn] = useState(false);
  const [problem, setProblem] = useState<Message | null>(null);
  const [busy, setBusy] = useState(false);
  const load = () =>
    void readTerms().then(
      (v) => (setView(v), setOn(false)),
      (e) => setProblem(messageOf(e)),
    );
  useEffect(load, []);

  const agree = async () => {
    if (!view || !on || busy) return;
    setBusy(true);
    try {
      await fromGate(() => json<unknown>("POST", "/api/auth/terms", { version: view.version }));
      onAgreed();
    } catch (e) {
      setProblem(messageOf(e));
      load(); // the texts changed meanwhile: the new ones, to be read and ticked again
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-card glass clear terms-card" role="dialog" aria-label={t("ui.terms.both")}>
      <h1>{over ? t("ui.terms.both_changed") : t("ui.terms.both")}</h1>
      <p className="login-lede">
        {over
          ? t("ui.terms.changed_lede")
          : t("ui.terms.lede")}
        {view ? t("ui.terms.version", { version: view.version }) : ""}
      </p>
      <TermsBox view={view} on={on} onChange={(v) => (setOn(v), setProblem(null))} />
      {problem && (
        <p className="login-problem" role="alert" data-code={problem.code}>
          {problem.text}
        </p>
      )}
      <div className="login-stack">
        <Button tip={on ? undefined : tipOf("disabled", t("ui.terms.need_tick"))} tone="primary" size="lg" layout="login-go" disabled={!on || busy} onClick={() => void agree()}>
          {busy ? t("ui.terms.submitting") : t("ui.terms.agree")}
        </Button>
        <Button tone="ghost" onClick={onLeave}>
          {t("ui.terms.log_out")}
        </Button>
      </div>
    </div>
  );
}
