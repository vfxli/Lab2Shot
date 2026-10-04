import { api } from "../api";
import { setCatalog } from "./catalog";
import { useResults } from "./results";
import { useSession } from "./session";
import { refreshTemplates } from "../editor/templatesList";
import { getLang, setLang, type Lang } from "../i18n/lang";
import { askStatus } from "../graph/asking";
import { followInLanguage } from "../graph/follow";
import { notesToSay } from "../model/nodeOutcome";

/** Switch the page's language (the account menu): the account keeps it (PUT /api/me/lang, which also sets this
 * browser's cookie), the page speaks it at once, and everything the server said in the old one is asked for again —
 * the node catalogue, the template cards, the login state (role names, …), the open graph's status (its messages and
 * plan) — without reloading the page. */
export async function switchLanguage(lang: Lang): Promise<void> {
  if (lang === getLang()) return;
  const got = await api.me.lang(lang);
  setLang(got.lang, true);
  // why a click could not be submitted was written on the blocking nodes in the old language (graph/actions.ts
  // sayBlocked): the marks go, the reasons stay in the log, which says them again in the new one (state/log.ts)
  useResults.setState((s) => ({
    blockedAt: -1,
    byNode: Object.fromEntries(Object.entries(s.byNode).map(([id, st]) => [id, st.blocked ? { ...st, blocked: undefined } : st])),
  }));
  followInLanguage(); // the followed job's stream: what the server says next is in the new language
  void sayNotesAgain(getLang());
  refreshTemplates();
  await Promise.all([
    api.catalog().then(setCatalog),
    useSession.getState().load(),
  ]);
  await askStatus(); // after the catalogue: the status reply is read against it
}

/** What the server said on the nodes' bottom rows (a queued node's reason: model/nodeOutcome.ts NoteSaid), said again by
 * it in `lang` (POST /api/said); the rows redraw with the new words. Not said again: the words it had. */
async function sayNotesAgain(lang: Lang): Promise<void> {
  const notes = notesToSay(Object.values(useResults.getState().byNode).map((st) => st.note), lang);
  if (!notes.length) return;
  const got = await api.me.said(notes.map((n) => ({ code: n.code, params: n.params }))).catch(() => null);
  if (!got) return;
  notes.forEach((n, i) => {
    if (got.texts[i]) n.texts[lang] = got.texts[i]!;
  });
  useResults.setState((s) => ({ byNode: { ...s.byNode } })); // the same notes, now with words in `lang`: redraw
}
