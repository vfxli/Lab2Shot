import { useEffect } from "react";
import { create } from "zustand";
import { same as sameJson, type Json } from "../model/graphPatch";
import { api, type Account, type AuthState } from "../api";
import { NOTHING } from "../api/applies";
import { changedAccount, LOGGED_IN, sawAccount, SIGNED_OUT } from "../platform/http";
import { setLang } from "../i18n/lang";

/** This browser's login, the single copy shared by every page behind the gate: the account and what the server allows
 * it to use (`applies`, read only through api/applies.ts). The editor's top bar, the feedback dialog, the page log and
 * the admin pages all read it here. It is requested when a page starts, and again after a login over the page
 * (LOGGED_IN), after the administrator's rights expire (SIGNED_OUT), or when the tab regains focus. A login that turns
 * out to be another account's opens the page again (platform/http.ts sawAccount). The gate requests it independently
 * before the page loads (gate.tsx: its bundle contains none of the page's code). */

interface SessionState {
  state: AuthState | null; // null: not yet known
  load: () => Promise<void>;
  set: (s: AuthState) => void;
  logout: () => void;
}

const same = (a: AuthState | null, b: AuthState) =>
  !!a && sameJson(a.applies as unknown as Json, b.applies as unknown as Json) && a.passphrase === b.passphrase &&
  a.user?.id === b.user?.id && a.user?.name === b.user?.name && a.user?.department === b.user?.department && a.expires === b.expires;

// the first read of the login under way: a second asker while the page opens (the page's root and the editor both ask)
// waits for it instead of sending the same request again. Once the login is known every read is a new one (after a
// login over the page the answer must be the new login's, never one asked before it)
let asking: Promise<void> | null = null;

export const useSession = create<SessionState>((set, get) => ({
  state: null,
  load: () => (get().state ? readLogin() : (asking ??= readLogin().finally(() => (asking = null)))),
  set: (s) => (setLang(s.lang), set({ state: s })),
  // logs out on this whole browser (every other tab of it follows: platform/http.ts) and returns to the login page
  logout: () => void api.auth.logout().finally(() => (changedAccount(null), window.location.assign("/"))),
}));

async function readLogin(): Promise<void> {
  const { getState: get, setState: set } = useSession;
  try {
    const s = await api.auth.state();
    if (sawAccount(s.user?.id ?? null)) return; // another account is logged in now: the page opens again
    setLang(s.lang); // the language the server speaks to this login (the account's choice first)
    if (!same(get().state, s)) set({ state: s });
  } catch {
    if (!get().state) set({ state: { user: null, applies: NOTHING } }); // the gate asks for the login when needed
  }
}

let watchers = 0;
const reload = () => void useSession.getState().load();

/** Follows the login (logged in, signed out, window focus regained): the first watcher adds the window listeners and
 * the last one to stop removes them. */
function watchSession(): () => void {
  if (watchers++ === 0) {
    window.addEventListener(LOGGED_IN, reload);
    window.addEventListener(SIGNED_OUT, reload);
    window.addEventListener("focus", reload);
  }
  return () => {
    if (--watchers === 0) {
      window.removeEventListener(LOGGED_IN, reload);
      window.removeEventListener(SIGNED_OUT, reload);
      window.removeEventListener("focus", reload);
    }
  };
}

/** Reads the login now and follows it (called by every page root). */
export function useSessionWatch(): void {
  useEffect(() => {
    const stop = watchSession();
    if (!useSession.getState().state) reload();
    return stop;
  }, []);
}

/** The login of a page the gate has already admitted (AdminGate, the editor behind the gate): never null there. */
export function useSignedIn(): AuthState {
  const s = useSession((x) => x.state);
  if (!s) throw new Error("useSignedIn outside a page the login gate let in");
  return s;
}

/** The account this page is logged in as, reading the login first when it is not known yet (a page behind the gate is
 * always logged in; the gate asks again before anything here is used when it is not). */
export async function signedIn(): Promise<Account> {
  if (!useSession.getState().state) await useSession.getState().load();
  const user = useSession.getState().state?.user;
  if (!user) throw new Error("signedIn outside a page the login gate let in");
  return user;
}

/** The account currently logged in (null: none, or not yet known). */
export const account = (): Account | null => useSession.getState().state?.user ?? null;
