/** What this browser tells the server about itself with a job or an upload, for the administrator (the server adds
 * the address and the User-Agent). Who it is, is the account it logged in with (account.ts): never anything the page
 * says. */
export function clientInfo(): Record<string, string> {
  return {
    app: "web",
    platform: navigator.platform,
    language: navigator.language,
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    screen: `${screen.width}×${screen.height}`,
  };
}

const DEVICE_ID_KEY = "lab2shot:device-id";

/** This browser's own random id, made once and kept in localStorage: sent at every login (gate.tsx) so the 用户 page's
 * 最近登录 can tell "this browser, on this computer" apart from another (a browser can't be told the computer's own
 * name — the tooltip where it is shown says so). Falls back to a fresh one each time when storage is unavailable (a
 * private window, blocked site data): logins still work, they are just not told apart as reliably. */
export function deviceId(): string {
  try {
    const kept = localStorage.getItem(DEVICE_ID_KEY);
    if (kept) return kept;
    const made = crypto.randomUUID();
    localStorage.setItem(DEVICE_ID_KEY, made);
    return made;
  } catch {
    return crypto.randomUUID();
  }
}
