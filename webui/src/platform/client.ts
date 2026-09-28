import { randomId } from "./randomId";

/** What this browser tells the server about itself with a job, an upload, a log or a feedback, for the administrator
 * (the server adds the address and the User-Agent). Who it is, is the account it logged in with: never anything the
 * page says. */
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

/** This browser's own random id, made once and kept in localStorage: sent at every login and registration (gate.tsx,
 * register.tsx) so the server's login log can tell "this browser, on this computer" apart from another (a browser
 * can't be told the computer's own name): the 设备 count on an account's page and the stricter lockout for a device
 * never seen before (lab2shot/accounts.py known_device) rest on it. Falls back to a fresh one each time when storage is unavailable (a
 * private window, blocked site data): logins still work, they are just not told apart as reliably. */
export function deviceId(): string {
  try {
    const kept = localStorage.getItem(DEVICE_ID_KEY);
    if (kept) return kept;
    const made = randomId();
    localStorage.setItem(DEVICE_ID_KEY, made);
    return made;
  } catch {
    return randomId();
  }
}
