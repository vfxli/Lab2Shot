/** What is there for whoever looks, as the server resolved it (lab2shot/availability.py): the one place every page reads
 * it. A subject (a node's parameter or a table row's field
 * ("layers[2].scale"), a section or a button of the admin page, an action on an account's row, a display option) is
 *   - available: shown and usable;
 *   - inactive: shown greyed, with why (a data condition: nothing wired, the rights ran out, the built-in administrator account);
 *   - pending: shown as usable; whether it did anything is known once it cooks;
 *   - in none of them: hidden when it is a subject that declares a condition (not this login's tool; nothing to change).
 * A node's parameter without a condition is not a subject: it is never in the answer and always shown.
 *
 * The answer comes with the node's status (`applies`), the login state (`applies`) and each row of 用户 (`applies`).
 * The viewer's display options are the browser's own state (what only changes how things look never waits for the
 * server), so their table (model/viewControls.ts) is resolved here into the same answer (resolveLocal).
 * Components only ask by id; none looks at a role, a capability or the lists themselves.
 *
 * Pure: types only. */

export interface MessageJson {
  code: string;
  level: string;
  text: string;
  params?: Record<string, unknown>;
}

/** The resolved answer (lab2shot/availability.py Availability.json()). */
export interface Availability {
  available: string[];
  inactive: Record<string, MessageJson>;
  pending?: Record<string, MessageJson>;
}

type Answer = Availability | null | undefined;

export const NOTHING: Availability = { available: [], inactive: {}, pending: {} };

/** Shown at all: available, greyed or pending. */
export const shown = (a: Answer, id: string): boolean => !!a && (a.available.includes(id) || id in a.inactive || id in (a.pending ?? {}));

/** Usable now. */
export const usable = (a: Answer, id: string): boolean => !!a && a.available.includes(id);

/** Greyed: shown, doing nothing here. */
export const greyed = (a: Answer, id: string): boolean => !!a && id in a.inactive;

/** A node type as a subject (server/available.py: `node:<type id>`): usable now, or installed but not usable with why
 * (E-EXT-NOTINSTALLED, E-EXT-OUTDATED, ...); a type the account may not use is in neither, and not in the catalogue. */
const nodeSubject = (typeId: string): string => `node:${typeId}`;
export const nodeUsable = (a: Answer, typeId: string): boolean => usable(a, nodeSubject(typeId));
export const nodeWhy = (a: Answer, typeId: string): string => why(a, nodeSubject(typeId));

/** Why it is greyed ("" when it is not). */
export const why = (a: Answer, id: string): string => a?.inactive[id]?.text ?? "";

/** What it waits for a cook to tell ("" when it does not). */
export const until = (a: Answer, id: string): string => a?.pending?.[id]?.text ?? "";

/** The items of a list (sections, by `id`) shown, in the list's order. */
export const visible = <T extends { id: string }>(items: readonly T[], a: Answer): T[] => items.filter((x) => shown(a, x.id));

/** What a page behind the admin login does for a login: "open" (something of it is available), "login" (not logged
 * in, or its rights ran out: the password again), "not-yours" (nothing of it is this login's). */
export function pageAccess(state: { user: unknown; applies: Availability } | null, page: string): "open" | "login" | "not-yours" {
  if (!state || !state.user) return "login";
  if (usable(state.applies, page)) return "open";
  return greyed(state.applies, page) ? "login" : "not-yours";
}

/** A table of the browser's own subjects (each: does it apply to these facts) resolved into the same answer. One that
 * does not apply is greyed with its reason (`why`, per subject: a control stays in
 * place and is only usable or not, rather than appearing and disappearing), or, with no reason written, left out
 * altogether as a 3D control on the 2D stage is. */
export function resolveLocal<F>(table: Readonly<Record<string, (facts: F) => boolean>>, facts: F, why: Readonly<Record<string, string>> = {}): Availability {
  const available: string[] = [];
  const inactive: Record<string, MessageJson> = {};
  for (const id of Object.keys(table)) {
    if (table[id](facts)) available.push(id);
    else if (why[id]) inactive[id] = { code: "", level: "I", text: why[id] }; // a hover tip beside the control, not a message of the message catalogue
  }
  return { available, inactive, pending: {} };
}
