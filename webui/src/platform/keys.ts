import { createContext, useContext, useEffect, useRef } from "react";

/** The page's keyboard shortcut registry: a single window listener dispatches each key to the shortcut that should
 * receive it; no component listens for keys on its own.
 *
 * Dispatch order:
 *   1. only the topmost layer's shortcuts (a sheet, a popover, a menu: while one is open, nothing beneath it receives keys);
 *   2. within that layer (the page itself is layer 0), shortcuts tied to an element under the pointer come first (the 2D
 *      stage's F, the 3D stage's H / F / Esc, the node graph's Tab), innermost first, then the layer's other shortcuts;
 *   3. a shortcut whose `run` returns false declines the key, and the next one is tried.
 * While the user is typing (a text field, a text area, a pull-down, editable text) only shortcuts marked `inText` run
 * (Ctrl+S, Ctrl+Enter, a layer's Esc); a slider, checkbox, radio or button keeps the keys that operate it. */

interface Shortcut {
  keys: string[]; // keyName(): "mod+s", "mod+shift+z", "shift+o", "space", "arrowleft", "f", "escape", "tab"; "*": any key without a modifier
  run: (e: KeyboardEvent) => void | boolean; // false: declined (the next shortcut receives the key)
  inText?: boolean; // also runs while typing
}

export interface Entry {
  layer: number; // 0: the page
  over?: () => Element | null; // only while the pointer is over this element
  shortcut: () => Shortcut;
}

/** Whether a key press belongs to an input method still composing (pinyin being turned into characters): its Enter or
 * Escape confirms or drops the characters and is not the field's or the page's key. Safari reports that Enter with
 * keyCode 229 instead of isComposing. The one test every Enter handler and the registry below use. */
export function composing(e: Pick<KeyboardEvent, "isComposing" | "keyCode"> | { nativeEvent: Pick<KeyboardEvent, "isComposing" | "keyCode"> }): boolean {
  const n = "nativeEvent" in e ? e.nativeEvent : e;
  return n.isComposing || n.keyCode === 229;
}

/** The key of an event as shortcuts name it: modifiers (mod = Ctrl, or ⌘ on a Mac; alt; shift) then the key. */
function keyName(e: Pick<KeyboardEvent, "key" | "ctrlKey" | "metaKey" | "shiftKey" | "altKey">): string {
  const key = e.key === " " ? "space" : e.key.toLowerCase();
  return [e.ctrlKey || e.metaKey ? "mod" : "", e.altKey ? "alt" : "", e.shiftKey ? "shift" : "", key].filter(Boolean).join("+");
}

const OPERATES = /^(arrow(left|right|up|down)|space|enter|home|end|pageup|pagedown)$/;

/** Whether the focused element consumes this key itself: "typing" (every key except inText shortcuts), "control" (a
 * slider's arrows, a button's space), or null. */
function ownsKey(el: EventTarget | null, name: string): "typing" | "control" | null {
  if (typeof HTMLElement === "undefined" || !(el instanceof HTMLElement)) return null;
  if (el.isContentEditable || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return "typing";
  if (el.tagName === "INPUT") {
    const type = (el as HTMLInputElement).type;
    if (!["range", "checkbox", "radio", "button", "submit", "reset", "color", "file", "image"].includes(type)) return "typing";
    return OPERATES.test(name) ? "control" : null;
  }
  return el.tagName === "BUTTON" && /^(space|enter)$/.test(name) ? "control" : null;
}

const depth = (el: Element | null) => {
  let n = 0;
  for (let p = el; p; p = p.parentElement) n++;
  return n;
};

/** The shortcuts that may take a key, in the order they are tried (pure). */
function candidates(list: Entry[], name: string, owner: "typing" | "control" | null, under: (el: Element) => boolean): Shortcut[] {
  if (owner === "control") return [];
  const top = Math.max(0, ...list.map((e) => e.layer));
  const here = list.filter((e) => e.layer === top);
  const hovered = here
    .flatMap((e) => {
      const el = e.over?.();
      return el && under(el) ? [{ e, d: depth(el) }] : [];
    })
    .sort((a, b) => b.d - a.d)
    .map((x) => x.e);
  const rest = here.filter((e) => !e.over);
  return [...hovered, ...rest].map((e) => e.shortcut()).filter((s) => (s.keys.includes(name) || (s.keys.includes("*") && !name.includes("+"))) && (owner !== "typing" || s.inText));
}

const entries = new Set<Entry>();
let pointer: { x: number; y: number } | null = null;
let installed = false;
let layers = 0;

function install(): void {
  if (installed || typeof window === "undefined") return;
  installed = true;
  window.addEventListener("pointermove", (e) => (pointer = { x: e.clientX, y: e.clientY }), { passive: true, capture: true });
  window.addEventListener("keydown", (e) => {
    if (composing(e)) return; // the input method's own key
    const name = keyName(e);
    const under = (el: Element) => {
      if (!pointer) return false;
      const r = el.getBoundingClientRect();
      return pointer.x >= r.left && pointer.x <= r.right && pointer.y >= r.top && pointer.y <= r.bottom;
    };
    for (const s of candidates([...entries], name, ownsKey(e.target, name), under)) {
      if (s.run(e) !== false) {
        e.preventDefault();
        return;
      }
    }
  });
}

/** A key that does one thing while held and another when released without having done anything. Space over the node
 * graph pans the canvas with the left button while it is down (@xyflow/react panActivationKeyCode), and a press and
 * release with no drag in between still toggles 播放 / 暂停, so the playhead keeps working while the pointer rests over
 * the graph. `did`: whether the held key did its work (something was dragged); `again`: re-dispatches the key to the
 * shortcut that would otherwise have received it (the registry sees an ordinary press). The key release is observed
 * here, in the only place that listens for keys; `stop` removes that listener.
 * Pure over `did` and `again`. */
export function heldKey(key: string, did: () => boolean, again: () => void) {
  let held = false;
  let sending = false;
  const out = {
    /** Key down: returns `false` (declined) only for the key this re-dispatched. */
    down(): false | undefined {
      if (sending) return false;
      held = true;
      return undefined;
    },
    /** Key up: if nothing was done while it was held, the key is delivered to the page. */
    up(): void {
      if (!held) return;
      held = false;
      if (did()) return;
      sending = true;
      again();
      sending = false;
    },
    held: () => held,
    stop: () => undefined as void,
  };
  if (typeof window !== "undefined") {
    const up = (e: KeyboardEvent) => e.key === key && out.up();
    window.addEventListener("keyup", up);
    out.stop = () => window.removeEventListener("keyup", up);
  }
  return out;
}

/** Adds a shortcut to the registry (installing the single window listener on first use); the returned function removes
 * it. A subscriber registry: it holds only what is mounted. */
export function register(entry: Entry): () => void {
  install();
  entries.add(entry);
  return () => void entries.delete(entry);
}

/** The layer the component sits in (a sheet or a popover provides it to what it holds); 0: the page. */
export const KeyLayer = createContext(0);

/** Opens a new layer while `open` (a sheet, a popover, a menu): returns its number, used to register the layer's own
 * shortcuts and to provide to its contents (KeyLayer); 0 while closed. */
export function useKeyLayer(open: boolean): number {
  const order = useRef(0);
  if (open && !order.current) order.current = ++layers;
  if (!open) order.current = 0;
  return order.current;
}

/** Registers a shortcut in the component's layer (or `layer`); `over`: only while the pointer is over that element. */
export function useShortcut(shortcut: Shortcut, options: { over?: () => Element | null; layer?: number; enabled?: boolean } = {}): void {
  const inherited = useContext(KeyLayer);
  const layer = options.layer ?? inherited;
  const enabled = options.enabled ?? true;
  const latest = useRef(shortcut);
  latest.current = shortcut;
  const over = useRef(options.over);
  over.current = options.over;
  useEffect(() => {
    if (!enabled) return;
    return register({ layer, shortcut: () => latest.current, ...(options.over ? { over: () => over.current?.() ?? null } : {}) });
  }, [enabled, layer, !!options.over]); // eslint-disable-line react-hooks/exhaustive-deps
}
