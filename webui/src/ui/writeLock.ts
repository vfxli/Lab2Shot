import { createContext, useContext } from "react";
import { readOnlyWhy, useReadOnly } from "../state/cookInputs";

/** Why nothing may be written here ("" writable): the one question every write control asks, inside an editor window
 * or not. An editor window (editor/ParamSheet.tsx SheetWindow) provides its reason (read-only tab, or the parameter
 * not applicable now) to everything in it, the 3D handles of its stages included; outside one it is the read-only tab.
 * Every write control reads this (parameter rows, node rows, the node graph, the 2D and 3D handles, paste);
 * useReadOnly itself is left to the page's frame (the read-only banner, the file menu, the tab's class). */
export const WriteLock = createContext<string | null>(null);

export function useWriteLock(): string {
  const why = useContext(WriteLock);
  const readOnly = useReadOnly();
  return why ?? (readOnly ? readOnlyWhy() : "");
}
