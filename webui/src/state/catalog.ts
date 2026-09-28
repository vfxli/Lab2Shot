import { useSyncExternalStore } from "react";
import type { Catalog, DataType, NodeTypeDef } from "../api";

/** The node catalog the server describes (/api/catalog): every node type this account may use, every data type,
 * every 3D kind. It is not this document's, this browser's or this account's own choices but the server's schema, the
 * same for every graph and every browser signed in as this account, fetched once and refreshed when another tab
 * installs an extension (App.tsx's focus handler). That is why it lives outside webui/src/state/'s zustand stores, as
 * a small hand-rolled external store: the zustand stores hold only state that is really owned by the page. */

let catalog: Catalog | null = null;
let text = ""; // the catalogue as last taken: the same answer again (every window focus reads it) changes nothing
let types: Record<string, DataType> = {};
let nodeDefs: Record<string, NodeTypeDef> = {};
const listeners = new Set<() => void>();

export function setCatalog(c: Catalog): void {
  const now = JSON.stringify(c);
  if (now === text) return; // unchanged: no node of the graph is redrawn for it
  text = now;
  catalog = c;
  types = Object.fromEntries(c.types.map((t) => [t.id, t]));
  nodeDefs = Object.fromEntries(c.nodes.map((n) => [n.id, n]));
  for (const l of listeners) l();
}

function subscribe(l: () => void): () => void {
  listeners.add(l);
  return () => listeners.delete(l);
}

export const getCatalog = (): Catalog | null => catalog;
export const getTypes = (): Record<string, DataType> => types;
export const getLayerPorts = (): Record<string, string> => catalog?.layer_ports ?? {};
export const getNodeDefs = (): Record<string, NodeTypeDef> => nodeDefs;

export const useCatalog = (): Catalog | null => useSyncExternalStore(subscribe, getCatalog);
export const useTypes = (): Record<string, DataType> => useSyncExternalStore(subscribe, getTypes);
