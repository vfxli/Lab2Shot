/** Node names (Houdini's): a node is called by its id (`retarget1`, `char_fbx`) and its type (`retarget`, `fbx.import`)
 * is said beside it. The page's twin of lab2shot/engine/naming.py: the same rule for a name, the same default numbering,
 * the same way a message points at a node. Pure but for the words, which are the page's language's (i18n/t.ts). */

import type { Words } from "../api";
import { pick, t } from "../i18n/t.ts";
import { sayWord, type Said } from "../messages/format.ts";

export const NAME_MOST = 32;
const NAME = /^[a-z][a-z0-9_]*$/;

/** How a message points at a node, as its parameter: the word `name（type）` / `name (type)` kept by its key, so the
 * message (and the log keeping it) says it again in another language with the brackets right (messages/format.ts
 * Said; the server's node_ref travels the same way). Its id alone when its type is not known. A comment never goes
 * into a message. */
export const nodeWord = (id: string, typeId: string | undefined): Said | string => (typeId ? { said: "ui.node.ref", params: { name: id, type: typeId } } : id);

/** The same, as text in the page's language (a label shown, not a message's parameter). */
export const nodeRef = (id: string, typeId: string | undefined): string => {
  const w = nodeWord(id, typeId);
  return typeof w === "string" ? w : sayWord(w);
};

/** The stem of a type's default node names: the type name with its dot an underscore (`fbx.import` → `fbx_import`). */
export const nameStem = (typeId: string): string => typeId.replace(/\./g, "_");

/** A new node's name: the type's stem and the least number not `taken` (`fbx_import1`, `retarget1`). The one place a
 * node id is made (a new node, a paste, a merge). */
export function defaultName(typeId: string, taken: (id: string) => boolean): string {
  const stem = nameStem(typeId);
  let n = 1;
  while (taken(`${stem}${n}`)) n++;
  return `${stem}${n}`;
}

/** What is wrong with `name` as the new name of a node (null: nothing). `taken`: another node of the graph has it. */
export function nameProblem(name: string, taken: (id: string) => boolean): string | null {
  if (!name) return t("ui.node.name_empty");
  if (name.length > NAME_MOST) return t("ui.node.name_long", { most: NAME_MOST });
  if (!NAME.test(name)) return t("ui.node.name_rule");
  if (taken(name)) return t("ui.node.name_taken", { name });
  return null;
}

/** Whether the node still has a default name of its type (`fbx_import3`): never renamed. */
export const isDefaultName = (id: string, typeId: string): boolean => new RegExp(`^${nameStem(typeId)}[0-9]+$`).test(id);

/** nodeRef of a node found in a list of drawn nodes (a snapshot's): its id alone when it is not there. */
export const refIn = (nodes: readonly { id: string; data: { typeId: string } }[], id: string): string => nodeRef(id, nodes.find((n) => n.id === id)?.data.typeId);

/** nodeWord of a node found in a list of drawn nodes: a message's parameter. */
export const wordIn = (nodes: readonly { id: string; data: { typeId: string } }[], id: string): Said | string => nodeWord(id, nodes.find((n) => n.id === id)?.data.typeId);

/** A group box's name as shown: the one given it, else (a new box keeps none: graph/edit.ts addBox) the default said now,
 * in the page's language, numbered by its id (`box:3` → 「分组 3」 / "Network Box 3"), so it follows the language. */
export function boxLabel(box: { id: string; label: Words }): string {
  return pick(box.label) || t("ui.graph.box_default", { n: Number(box.id.split(":")[1]) || 1 });
}
