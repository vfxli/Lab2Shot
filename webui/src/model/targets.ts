/** The node parameters an exposed parameter drives (api/catalog.ts ExposedParam `target`): one `node.param` key, or
 * several of one kind. The one reading of `target` on the page (the server's engine/templates.py targets_of); pure, so
 * the models (model/groupSwitches.ts) and their tests read it too. state/cookInputs.ts re-exports it. */

import type { ExposedParam } from "../api/catalog.ts";

/** Its targets, in order. They hold one value of one kind (the server checks it: E-EXPOSED-TARGETS), so the first
 * gives the control and the value; which node the row speaks for is shownTarget's. */
export const targetsOf = (x: Pick<ExposedParam, "target">): string[] => (Array.isArray(x.target) ? x.target : [x.target]);

/** A `node.param` key as [node id, parameter name] (the name "" when there is no dot). */
export function splitTarget(key: string): [string, string] {
  const i = key.indexOf(".");
  return i < 0 ? [key, ""] : [key.slice(0, i), key.slice(i + 1)];
}

/** The entry's first target: where its control and value are read (every target holds the same). */
export const firstTarget = (x: Pick<ExposedParam, "target">): [string, string] => splitTarget(targetsOf(x)[0] ?? "");

/** The node parameter the row speaks for — why it is greyed, 「待更新」, the node 「在视图里点选」 and
 * 「修改后在视图里显示这个节点」 go to: its first target whose node is on now, else (every one off, or none known yet)
 * its first. `off`: a node switched off now (model/nodeOutcome.ts switchedOffNodes). The one rule for it: an entry
 * driving TAPNext++ and CoTracker3 with only TAPNext++ on speaks for TAPNext++, not for the tracker that does not cook. */
export function shownTarget(x: Pick<ExposedParam, "target">, off: (node: string) => boolean): [string, string] {
  const all = targetsOf(x).map(splitTarget);
  return all.find(([node]) => !off(node)) ?? all[0] ?? ["", ""];
}

/** Whether the entry drives this node parameter. */
export const drives = (x: Pick<ExposedParam, "target">, key: string): boolean => targetsOf(x).includes(key);
