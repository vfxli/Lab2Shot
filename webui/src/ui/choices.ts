import { useEffect, useState } from "react";
import { api, type Choice, type ParamDef } from "../api";
import { getNodeDefs } from "../state/catalog";
import { useCookInputs } from "../state/cookInputs";
import { useResults } from "../state/results";

// Options that come from a file or from what is wired in (P(choices_from), NodeDef.choices), asked from the server
// whenever what they come from changes: the parameters it names, the results cooked for the input ports it names
// (not asked while one of those has not been cooked). One request per question, shared by every control asking it.
const cache = new Map<string, Promise<Record<string, Choice>>>();

/** Every option set NodeDef.choices gives for what `p`'s options come from (one answer covers every parameter
 * listed from the same things: an import node's five selections come with one read of its file). Null while they are
 * not known (nothing to ask yet, or asking); an error comes back as {"": {options: [], empty: its message}}. */
export function useChoiceSet(nodeId: string, p: ParamDef): Record<string, Choice> | null {
  const typeId = useCookInputs((s) => s.nodes[nodeId]?.typeId ?? "");
  const nodeParams = useCookInputs((s) => s.nodes[nodeId]?.params);
  const results = useResults((s) => s.results);
  const from = (() => {
    const def = getNodeDefs()[typeId];
    if (!nodeParams || !def) return "";
    const inputs: Record<string, string> = {};
    const values: Record<string, unknown> = {};
    for (const name of p.choices_from) {
      if (!def.inputs.some((i) => i.name === name)) {
        values[name] = nodeParams[name] ?? null;
        continue;
      }
      // the packet that stands for this input port is given by the server in the status reply (nodes[id].stand_ins,
      // engine/evaluation.py stand_ins: the wired-in packet or, while it is not cooked yet, the nearest upstream one
      // that can stand for it: a per-item block's list, the whole before it was split into a list, the branch a
      // switch selects); the page never looks for one itself
      const fp = results[nodeId]?.stand_ins?.[name];
      if (!fp) return "";
      inputs[name] = fp;
    }
    return JSON.stringify([inputs, values]);
  })();
  const [got, setGot] = useState<Record<string, Choice> | null>(null);
  const key = `${typeId}|${from}`;
  useEffect(() => {
    if (!from) return setGot(null);
    let live = true;
    if (!cache.has(key)) {
      if (cache.size > 64) cache.clear();
      // only what the options come from goes with the question (the others as their defaults): an import node's 3,000
      // selected paths stay home
      const [inputs, values] = JSON.parse(from) as [Record<string, string>, Record<string, unknown>];
      const asked = api.choices(typeId, { ...getNodeDefs()[typeId]?.defaults, ...values }, inputs);
      asked.catch(() => cache.delete(key)); // asked again next time (a file that couldn't be read then)
      cache.set(key, asked);
    }
    cache
      .get(key)!
      .then((c) => live && setGot(c))
      .catch((e: Error) => live && setGot({ "": { options: [], empty: e.message } }));
    return () => {
      live = false;
    };
  }, [key]); // eslint-disable-line react-hooks/exhaustive-deps
  return got;
}

/** One parameter's options (see useChoiceSet). */
export function useChoices(nodeId: string, p: ParamDef): Choice | null {
  const set = useChoiceSet(nodeId, p);
  return set ? (set[p.name] ?? set[""] ?? null) : null;
}
