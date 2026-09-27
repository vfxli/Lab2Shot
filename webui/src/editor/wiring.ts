/** Wiring by clicks, as one state machine: pure state and transitions, so the editor component only feeds it
 * what happened and carries out what it answers (webui/tests/wiring.test.ts drives every transition without a browser).
 *
 * Houdini's way: a click on a port picks the wire up instead of holding the mouse down; a dashed line then follows the
 * pointer until the next click says where it goes. Dragging a wire is the node editor's own (@xyflow/react); this is
 * what happens when the pointer barely moved between press and release, which the editor treats as a click.
 *
 *   idle  --click an output-------------------->  wiring from that output
 *   idle  --click an input with a wire--------->  wiring from the wire's output, that input taken off (`detached`)
 *   idle  --click an input with no wire-------->  wiring from that input (backwards: the next click is an output)
 *   wiring --click a port that fits------------>  idle, and the wire is made (an input taken off is moved, not doubled)
 *   wiring --click empty canvas---------------->  idle, and: a taken-off wire is cut; anything else opens the node menu
 *                                                 for what the wire needs (only node types that fit are offered there)
 *   wiring --Esc, or a right click in place---->  idle, nothing changed
 *
 * A port that does not fit, and the port the wire started from, are no answer at all: the wiring stays as it is, so a
 * mis-click never loses the wire. */

/** One end of a wire: a node and one of its ports. */
export interface End {
  node: string;
  port: string;
}

/** Nothing being wired (null), or the wire the pointer is carrying. `from`: where it starts, an output when the wire
 * goes forwards, an input when it was started backwards from an unwired input. `detached`: the input whose wire was
 * taken off (its wire is drawn from its own output meanwhile, and is cut if the wire is let go on empty canvas). */
export type Wiring = null | {
  from: End;
  side: "output" | "input";
  detached: End | null;
};

export type WireEvent =
  /** A port was clicked. `wired`: for an input, the output whose wire ends there (the nearest one on a multi input), else null. */
  | { at: "port"; end: End; side: "output" | "input"; wired: End | null }
  /** Empty canvas was clicked, at this point on the screen. */
  | { at: "canvas"; x: number; y: number }
  /** Escape, or a right click that did not move: whatever is being wired is let go. */
  | { at: "cancel" };

export type WireAction =
  | { do: "nothing" }
  /** Make the wire. `was`: the input it was taken off (its old wire goes with the move), null when it is a new wire. */
  | { do: "connect"; from: End; to: End; was: End | null }
  /** Cut the wire that ends at this input (its head was let go on empty canvas). */
  | { do: "cut"; input: End }
  /** Offer the node types that fit at this point, and wire the one chosen (`side`: which side of it the wire takes). */
  | { do: "offer"; x: number; y: number; from: End; side: "output" | "input" };

/** Where the machine goes on an event, and what the editor is to do. `fits(output, input)`: may a wire from that output
 * end at that input (the server's rules, graph/rules.ts; never judged here). */
export function wiring(state: Wiring, event: WireEvent, fits: (output: End, input: End) => boolean): { state: Wiring; action: WireAction } {
  const nothing = (next: Wiring): { state: Wiring; action: WireAction } => ({ state: next, action: { do: "nothing" } });

  if (event.at === "cancel") return nothing(null);

  if (!state) {
    if (event.at !== "port") return nothing(null);
    if (event.side === "output") return nothing({ from: event.end, side: "output", detached: null });
    // an input: its wire is taken off and carried on from the output it came from; an unwired one is wired backwards
    return nothing(event.wired ? { from: event.wired, side: "output", detached: event.end } : { from: event.end, side: "input", detached: null });
  }

  if (event.at === "canvas")
    return state.detached
      ? { state: null, action: { do: "cut", input: state.detached } }
      : { state: null, action: { do: "offer", x: event.x, y: event.y, from: state.from, side: state.side } };

  // a port: it makes the wire when it is the other side and fits; anything else leaves the wiring as it is
  if (event.end.node === state.from.node && event.end.port === state.from.port) return nothing(state);
  if (event.side === state.side) return nothing(state);
  const [from, to] = state.side === "output" ? [state.from, event.end] : [event.end, state.from];
  if (from.node === to.node || !fits(from, to)) return nothing(state);
  return { state: null, action: { do: "connect", from, to, was: state.detached } };
}
