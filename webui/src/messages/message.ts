import { isLevel, levelOf, render, type Level, type Params } from "./format";

/** One message as the page keeps it: its code, level letter and words, made here from
 * the page's own catalogue (`msg`), or as the server sent it (`fromServer`: the server writes its own words). `port` /
 * `param`: the input or parameter it is about (clicking it in the 「消息」 panel goes there); `fix`: the node that,
 * inserted in front of `port`, puts it right; `refused`: a refused wire's reason (the node's error, not a warning). */
export interface Message {
  code: string;
  level: Level;
  text: string;
  port?: string;
  param?: string;
  fix?: { insert: string; label: string };
  refused?: boolean;
}

/** A server message (or a sentence already written) as the words a person reads: never an object in a line of text. */
export const textOf = (m: { text: string } | string | null | undefined): string => (typeof m === "string" ? m : m?.text ?? "");

/** A message of the page's own, by code and the parameters its template names. */
export const msg = (code: string, params: Params = {}, anchors: { port?: string; param?: string } = {}): Message => ({
  code,
  level: levelOf(code),
  text: render(code, params),
  ...anchors,
});

/** A message from the server (a cook event, a node's status, an answer's error). */
export function fromServer(m: { code?: string; level?: string; text?: string; port?: string; param?: string; fix?: Message["fix"]; refused?: boolean }): Message {
  const code = m.code ?? "E-COOK-PROBLEM";
  const level = isLevel(m.level) ? m.level : levelOf(code);
  return { code, level, text: m.text ?? "", ...(m.port ? { port: m.port } : {}), ...(m.param ? { param: m.param } : {}), ...(m.fix ? { fix: m.fix } : {}), ...(m.refused ? { refused: true } : {}) };
}

/** An error that is a message: thrown where something fails, it keeps the code (and its level) with the words, so
 * whoever catches it says the message itself (`messageOf`), never only its text. Made from the page's own code and
 * parameters, or from a message already made (the server's: platform/http.ts ApiError). */
export class MessageError extends Error {
  readonly said: Message;
  constructor(said: string | Message, params: Params = {}) {
    const m = typeof said === "string" ? msg(said, params) : said;
    super(m.text);
    this.said = m;
  }
}

/** What went wrong as a message: the one an error carries, or E-PAGE-ERROR with an error's own words (a bug of the
 * page's, or the browser's refusal), never a bare text. */
export const messageOf = (e: unknown): Message => (e instanceof MessageError ? e.said : msg("E-PAGE-ERROR", { detail: e instanceof Error ? e.message : String(e) }));

/** An error caught from a request (api.ts ok(): its `code` when the server gave one) or from the page itself, as the
 * reason a message of the page's own names. */
export const reasonOf = (e: unknown): string => (e instanceof Error ? e.message : String(e));

