import type { ServerNoticeText } from ".";
import { json } from "../platform/http";

/** The administrator's notice (lab2shot/server/notice.py): one line every page shows at the top while it is
 * on. A page never polls this route: it follows the one poll of the server's state it already has (state/server.ts,
 * GET /api/server `notice` — when the notice last changed) and reads the notice itself only when that number changes.
 * Nothing of it reaches a browser before the login (the route needs one). */
export const readNotice = () => json<ServerNoticeText>("GET", "/api/notice");
