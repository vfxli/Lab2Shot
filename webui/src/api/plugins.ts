import { json } from "../platform/http";

/** One DCC in 「DCC 插件」 (GET /api/plugins, lab2shot/server/plugins.py): offered or not, and where its zip is. */
export interface DccPlugin {
  id: string; // maya, houdini, nuke
  label: string;
  available: boolean;
  url: string; // the zip; "" when not offered
}

/** `open`: the setting plugins.download, the plugins are open to every user; false, only administrators get the list (it
 * is refused to anyone else, and the top bar does not offer it: server/available.py "plugins.download"). */
export const readPlugins = () => json<{ plugins: DccPlugin[]; open: boolean }>("GET", "/api/plugins");
