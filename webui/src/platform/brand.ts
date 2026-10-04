import { createElement } from "react";
import { useT } from "../i18n/t";

/** Who the page belongs to, shown on the gate and the welcome page: an element (`{COPYRIGHT}`) that says the line in the
 * page's language, following it when it changes (it subscribes: the same element is reused, so it renders by itself). */
function Copyright() {
  return useT()("ui.misc.copyright");
}
export const COPYRIGHT = createElement(Copyright);

/** The project's public page (the login page links it under the form, gate.tsx About). */
export const PROJECT_URL = "https://github.com/vfxli/Lab2Shot";
