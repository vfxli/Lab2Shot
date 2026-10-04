import { resolveLocal, type Availability } from "../api/applies";
import { msg, textOf } from "../messages/message";
import { VIEW_CONTROLS, WHY_OFF, type ViewControl, type ViewFacts } from "../model/viewControls";

/** 视图控件可用性的统一计算处（声明表见 model/viewControls.ts，中文模板见消息目录 lab2shot/i18n/<lang>/messages/web.toml）。
 * 不可用的控件变灰并附原因，不隐藏。
 *
 * 工具条（editor/Viewer.tsx）、显示选项面板（ui/DisplayOptions.tsx）与三维视图（view/Stage3D.tsx）均由此取得结果，
 * 不各自将编号转换为文字：`WHY_OFF` 仅含编号，中文仅存在于消息目录。 */

/** 编号到中文的映射，模块加载时一次性生成的固定表，并非可增长的缓存。 */
const WHY_TEXT: Record<string, string> = Object.fromEntries(
  (Object.keys(WHY_OFF) as ViewControl[]).map((control) => [control, textOf(msg(WHY_OFF[control]))]),
);

/** 根据给定事实计算各控件的可用性，不可用的附带原因。 */
export const viewAvailable = (facts: ViewFacts): Availability => resolveLocal<ViewFacts>(VIEW_CONTROLS, facts, WHY_TEXT);
