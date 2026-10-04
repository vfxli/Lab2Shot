import { useState } from "react";
import { api } from "../api";
import { Button } from "../ui/Button";
import { setParams } from "../graph/edit";
import { say, msg, messageOf } from "../state/say";
import { readClipboard } from "../platform/util";
import { useCookInputs } from "../state/cookInputs";
import { useWriteLock } from "../ui/writeLock";
import type { NodeTypeDef } from "../api/catalog";
import { t } from "../i18n/t";

/** 从别的软件粘贴一组参数（3DE / Nuke 镜头数据）：本模块负责「粘贴参数」按钮与剪贴板读不到时的手动粘贴框。
 *
 * 节点声明了 `paste`（是哪个软件）才画这个按钮；一次改完 = 一步撤销——粘进来的一整颗镜头十几个数，
 * 撤销一次全回去，不是撤十几次。
 *
 * 这里不认识任何软件、任何格式：文字原样交给服务器，服务器交给节点类，节点自己读
 * （lab2shot/nodes/core/lens_distortion.py read_pasted）。读不出来时节点说的那句话原样显示。
 *
 * 剪贴板读不到的情况（http:// 从别的机器打开、浏览器拒绝了权限）不是失败：换成一个文本框让人自己粘，
 * 不是弹一句「读不到剪贴板」就把人扔在那。 */

const APPS: Record<string, string> = { nuke: "Nuke" };

export function PasteFrom({ nodeId, def }: { nodeId: string; def: NodeTypeDef }) {
  const [typing, setTyping] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const readOnly = !!useWriteLock(); // 写不了：粘贴会改参数，不给
  if (!def.paste || readOnly) return null;
  const app = APPS[def.paste] ?? def.paste;

  const read = (text: string) => {
    if (!text.trim()) return;
    setBusy(true);
    const graph = useCookInputs.getState().graphId; // 结果属于这张图的节点，而非期间打开的另一张图中的同名节点
    void api.paste(def.id, text).then(
      (got) => {
        setBusy(false);
        setTyping(null);
        if (useCookInputs.getState().graphId !== graph) return;
        setParams(nodeId, got);
        say(msg("N-WEB-PASTED", { app, count: Object.keys(got).length }), nodeId);
        // 画面宽高是这张表里粘不进来的两个数（LD_3DE4 的旋钮里没有），而畸变的坐标要靠它们。
        // 粘贴时即说明，不等到计算时才发现；接了「图像」就跟画面走，那时不用说
        const ci = useCookInputs.getState();
        const n = ci.nodes[nodeId];
        const hasImage = Object.values(ci.edges).some((e) => e.target === nodeId && e.targetHandle === "image");
        if (n && !hasImage && !n.params.width && !n.params.height) {
          say(msg("N-WEB-PASTENORASTER", { app }), nodeId);
        }
      },
      // 读不出来时，节点说的那句话原样显示（带它自己的编号）：页面不重写一遍理由
      (e) => {
        setBusy(false);
        say(messageOf(e), nodeId);
      },
    );
  };

  if (typing !== null) {
    return (
      <div className="paste-from">
        <textarea
          className="field" rows={4} autoFocus value={typing} data-field="paste-text"
          aria-label={t("ui.params.paste.text", { app })}
          placeholder={t("ui.params.paste.placeholder", { app })}
          onChange={(e) => setTyping(e.target.value)}
        />
        <div className="paste-row">
          <Button size="sm" disabled={busy || !typing.trim()} onClick={() => read(typing)}>{t("ui.params.paste.read")}</Button>
          <Button size="sm" tone="ghost" onClick={() => setTyping(null)}>{t("ui.common.cancel")}</Button>
        </div>
      </div>
    );
  }

  return (
    <div className="paste-from">
      <Button size="sm" disabled={busy} data-field="paste-from"
        onClick={() => void readClipboard().then((text) => (text.trim() ? read(text) : setTyping("")))}>
        {t("ui.params.paste.from", { app })}
      </Button>
    </div>
  );
}
