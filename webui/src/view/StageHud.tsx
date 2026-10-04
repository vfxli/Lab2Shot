/** Placeholder shown by the 3D stage (view/Stage3D.tsx) while the server is still preparing the view data.
 * The view buttons (视角, 框显) live on the toolbar, not in this component. */

import { useEffect, useState } from "react";
import { t } from "../i18n/t";

export function Preparing() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const tick = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(tick);
  }, []);
  return (
    <div className="empty">
      <div>
        {seconds >= 2 ? t("ui.view.preparing_3d_seconds", { seconds }) : t("ui.view.preparing_3d")}
        {seconds >= 8 && <div className="empty-hint">{t("ui.view.preparing_3d_hint")}</div>}
      </div>
    </div>
  );
}
