/** Placeholder shown by the 3D stage (view/Stage3D.tsx) while the server is still preparing the view data.
 * The view buttons (视角, 框显) live on the toolbar, not in this component. */

import { useEffect, useState } from "react";

export function Preparing() {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    const t = setInterval(() => setSeconds((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="empty" data-tip="服务器正在从结果里读出要显示的模型、角色、点云和相机；读完先显示当前帧，其余的帧陆续补齐">
      <div>
        正在准备三维显示数据…{seconds >= 2 ? ` ${seconds} 秒` : ""}
        {seconds >= 8 && <div className="empty-hint">大的场景第一次显示要把整个文件读一遍，最多两分钟；超过会停下并说明原因</div>}
      </div>
    </div>
  );
}
