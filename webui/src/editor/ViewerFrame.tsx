import { useEffect, useState } from "react";
import { textOf } from "../messages/message";
import { IconEye } from "../ui/icons";
import { ItemsBar } from "../ui/ItemsBar";
import { Timeline } from "./Timeline";
import type { DisplayPlan } from "../view/plan";
import { useStageNotes, useViewerNote } from "../state/viewer";
import { ViewNotices, type Notice } from "../ui/ViewNotices";
import { ProjectNotice } from "./ProjectNotice";
import { useAppMode } from "./AppMode";

/** 视图外框（包裹 editor/Viewer.tsx 中的舞台），视图版面的唯一所在：上方为显示节点的名称与控制栏，中间为舞台，
 * 下方为光标读数、手柄提示与时间条。
 *
 * 关于当前画面的说明全部位于视图通知区（ui/ViewNotices.tsx），位于左上角，位置与顺序固定：
 * 视图的通知若放在节点旁边容易被忽略。
 *
 * 底部只保留非通知类内容：光标读数（随鼠标变化的测量值）与手柄提示（手柄用法说明）。
 *
 * 此处只负责排版，舞台上显示的内容由 Viewer.tsx 决定。胶囊只显示一句短文字（界面文字不换行、不截断），
 * 原因写在其悬停提示中。
 *
 * 上方的工具栏不显示悬停提示（data-no-tips，platform/tips.ts）；它打开的菜单照常显示。 */
export function ViewerFrame({
  plan,
  tools,
  body,
  hint,
  strip,
  stage2d,
  proxy,
  shown,
}: {
  plan?: DisplayPlan;
  tools?: React.ReactNode;
  body: React.ReactNode;
  hint?: string | null;
  strip?: React.ReactNode;
  stage2d?: boolean;
  shown?: string | null; // 视图当前显示的节点：它所在的逐项处理块显示条目栏
  proxy?: Notice[] | null; // 点云代理显示（倍数由 view/kinds3d.ts useCloudProxy 计算，文字在 Viewer.tsx 生成）
}) {
  // 二维只有视图代理一种画质，没有可切换的档位，也就没有角标或下拉；查看原始数据需下载交付后在 DCC 中查看。
  const did = useTransientNote();
  // 视图通知区的全部内容集中计算于此（ViewNotices 只负责绘制与排序），画面上的每一条说明均在此汇总：
  // 舞台写入 state 的条目（本机画面、相机、显示的错误、三维手柄提示）以及本层自身掌握的信息。
  const staged = useStageNotes((s) => s.notes);
  const notices: Notice[] = [
    ...Object.entries(staged)
      .filter(([, v]) => !!v)
      .map(([kind, v]) => ({ kind: kind as Notice["kind"], key: kind, text: v!.text, tip: v!.tip })),
    ...(hint ? [{ kind: "hint" as const, key: "hint2d", text: hint, tip: hint }] : []),
    ...(did ? [did] : []),
    ...(proxy ?? []),
  ];
  const appMode = useAppMode((m) => m.mode === "app");
  return (
    <div className="viewer">
      {/* 画面上方的独立工具栏，与视图同宽（参照 Nuke）：不得绝对定位浮于画面上，否则会遮挡画面右上角。 */}
      <div className="view-bar" data-no-tips>
        <div className="view-bar-name">
          <IconEye size={12} color="var(--accent)" />
          <span data-user-data>{plan ? plan.node.data.label : "未选择显示节点"}</span>
        </div>
        <div className="hud-tools">{tools}</div>
      </div>
      <div className="view-stage">
        {body}
        <ViewNotices notices={notices} />
        <ItemsBar nodeId={shown ?? null} />
        {/* 舞台上不设第二处文字：手柄用法说明同样进入左上角的统一通知区 */}
        {/* 应用模式收起了节点图，节点图左下角的研究出处说明改在舞台左下角（同一个组件，与三维坐标轴错开） */}
        {appMode && <ProjectNotice className="view-notice" />}
      </div>
      {strip}
      <Timeline stage2d={!!stage2d} />
    </div>
  );
}

/** 视图刚刚自动执行的操作（state/viewTools.ts 的 useViewerNote），4 秒后自动消失，是通知区中唯一会自动消失的一类：
 * 「已自动切换模式」一类的提示过一段时间即失去意义，而「当前显示的并非数据本身」一类的提示必须持续显示。 */
function useTransientNote(): Notice | null {
  const note = useViewerNote((s) => s.note);
  const why = useViewerNote((s) => s.why);
  const n = useViewerNote((s) => s.n);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (!n) return;
    setShown(n);
    const t = window.setTimeout(() => setShown(0), 4000);
    return () => window.clearTimeout(t);
  }, [n]);
  if (!note || shown !== n) return null;
  return { kind: "did", key: `did-${n}`, text: textOf(note), tip: textOf(why) || textOf(note) };
}
