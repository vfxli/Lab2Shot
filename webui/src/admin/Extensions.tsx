import { useCallback, useState } from "react";
import { api, type ExtensionRow, type ManualItem, type ManualView } from "../api";
import { usable } from "../api/applies";
import { taskLive } from "../api/tasks";
import { useSignedIn } from "../state/session";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Section, useAdmin } from "./common";
import { InstallControl } from "../install/Install";
import { InboxNote, ManualRowControl, useManualView } from "../install/Manual";
import { Table, type Column } from "../ui/Table";
import { Button } from "../ui/Button";
import { Loading } from "../ui/Loading";
import { t } from "../i18n/t";
import { tipAttrs, tipOf } from "../platform/tips";

/** 扩展包：各第三方项目的安装状态、缺失项及安装操作（lab2shot/installer，GET /api/admin/extensions）。
 *
 * 安装仅限管理员（本区读写的路由均要求 installs.run），因此整个功能位于后台。
 *
 * 安装控件本身为 install/Install.tsx，手动下载为 install/Manual.tsx。
 * 本区只负责布局：一张表加手动下载区块。
 *
 * 能否安装、卸载、回退均由服务器计算（server/available.py extension → 每行的 `actions`），此处不做角色判断。 */
export function ExtensionsSection() {
  const { problem } = useAdmin();
  const state = useSignedIn();
  const [version, setVersion] = useState(0);
  const read = useCallback(() => api.installs.extensions(), []);
  const [busy, setBusy] = useState(false);
  // 有安装任务时每三秒刷新一次（该行的状态和步骤条随安装任务更新）；无任务时只读取一次，不轮询：
  // 每次读取都要检查所有扩展的环境、权重记录和自检记录。
  const { data } = usePoll(read, busy ? 3000 : null, { key: version, onError: (e) => problem(reasonOf(e)) });
  const rows = data?.extensions ?? [];
  const installing = rows.some((e) => taskLive(e.job));
  if (installing !== busy) setBusy(installing); // 在渲染过程中更新自身状态：React 会立即重新渲染，不经过副作用。
  const can = { manual: usable(state?.applies, "help.manual"), consent: usable(state?.applies, "help.consent") };
  const manualView = useManualView(version, can.manual);
  // 「再看一遍」：重新读取扩展状态，并重新整理收件文件夹（可识别的文件会被自动安装）。
  const recheck = useCallback(() => setVersion((v) => v + 1), []);

  const ready = rows.filter((e) => e.ready).length;
  // 需要处理的排在前面（有缺失项 → 安装中 → 已就绪），同档按名称排序。管理员查看本区主要是为了确认待办事项，
  // 已就绪的项目排在前面会妨碍查找。
  const sorted = [...rows].sort((a, b) => Number(a.ready) - Number(b.ready) || a.title.localeCompare(b.title));
  const columns: Column<ExtensionRow>[] = [
    {
      id: "title",
      label: t("ui.admin.extensions.col_project"),
      className: "ext-proj",
      cell: (e) => (
        <>
          <span className="ext-title">{e.title}</span>
          {/* 状态、节点、安装三列为定宽，本列占据剩余宽度，仅显示名称会留下大片空白。
              `reason` 在「已就绪」时为空字符串，因此不可用时显示缺失原因，可用时显示项目简介
              （`summary`，摘自上游原文）。两种情况下均不为空。 */}
          <span className="ext-why">{e.reason || e.summary}</span>
        </>
      ),
    },
    {
      id: "state",
      label: t("ui.admin.extensions.col_state"),
      width: "11rem",
      cell: (e) => (
        <span className={`chip${e.ready ? " ok" : e.installed ? " warn" : ""}`}>
          {e.label}
        </span>
      ),
    },
    { id: "nodes", label: t("ui.admin.extensions.col_nodes"), width: "5rem", className: "tnum", cell: (e) => e.nodes || "—" },
    {
      id: "install",
      label: t("ui.admin.extensions.col_install"),
      width: "21rem",
      cell: (e) => <InstallControl p={e} onChange={recheck} full />,
    },
  ];

  return (
    <Section
      title={t("ui.admin.nav.extensions")}
      lede={
        <>
          {t("ui.admin.extensions.lede")}
          {data && t("ui.admin.extensions.count", { all: rows.length, ready })}
        </>
      }
      actions={
        <Button tone="ghost" onClick={recheck}>
          {t("ui.admin.common.refresh")}
        </Button>
      }
    >
      <Table
        rows={sorted}
        columns={columns}
        rowKey={(e) => e.name}
        empty={data ? t("ui.admin.extensions.empty") : <Loading what={t("ui.admin.nav.extensions")} />}
      />
      <ManualBlock view={manualView} can={can} onChange={recheck} />
    </Section>
  );
}

const MANUAL_STATE: Record<ManualItem["state"], () => string> = {
  ready: () => t("ui.admin.extensions.manual_ready"),
  consent: () => t("ui.admin.extensions.manual_consent"),
  unrecognised: () => t("ui.admin.extensions.manual_unrecognised"),
  missing: () => t("ui.admin.extensions.manual_missing"),
};

/** 手动下载：安装流程的一部分。SMPL-X、FLAME、Autodesk FBX SDK 等需使用者自行从官网下载（部分还须本人
 * 同意许可协议），体检清单中对应两项（lab2shot/installer/preflight.py 的 B-INSTALL-MANUAL / B-INSTALL-CONSENT）
 * 不满足时无法安装，因此与安装放在同一区，而非单独成页。
 * 当前登录无法查看收件文件夹时（help.manual 不可用），整个区块不显示。 */
function ManualBlock({ view, can, onChange }: {
  view: ManualView | null;
  can: { manual: boolean; consent: boolean };
  onChange: () => void;
}) {
  const columns: Column<ManualItem>[] = [
    {
      id: "title",
      label: t("ui.admin.extensions.col_file"),
      className: "ext-proj",
      cell: (m) => (
        <>
          <span className="ext-title">{m.title}</span>
          <span className="ext-why">{m.what}</span>
        </>
      ),
    },
    {
      id: "state",
      label: t("ui.admin.extensions.col_state"),
      width: "11rem",
      cell: (m) => <span className={`chip${m.state === "ready" ? " ok" : m.state === "missing" ? "" : " warn"}`}>{MANUAL_STATE[m.state]()}</span>,
    },
    {
      id: "needed",
      label: t("ui.admin.extensions.col_needed"),
      width: "8rem",
      cell: (m) => (
        <span {...tipAttrs(tipOf("value", m.needed_by.map((w) => w.title).join(t("list.sep")) || undefined))}>
          {m.needed_by.length ? t("ui.admin.extensions.needed_count", { n: m.needed_by.length }) : "—"}
        </span>
      ),
    },
    {
      id: "act",
      label: t("ui.admin.extensions.col_action"),
      width: "21rem",
      cell: (m) => <ManualRowControl manualKey={m.key} downloadPage={m.page} view={view} can={can} onChange={onChange} />,
    },
  ];
  if (!view) return null;
  return (
    <div className="ext-manual">
      <h3>{t("ui.admin.extensions.manual")}</h3>
      <InboxNote view={view} />
      <Table rows={view.items} columns={columns} rowKey={(m) => m.key} empty={t("ui.admin.extensions.manual_none")} />
    </div>
  );
}
