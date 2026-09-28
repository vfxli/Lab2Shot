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
      label: "项目",
      tip: "第三方项目的官方名字；下面一行：还不能用就写差什么，能用就写它是个什么项目",
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
      label: "状态",
      tip: "「已就绪」才能算：装好了、模型齐了、自检过了。别的状态鼠标停上去看差什么",
      width: "11rem",
      cell: (e) => (
        <span className={`chip${e.ready ? " ok" : e.installed ? " warn" : ""}`} data-tip={e.reason || `${e.title} 装好了，它的节点现在能用`}>
          {e.label}
        </span>
      ),
    },
    { id: "nodes", label: "节点", tip: "它给编辑器添了几个节点", width: "5rem", className: "tnum", cell: (e) => e.nodes || "—" },
    {
      id: "install",
      label: "安装",
      tip: "安装、补全、重新安装、回退到上一个环境、卸载；装着的时候这里是步骤条",
      width: "21rem",
      cell: (e) => <InstallControl p={e} onChange={recheck} full />,
    },
  ];

  return (
    <Section
      title="扩展包"
      lede={
        <>
          每个第三方项目自己一套环境（third_party/&lt;名字&gt;/），装好、模型齐了、自检过了才算「已就绪」，它的节点在编辑器里才能用。
          {data && ` 共 ${rows.length} 个，已就绪 ${ready} 个。`}
        </>
      }
      actions={
        <Button tip="重新读一遍每个扩展包的状态，顺便整理收件文件夹里新放进去的文件" tone="ghost" onClick={recheck}>
          刷新
        </Button>
      }
    >
      <Table
        rows={sorted}
        columns={columns}
        rowKey={(e) => e.name}
        empty={data ? "这台机器上一个扩展包都没有：adapters/ 里放进项目之后，这里列出来" : <Loading what="扩展包" />}
      />
      <ManualBlock view={manualView} can={can} onChange={recheck} />
    </Section>
  );
}

const MANUAL_STATE: Record<ManualItem["state"], string> = {
  ready: "已就位",
  consent: "等同意许可协议",
  unrecognised: "认不出来",
  missing: "还没下载",
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
      label: "文件",
      tip: "要手动下载的那一样东西的官方名字，下面一行是它是什么",
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
      label: "状态",
      tip: "「已就位」才算数；「等同意许可协议」要本人点一下，「认不出来」在上面那一块写着原因",
      width: "11rem",
      cell: (m) => <span className={`chip${m.state === "ready" ? " ok" : m.state === "missing" ? "" : " warn"}`}>{MANUAL_STATE[m.state]}</span>,
    },
    {
      id: "needed",
      label: "谁要它",
      tip: "哪些扩展包缺了它就装不了",
      width: "8rem",
      cell: (m) => (
        <span data-tip={m.needed_by.map((w) => w.title).join("、") || "还没有扩展包用到它"}>
          {m.needed_by.length ? `${m.needed_by.length} 个扩展包` : "—"}
        </span>
      ),
    },
    {
      id: "act",
      label: "操作",
      tip: "去官网下载、看许可协议原文并同意、放好文件后再查一遍",
      width: "21rem",
      cell: (m) => <ManualRowControl manualKey={m.key} downloadPage={m.page} view={view} can={can} onChange={onChange} />,
    },
  ];
  if (!view) return null;
  return (
    <div className="ext-manual">
      <h3>手动下载</h3>
      <InboxNote view={view} />
      <Table rows={view.items} columns={columns} rowKey={(m) => m.key} empty="没有需要手动下载的文件" />
    </div>
  );
}
