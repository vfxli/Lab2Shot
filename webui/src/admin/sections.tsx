import type { SettingsPageEntry } from "../api";
import { shown } from "../api/applies";
import { useSignedIn } from "../state/session";
import type { AdminContext } from "./common";
import { PasswordCard } from "./Auth";
import { CardsSection } from "./Cards";
import { DatabaseSection } from "./Database";
import { DiskSection } from "./Disk";
import { ExtensionsSection } from "./Extensions";
import { FeedbackSection } from "./Feedback";
import { InvitesPart } from "./Invites";
import { NoticeCard } from "./Notice";
import { TermsAdminCard } from "./Terms";
import { LogsSection } from "./Logs";
import { OverviewSection } from "./Overview";
import { QueueSection } from "./QueueSection";
import { ResidentSection } from "./Resident";
import { SecuritySection } from "./Security";
import { SettingsPage } from "./Settings";
import { UsageSection } from "./Usage";
import { UsersSection } from "./Users";

/** The admin page's sections, in the order of its side list: one line each. A section is a component of its own
 * (it reads what it needs; useAdmin() for what the page shares); `badge` puts a count or a mark next to its name.
 *
 * `group` bands the side list (人员, 数据, 系统, 设置); the sections before the first named group are the
 * machine's own running state and carry no heading. The 设置 band at the bottom is the settings pages, as the server
 * defines them (lab2shot/config.py PAGES, sent with the login state): sectionsFor adds them. A section is shown only
 * when the server says this login has it (server/available.py SECTIONS, read through applies.ts) — the list itself
 * never looks at a role. */
export interface AdminSection {
  id: string; // its address: /admin#<id>
  label: string; // short, one line, no brackets
  tip: string;
  Component: () => React.ReactNode;
  badge?: (ctx: AdminContext) => string | null;
  group?: string; // the band it opens ("" or absent: it goes on with the band before it)
}

const SECTIONS: AdminSection[] = [
  // the machine itself, first and unheaded: what is running right now
  { id: "overview", label: "概览", tip: "显卡、队列、内存、硬盘、常驻模型和服务本身一眼看全，还有要注意的事，和今天、最近的登录、注册、任务、流量、反馈", Component: OverviewSection },
  {
    id: "queue",
    label: "队列",
    tip: "所有人的任务、谁提交的、取消任务，和全部任务记录",
    Component: QueueSection,
    badge: (c) => {
      const n = c.queue?.jobs.filter((j) => j.state === "running" || j.state === "queued").length ?? 0;
      return n ? String(n) : null;
    },
  },
  { id: "cards", label: "显卡", tip: "每张卡的型号、显存、接不接任务、在跑什么，扩展包能不能在它上面跑；改授权之前先看后果", Component: CardsSection },
  { id: "resident", label: "常驻模型", tip: "算完留在进程里的模型：在哪张显卡、占多少显存和内存，卸载到内存或完全卸载", Component: ResidentSection },

  // 人员
  { id: "users", group: "人员", label: "用户", tip: "账号：给朋友新建账号，改到期时间、环节和能用哪些节点，停用、重设密码、删除；点用户名看他名下的一切", Component: UsersSection },
  {
    id: "feedback",
    label: "用户反馈",
    tip: "用户点「提交反馈」发来的问题和建议：写的内容、截图、查问题用的资料；标记已看、已解决，下载，删除",
    Component: FeedbackSection,
    badge: (c) => (c.overview?.feedback_new ? String(c.overview.feedback_new) : null),
  },
  { id: "usage", label: "使用统计", tip: "每个三方项目和节点、每个环节、每个账号用了多少，看哪些项目有用、谁在用", Component: UsageSection },

  // 数据。后台没有「模板」页：模板的管理全在编辑器的「模板」弹窗里（editor/Templates.tsx，管模板的账号
  // 多出那些操作）。装扩展只有管理员做得了（这一区的路由一律要 installs.run），所以扩展包在后台，
  // 见 admin/Extensions.tsx
  {
    id: "extensions",
    group: "数据",
    label: "扩展包",
    tip: "每个第三方项目装了没有、缺什么模型、装它；还有要人手动下载的文件（SMPL-X、FBX SDK…）放哪、同意许可协议",
    Component: ExtensionsSection,
  },
  { id: "disk", label: "硬盘", tip: "缓存、上传的素材、待取回的结果各占多少，清理没用的", Component: DiskSection },
  { id: "database", label: "数据库", tip: "任务记录和统计存放的数据库：大小、完整性、备份、立即备份、怎么恢复", Component: DatabaseSection },

  // 系统
  {
    id: "security",
    group: "系统",
    label: "安全",
    tip: "有几个登录；看起来像在试探的请求（输错密码、没开放的接口……）和被暂时封住的来源",
    Component: SecuritySection,
  },
  { id: "logs", label: "日志", tip: "服务日志：任务、出错、管理员的操作、用户发来的网页日志", Component: LogsSection },
];

/** 账号设置's cards besides its settings: the 管理员通知 (its own right, settings.notice: NoticeCard decides) and
 * the administrator's password, which goes with reading the settings. */
function AccountCards() {
  const state = useSignedIn();
  return (
    <>
      <NoticeCard />
      {shown(state?.applies, "settings.values") && <PasswordCard />}
    </>
  );
}

/** What a settings page holds besides its settings: the parts with a right of their own (server/available.py
 * _PARTS_OF_PAGE declares the same ones), and a mark of its own next to its name. 注册设置: the 用户协议与隐私政策
 * (terms.edit: TermsAdminCard decides) among its cards, the invite codes below them. */
const PAGE_PARTS: Record<string, { cards?: React.ReactNode; below?: React.ReactNode; badge?: (ctx: AdminContext) => boolean }> = {
  register: { cards: <TermsAdminCard />, below: <InvitesPart />, badge: (c) => !!c.overview?.registering.paused },
  accounts: { cards: <AccountCards /> },
};

/** Every section, in the side list's order: the fixed ones, then the 设置 band, one section per settings page. A
 * page is marked when a setting on it waits for a restart (or, 注册设置, when registering paused itself). */
export function sectionsFor(pages: readonly SettingsPageEntry[]): AdminSection[] {
  const settings = pages.map((page, i): AdminSection => {
    const parts = PAGE_PARTS[page.id] ?? {};
    return {
      id: `settings-${page.id}`,
      group: i === 0 ? "设置" : undefined,
      label: page.label,
      tip: page.tip,
      Component: () => (
        <SettingsPage page={page} cards={parts.cards}>
          {parts.below}
        </SettingsPage>
      ),
      badge: (c) => (c.overview?.pending.some((p) => p.page === page.id) || parts.badge?.(c) ? "!" : null),
    };
  });
  return [...SECTIONS, ...settings];
}

/** The band each section sits in, carried down the list from the one that opens it: the heading follows the section,
 * not its position, so hiding the first section of a band still shows the band over the next one it may see. */
export function bandsOf(sections: readonly AdminSection[]): Record<string, string> {
  const out: Record<string, string> = {};
  let band = "";
  for (const s of sections) out[s.id] = band = s.group ?? band;
  return out;
}
