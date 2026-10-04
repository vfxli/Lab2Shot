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
import { t } from "../i18n/t";

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
  label: () => string; // short, one line, no brackets (asked while rendering: in the page's language)
  Component: () => React.ReactNode;
  badge?: (ctx: AdminContext) => string | null;
  group?: Band; // the band it opens (absent: it goes on with the band before it)
}

type Band = "people" | "data" | "system" | "settings";

/** A band's heading on the side list. */
export function bandLabel(band: string): string {
  const words: Record<Band, () => string> = {
    people: () => t("ui.admin.nav.people"),
    data: () => t("ui.admin.nav.data"),
    system: () => t("ui.admin.nav.system"),
    settings: () => t("ui.admin.nav.settings"),
  };
  return words[band as Band]?.() ?? band;
}

const SECTIONS: AdminSection[] = [
  // the machine itself, first and unheaded: what is running right now
  { id: "overview", label: () => t("ui.admin.nav.overview"), Component: OverviewSection },
  {
    id: "queue",
    label: () => t("ui.admin.nav.queue"),
    Component: QueueSection,
    badge: (c) => {
      const n = c.queue?.jobs.filter((j) => j.state === "running" || j.state === "queued").length ?? 0;
      return n ? String(n) : null;
    },
  },
  { id: "cards", label: () => t("ui.admin.nav.cards"), Component: CardsSection },
  { id: "resident", label: () => t("ui.admin.nav.resident"), Component: ResidentSection },

  // 人员
  { id: "users", group: "people", label: () => t("ui.admin.nav.users"), Component: UsersSection },
  {
    id: "feedback",
    label: () => t("ui.admin.nav.feedback"),
    Component: FeedbackSection,
    badge: (c) => (c.overview?.feedback_new ? String(c.overview.feedback_new) : null),
  },
  { id: "usage", label: () => t("ui.admin.nav.usage"), Component: UsageSection },

  // 数据. The admin page has no templates section: templates are managed entirely in the editor's 模板 dialog
  // (editor/Templates.tsx; an account that manages templates gets the extra actions there). Installing extensions is
  // the administrator's alone (every route of that section needs installs.run), so 扩展包 lives here; see
  // admin/Extensions.tsx
  {
    id: "extensions",
    group: "data",
    label: () => t("ui.admin.nav.extensions"),
    Component: ExtensionsSection,
  },
  { id: "disk", label: () => t("ui.admin.nav.disk"), Component: DiskSection },
  { id: "database", label: () => t("ui.admin.nav.database"), Component: DatabaseSection },

  // 系统
  {
    id: "security",
    group: "system",
    label: () => t("ui.admin.nav.security"),
    Component: SecuritySection,
  },
  { id: "logs", label: () => t("ui.admin.nav.logs"), Component: LogsSection },
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
      group: i === 0 ? "settings" : undefined,
      label: () => page.label,
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
