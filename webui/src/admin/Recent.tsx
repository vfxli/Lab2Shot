import { adminApi, type RecentPeriod, type RecentView } from "../api/admin";
import { countText, sizeText } from "../platform/format";
import { usePoll } from "../platform/poll";
import { onlineTip, useAdmin } from "./common";

/** 概览's 「今天和最近」: what happened today and lately, one card per subject (访问, 注册, 任务, 流量, 反馈), each opening
 * its section like the machine tiles above. Every card is the same row of columns — 现在, 今日, 近 7 天 (本周 for
 * 注册), 本月 — so a period's numbers stand under each other from card to card; a card leaves a column it has no
 * number for empty. The server sends only the groups this login may open the section of (server/available.py
 * RECENT), and counts by its own local time (lab2shot/periods.py). Read more slowly than the tiles: these are counts
 * of a day, not what runs now. */

const EVERY = 30_000;

const PERIOD: Record<RecentPeriod, string> = { today: "今日", days7: "近 7 天", week: "本周", month: "本月" };

// how each period is counted, for the tooltips (the server's local time; lab2shot/periods.py)
const SPAN: Record<RecentPeriod, string> = {
  today: "今天零点起",
  days7: "今天和之前 6 天",
  week: "本周一零点起",
  month: "本月 1 号零点起",
};

interface Cell {
  label: string;
  value: string;
  sub?: string;
  tip: string;
}

interface Group {
  id: string;
  title: string;
  page: string; // the section it opens, by name
  section: string;
  now?: Cell;
  periods: [RecentPeriod, Cell][]; // 今日, then 近 7 天 or 本周, then 本月
}

const PAGE: Record<string, string> = { security: "安全", users: "用户", queue: "队列", feedback: "用户反馈" };

function groups(r: RecentView): Group[] {
  const out: Group[] = [];
  const group = (id: string, title: string, section: string, periods: Group["periods"], now?: Cell) =>
    out.push({ id, title, section, page: PAGE[section] ?? section, periods, now });
  const span = (p: RecentPeriod) => `${PERIOD[p]}（${SPAN[p]}，按服务器的时间）`;

  if (r.access) {
    const a = r.access;
    group("access", "访问", a.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: `${PERIOD[p]}登录`,
      value: `${countText(a[p].people)} 人`,
      sub: `登录 ${countText(a[p].logins)} 次 · 失败 ${countText(a[p].failed)} 次`,
      tip: `${span(p)}登录成功过的账号，同一个账号登录多次算一个人；浏览器和插件（DCC、命令行）的登录都算，管理员自己的也算。\n` +
        "登录次数：成功的登录。\n" +
        "失败次数：没登录上的尝试（密码不对、账号停用或过期、试得太频繁被拒），不管输入的用户名是不是真有这个账号；同一个用户名只留最近 1000 次失败。",
    }]), {
      label: "在线",
      value: `${countText(a.online.count)} 人`,
      sub: `浏览器 ${a.online.browser} · 插件 ${a.online.client}`,
      tip: onlineTip(a.online),
    });
  }
  if (r.accounts) {
    const u = r.accounts;
    group("accounts", "注册", u.section, (["today", "week", "month"] as const).map((p) => [p, {
      label: `${PERIOD[p]}新账号`,
      value: `${countText(u[p].all)} 个`,
      sub: `自己注册 ${countText(u[p].self)} · 管理员建 ${countText(u[p].all - u[p].self)}`,
      tip: `${span(p)}新建的账号：自己注册的，和管理员在「用户」里或用命令行建的；建了之后删掉的照样算，永久删除了的不算。`,
    }]));
  }
  if (r.tasks) {
    const t = r.tasks;
    group("tasks", "任务", t.section, (["today", "days7", "month"] as const).map((p) => {
      const c = t[p];
      const parts = [`成功 ${countText(c.done)}`, ...(c.partial ? [`部分 ${countText(c.partial)}`] : []), `失败 ${countText(c.failed)}`, `取消 ${countText(c.cancelled)}`];
      return [p, {
        label: `${PERIOD[p]}任务`,
        value: `${countText(c.all)} 个`,
        sub: parts.join(" · "),
        tip: `${span(p)}提交的任务，所有人的，读文件这样的小任务也算；按现在的结果分：成功、部分（部分成功：有的分支出错，其余算完了）、失败、取消。` +
          "还在排队和计算的只算进总数（现在有几个，看上面的「队列」）。删掉了的任务不再算，任务文件夹到期清理掉的照样算。",
      }];
    }));
  }
  if (r.traffic) {
    const t = r.traffic;
    group("traffic", "流量", t.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: `${PERIOD[p]}流量`,
      value: sizeText(t[p]),
      tip: `${span(p)}服务器发给所有账号的字节：取帧、三维数据、下载、接口的回答都算，按实际发出的（压缩后）算，没登录的请求不算。` +
        "每个账号用了多少，在「用户」里看。",
    }]));
  }
  if (r.feedback) {
    const f = r.feedback;
    group("feedback", "反馈", f.section, (["today", "days7", "month"] as const).map((p) => [p, {
      label: `${PERIOD[p]}新反馈`,
      value: `${countText(f[p].came)} 条`,
      tip: `${span(p)}用户发来的反馈，不管现在处理到哪一步；删掉了的不算。`,
    }]), {
      label: "未解决",
      value: `${countText(f.open)} 条`,
      sub: `其中 ${countText(f.new)} 条还没看`,
      tip: "还没标成「已解决」的反馈（「新」和「已看」），不管哪天发来的；「还没看」是其中还是「新」的。",
    });
  }
  return out;
}

function CellView({ cell }: { cell?: Cell }) {
  if (!cell) return <span className="rc-cell" />;
  return (
    <span className="rc-cell" data-tip={cell.tip}>
      <span className="adm-tile-label">{cell.label}</span>
      <b className="tnum">{cell.value}</b>
      {cell.sub && (
        // each "label N" part stays whole: a narrow window breaks the line between parts, never inside one
        <span className="adm-tile-sub rc-sub">
          {cell.sub.split(" · ").map((part) => (
            <span key={part}>{part}</span>
          ))}
        </span>
      )}
    </span>
  );
}

export function RecentPart() {
  const { go } = useAdmin();
  const { data } = usePoll(adminApi.recent, EVERY);
  if (!data) return null;
  const shown = groups(data);
  if (!shown.length) return null;
  return (
    <div className="rc">
      <h3>今天和最近</h3>
      <p className="adm-lede">
        按服务器的时间：今日从零点起，近 7 天是今天和之前 6 天，本周从周一起，本月从 1 号起。每 30 秒更新一次；停在数字上看它数的是什么，点一张进到对应的页面。
      </p>
      <div className="rc-cards">
        {shown.map((g) => (
          <button key={g.id} className="rc-card" data-tip={`打开「${g.page}」`} onClick={() => go(g.section)}>
            <span className="rc-title">
              <b>{g.title}</b>
              <span className="adm-tile-sub">{g.page}</span>
            </span>
            <CellView cell={g.now} />
            {g.periods.map(([p, c]) => (
              <CellView key={p} cell={c} />
            ))}
          </button>
        ))}
      </div>
    </div>
  );
}
