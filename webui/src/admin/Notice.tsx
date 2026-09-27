import { useEffect, useState } from "react";
import { adminApi } from "../api/admin";
import { Banner, type BannerTone } from "../ui/Banner";
import { Button, Segmented, Switch } from "../ui/Button";
import { reasonOf } from "../messages/message";
import { shown, usable, why } from "../api/applies";
import { useSignedIn } from "../state/session";
import { whenText } from "../platform/format";

/** 管理员通知 section: a single line shown at the top of the editor, the help page and the admin page — plain text,
 * a colour indicating its severity, and an on/off switch. It is server state (lab2shot/server/notice.py, meta
 * server.notice); every page picks up changes through its existing poll of the server state, so nothing is pushed.
 *
 * Red is not offered: it is reserved for conditions that can cause a production incident. Whether this card is
 * shown is decided by the server (settings.notice through available.py), not by a role check here. */

const CHARS = 200; // Must match lab2shot/server/notice.py NOTICE_CHARS.

const TONES: { value: BannerTone; label: string; tip: string }[] = [
  { value: "info", label: "信息", tip: "蓝色：一般的告知，比如「今晚 22 点重启服务」" },
  { value: "notice", label: "提醒", tip: "蓝色：要人留意但不影响使用的事" },
  { value: "warn", label: "警告", tip: "橙色：会影响使用的事，比如「测试阶段，请勿上传项目正式素材」" },
];

export function NoticeCard() {
  const state = useSignedIn();
  const may = shown(state?.applies, "settings.notice");
  const [text, setText] = useState("");
  const [tone, setTone] = useState<BannerTone>("info");
  const [on, setOn] = useState(false);
  const [saved, setSaved] = useState<{ text: string; tone: string; on: boolean; updated: number; by?: string } | null>(null);
  const [problem, setProblem] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!may) return;
    let live = true;
    adminApi.notice().then(
      (n) => live && (setSaved(n), setText(n.text), setTone(n.tone as BannerTone), setOn(n.on)),
      (e: Error) => live && setProblem(reasonOf(e)),
    );
    return () => {
      live = false;
    };
  }, [may]);

  if (!may) return null;
  const left = CHARS - [...text].length;
  // Input the server would reject is reported here first, so a visible error never leaves the page.
  const blocked = left < 0 ? `通知最多 ${CHARS} 个字，现在多了 ${-left} 个` : on && !text.trim() ? "打开了通知，但没有写内容" : "";
  const dirty = !saved || saved.text !== text.trim() || saved.tone !== tone || saved.on !== on;

  const save = async () => {
    setSaving(true);
    try {
      const n = await adminApi.setNotice({ text: text.trim(), tone, on });
      setSaved(n);
      setText(n.text);
      setProblem("");
    } catch (e) {
      setProblem(reasonOf(e as Error));
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="set-card" data-group="notice">
      <h3>管理员通知</h3>
      <p className="adm-lede">
        一行字，出现在编辑器、帮助页和这个页面的顶部，普通用户关不掉。纯文本，不认 HTML。红色留给可能造成生产事故的提醒，这里不提供。
      </p>
      <div className="set-row" data-key="notice.text">
        <span className="set-label" data-tip={`通知的内容，最多 ${CHARS} 个字；写清是什么事、什么时候、要做什么`}>
          文字
        </span>
        <span className="set-ctl" data-tip={`通知的内容，最多 ${CHARS} 个字；写清是什么事、什么时候、要做什么`}>
          <textarea
            className="field notice-text"
            value={text}
            rows={2}
            aria-label="通知的文字"
            data-tip={`纯文本，最多 ${CHARS} 个字；写清是什么事、什么时候、要做什么`}
            placeholder="例：今晚 22:00 重启服务升级扩展包，排队的任务会等算完再重启。"
            onChange={(e) => setText(e.target.value)}
          />
        </span>
        <span className="set-tags">
          <span className={`chip tnum${left < 0 ? " set-over" : ""}`} data-tip={`还能写 ${left} 个字`}>
            {left}
          </span>
        </span>
      </div>
      <div className="set-row" data-key="notice.tone">
        <span className="set-label" data-tip="通知条的颜色：按这条通知要人怎么对待它来选">
          颜色
        </span>
        <span className="set-ctl" data-tip="通知条的颜色：按这条通知要人怎么对待它来选">
          <Segmented label="通知的颜色" value={tone} options={TONES.map((t) => ({ value: t.value, label: t.label, tip: t.tip }))} onChange={(v) => setTone(v)} />
        </span>
      </div>
      <div className="set-row" data-key="notice.on">
        <span className="set-label" data-tip="关掉以后文字还留着，下次打开就又是它">
          显示
        </span>
        <span className="set-ctl" data-tip="关掉以后文字还留着，下次打开就又是它">
          <Switch on={on} label="显示通知" tip={on ? "关掉：所有页面顶部不再显示这条通知，文字留着" : "打开：所有页面顶部显示这条通知"} onChange={setOn} />
        </span>
        <span className="set-tags">
          {saved?.updated ? (
            <span className="dim" data-tip={`${saved.by ?? "管理员"} 在 ${whenText(saved.updated)} 改的`}>
              {saved.by ?? "管理员"} · {whenText(saved.updated)}
            </span>
          ) : (
            <span className="dim">还没有设过</span>
          )}
        </span>
      </div>
      <div className="set-row notice-preview">
        <span className="set-label" data-tip="通知条在三处页面顶部的样子">
          效果
        </span>
        <span className="set-ctl">
          {text.trim() ? (
            <Banner tone={tone} aside="管理员通知" tip="用户看到的就是这个样子">
              {text.trim()}
            </Banner>
          ) : (
            <span className="dim">写了文字才看得到效果</span>
          )}
        </span>
      </div>
      {(problem || blocked) && <div className="set-why bad">{problem || blocked}</div>}
      <div className="set-row">
        <span className="set-label" />
        <span className="set-ctl">
          {/* Not the page's primary (filled) button: 设置 already has one, and a page has at most one. */}
          <Button
            tip={blocked || why(state?.applies, "settings.notice") || "保存并立刻生效：所有打开着的页面在下一次查服务器状态时换上"}
            disabled={saving || !dirty || !!blocked || !usable(state?.applies, "settings.notice")}
            onClick={() => void save()}
          >
            {saving ? "保存中…" : dirty ? "保存通知" : "已保存"}
          </Button>
        </span>
      </div>
    </div>
  );
}
