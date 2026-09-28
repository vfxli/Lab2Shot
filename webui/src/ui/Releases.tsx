import "./releases.css";
import { useEffect, useState } from "react";
import { readReleases, type ReleaseNotes } from "../api/releases";
import { readLocal, webAddress, writeLocal } from "../platform/util";
import { reasonOf } from "../messages/message";
import { Button } from "./Button";
import { Empty } from "./Empty";
import { Loading } from "./Loading";
import { Sheet } from "./Sheet";

// The newest version this browser has shown the dialog for (its name: names are unique, lab2shot/releases.py). Kept
// per browser, not per account: nothing about reading the notes goes to the server. Without storage (a private window,
// storage blocked) the dot simply shows again next time.
const SEEN = "lab2shot.releases.seen";

/** 「更新说明」 in the top bar: what changed in each version (CHANGELOG.toml, served by lab2shot/server/releases.py).
 * A dot marks it while the newest version has not been opened in this browser; opening the dialog clears it. The notes
 * are read once when the page loads, which is also when a new version arrives (an update restarts the server and the
 * page reloads). */
export function ReleasesButton() {
  const [notes, setNotes] = useState<ReleaseNotes | null>(null);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [seen, setSeen] = useState(() => readLocal(SEEN));
  const load = () => {
    setError("");
    readReleases().then(setNotes, (e: unknown) => setError(reasonOf(e)));
  };
  useEffect(load, []);
  const newest = notes?.releases[0]?.name ?? null;
  const unseen = newest !== null && newest !== seen;
  const show = () => {
    setOpen(true);
    if (!notes) load(); // the first reading failed: try again now that someone asks
  };
  useEffect(() => {
    if (!open || newest === null) return;
    writeLocal(SEEN, newest);
    setSeen(newest);
  }, [open, newest]);
  return (
    <>
      <Button tip={`更新说明：每个版本改了什么，最新的在最上面${unseen ? "\n有新版本的更新还没看过" : ""}`} tone="ghost" layout="rn-button" onClick={show}>
        更新说明
        {unseen && <span className="rn-dot" aria-label="有新版本" />}
      </Button>
      {open && (
        <Sheet title="更新说明" width={600} onClose={() => setOpen(false)}>
          {notes ? <ReleaseList notes={notes} /> : error ? <Empty title="更新说明没有取到" hint={error} /> : <Loading what="更新说明" />}
        </Sheet>
      )}
    </>
  );
}

/** The project's address first, then each version: 名称 and 日期, 说明, and its lines. Everything is text: React
 * escapes it, and the address is a link only when it is an https one (the server checks the same). */
function ReleaseList({ notes }: { notes: ReleaseNotes }) {
  return (
    <div className="rn-list">
      {webAddress(notes.project) && (
        <a className="chip link rn-project" href={notes.project} target="_blank" rel="noopener noreferrer" data-tip="在新标签页打开项目主页">
          {notes.project}
        </a>
      )}
      {notes.releases.map((r) => (
        <article key={r.name} className="rn-card">
          <div className="rn-head">
            <span className="rn-name">{r.name}</span>
            <span className="rn-date tnum">{r.date}</span>
          </div>
          <p className="rn-about">{r.about}</p>
          {r.changes.length > 0 && (
            <ul className="rn-changes">
              {r.changes.map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          )}
          {!!r.admin?.length && (
            <>
              <p className="rn-admin">管理员</p>
              <ul className="rn-changes">
                {r.admin.map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ul>
            </>
          )}
        </article>
      ))}
    </div>
  );
}
