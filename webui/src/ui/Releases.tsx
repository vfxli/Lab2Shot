import "./releases.css";
import { useEffect, useState } from "react";
import { readReleases, type Lines, type ReleaseNotes } from "../api/releases";
import { readLocal, webAddress, writeLocal } from "../platform/util";
import { reasonOf } from "../messages/message";
import { Button } from "./Button";
import { Empty } from "./Empty";
import { Loading } from "./Loading";
import { Sheet } from "./Sheet";
import { pick, t } from "../i18n/t";
import { getLang } from "../i18n/lang";
import { tipOf } from "../platform/tips";

// The newest version this browser has shown the dialog for (its Chinese name, the same whatever language the page
// speaks: names are unique, lab2shot/releases.py). Kept
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
  const first = notes?.releases[0];
  const newest = first ? pick(first.name, "zh") : null;
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
      <Button tip={unseen ? tipOf("value", t("ui.misc.releases_unseen")) : undefined} tone="ghost" layout="rn-button" onClick={show}>
        {t("ui.misc.releases")}
        {unseen && <span className="rn-dot" aria-label={t("ui.misc.releases_new")} />}
      </Button>
      {open && (
        <Sheet title={t("ui.misc.releases")} width={600} onClose={() => setOpen(false)}>
          {notes ? <ReleaseList notes={notes} /> : error ? <Empty title={t("ui.misc.releases_failed")} hint={error} /> : <Loading what={t("ui.misc.releases_loading")} />}
        </Sheet>
      )}
    </>
  );
}

/** A version's lines in the page's language (the other one when it has none). */
function lines(v: Lines | undefined): string[] {
  const lang = getLang();
  return v?.[lang]?.length ? v[lang]! : v?.zh?.length ? v.zh : v?.en ?? [];
}

/** The project's address first, then each version: 名称 and 日期, 说明, and its lines. Everything is text: React
 * escapes it, and the address is a link only when it is an https one (the server checks the same). */
function ReleaseList({ notes }: { notes: ReleaseNotes }) {
  return (
    <div className="rn-list">
      {webAddress(notes.project) && (
        <a className="chip link rn-project" href={notes.project} target="_blank" rel="noopener noreferrer">
          {notes.project}
        </a>
      )}
      {notes.releases.map((r) => (
        <article key={pick(r.name, "zh")} className="rn-card">
          <div className="rn-head">
            <span className="rn-name">{pick(r.name)}</span>
            <span className="rn-date tnum">{r.date}</span>
          </div>
          <p className="rn-about">{pick(r.about)}</p>
          {lines(r.changes).length > 0 && (
            <ul className="rn-changes">
              {lines(r.changes).map((line, i) => (
                <li key={i}>{line}</li>
              ))}
            </ul>
          )}
          {lines(r.admin).length > 0 && (
            <>
              <p className="rn-admin">{t("ui.misc.releases_admin")}</p>
              <ul className="rn-changes">
                {lines(r.admin).map((line, i) => (
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
