import "./plugins.css";
import { useState } from "react";
import { shown } from "../api/applies";
import { readPlugins, type DccPlugin } from "../api/plugins";
import { useSession } from "../state/session";
import { reasonOf } from "../messages/message";
import { Button, ButtonLink } from "./Button";
import { Empty } from "./Empty";
import { Loading } from "./Loading";
import { Sheet } from "./Sheet";
import { t } from "../i18n/t";

/** 「DCC 插件」 in the top bar, right of 「更新说明」: the DCCs and their plugins (lab2shot/server/plugins.py). A DCC whose
 * plugin this server does not ship yet says so; one it ships has a download, zipped by the server from its source
 * at that moment (the plugin follows this repository, no version of its own). The list is read when the dialog opens.
 * Shown only to a login the server lets download them (subject "plugins.download": everyone while the setting
 * plugins.download is on, else administrators only, whose dialog says the plugins are not open to users). */
export function PluginsButton() {
  const applies = useSession((st) => st.state)?.applies;
  const [open, setOpen] = useState(false);
  const [list, setList] = useState<{ plugins: DccPlugin[]; open: boolean } | null>(null);
  const [error, setError] = useState("");
  const show = () => {
    setOpen(true);
    setError("");
    readPlugins().then(setList, (e: unknown) => setError(reasonOf(e)));
  };
  if (!shown(applies, "plugins.download")) return null;
  return (
    <>
      <Button tone="ghost" onClick={show}>
        {t("ui.misc.plugins")}
      </Button>
      {open && (
        <Sheet title={t("ui.misc.plugins")} width={460} onClose={() => setOpen(false)}>
          {list ? <PluginList list={list.plugins} closed={!list.open} /> : error ? <Empty title={t("ui.misc.plugins_failed")} hint={error} /> : <Loading what={t("ui.misc.plugins_loading")} />}
        </Sheet>
      )}
    </>
  );
}

function PluginList({ list, closed }: { list: DccPlugin[]; closed: boolean }) {
  return (
    <div className="dcc-list">
      {closed && (
        <div className="dcc-closed">
          <span className="dcc-closed-tag">{t("ui.misc.plugins_closed")}</span>
          <span>{t("ui.misc.plugins_closed_note")}</span>
        </div>
      )}
      {list.map((p) => (
        <div key={p.id} className="dcc-row">
          <span className="dcc-name">{p.label}</span>
          {p.available ? (
            <ButtonLink tone="primary" href={p.url} download>
              {t("ui.common.download")}
            </ButtonLink>
          ) : (
            <span className="dcc-none">{t("ui.misc.plugins_none")}</span>
          )}
        </div>
      ))}
      <p className="dcc-note">{t("ui.misc.plugins_note")}</p>
    </div>
  );
}

