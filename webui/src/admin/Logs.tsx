import { useCallback, useEffect, useRef, useState } from "react";
import { ByUser } from "./Resources";
import { api } from "../api";
import { copyText } from "../platform/util";
import { Section, useAdmin } from "./common";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";
import { t } from "../i18n/t";

const LINES = 800;

/** 日志 section: the server log — jobs, uploads, failed requests, errors with tracebacks, administrator changes,
 * and logs sent from user pages. Log size and verbosity are configured in 设置. */
export function LogsSection() {
  const { problem, go } = useAdmin();
  const read = useCallback(() => api.admin.log(LINES), []);
  const { data: log, reload: reload } = usePoll(read, null, { onError: (e) => problem(reasonOf(e)) });
  const [copied, setCopied] = useState(false);
  const box = useRef<HTMLPreElement>(null);
  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight });
  }, [log]);

  const copy = async () => {
    await copyText(log?.lines.join("\n") ?? "");
    setCopied(true);
  };

  return (
    <Section
      title={t("ui.admin.logs.title")}
      lede={t("ui.admin.logs.lede", { lines: LINES, file: log?.file ?? "work/logs/lab2shot.log" })}
      actions={
        <>
          <Button tone="ghost" onClick={() => go("settings-storage")}>
            {t("ui.admin.logs.settings")}
          </Button>
          <Button tone="ghost" onClick={() => void copy()}>
            {copied ? t("ui.admin.common.copied") : t("ui.admin.common.copy")}
          </Button>
          <Button tone="ghost" onClick={reload}>
            {t("ui.admin.common.refresh")}
          </Button>
        </>
      }
    >
      <pre className="server-log" ref={box}>
        {log ? log.lines.join("\n") || t("ui.admin.logs.empty") : t("ui.admin.common.reading")}
      </pre>
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="logs" />
    </Section>
  );
}
