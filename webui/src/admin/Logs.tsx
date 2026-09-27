import { useCallback, useEffect, useRef, useState } from "react";
import { ByUser } from "./Resources";
import { api } from "../api";
import { copyText } from "../platform/util";
import { Section, useAdmin } from "./common";
import { usePoll } from "../platform/poll";
import { reasonOf } from "../messages/message";
import { Button } from "../ui/Button";

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
      title="服务日志"
      lede={`最近 ${LINES} 行，最新的在最后：任务、上传、出错的请求、程序异常、管理员的操作，还有用户从网页「日志」里发来的内容。完整文件在服务器的 ${log?.file ?? "work/logs/lab2shot.log"}。`}
      actions={
        <>
          <Button tip="日志多大换新文件、留几份、要不要详细日志，在「设置」的「存储与清理」里改" tone="ghost" onClick={() => go("settings")}>
            日志设置
          </Button>
          <Button tip="把显示的日志复制到剪贴板" tone="ghost" onClick={() => void copy()}>
            {copied ? "已复制" : "复制"}
          </Button>
          <Button tip="重新读取日志" tone="ghost" onClick={reload}>
            刷新
          </Button>
        </>
      }
    >
      <pre className="server-log" ref={box}>
        {log ? log.lines.join("\n") || "还没有记录" : "读取中…"}
      </pre>
      {/* Filter by user: the same table and listing function as the 用户 detail page. */}
      <ByUser section="logs" />
    </Section>
  );
}
