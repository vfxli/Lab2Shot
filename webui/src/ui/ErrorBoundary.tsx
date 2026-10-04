import { Component, type ReactNode } from "react";
import { msg } from "../messages/message";
import { notePageError } from "../platform/pageErrors";
import { PageOutdated, retryLoads } from "../platform/lazyRetry";
import { Button } from "./Button";
import { MessageText } from "./MessageText";
import { t } from "../i18n/t";
import { tipOf } from "../platform/tips";

/** Keeps one broken part from blanking the page: every page root and every editor panel
 * sits in one. It says which part failed and what to do, offers a retry, and resets on its own when `resetKey` changes
 * (another node shown). The error goes to the page's one error catcher (platform/pageErrors.ts): the log and 提交反馈
 * see it, counted when it comes again. */
export class ErrorBoundary extends Component<{ name: string; resetKey?: unknown; children: ReactNode }, { error: Error | null; key: unknown }> {
  state = { error: null as Error | null, key: this.props.resetKey };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  static getDerivedStateFromProps(props: { resetKey?: unknown }, state: { error: Error | null; key: unknown }) {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null;
  }

  componentDidCatch(error: Error) {
    if (error instanceof PageOutdated) {
      notePageError("outdated", error.message, this.props.name, error.stack ?? "", msg("W-PAGE-OUTDATED").text);
      return;
    }
    notePageError("panel", error.message, this.props.name, error.stack ?? "", t("ui.misc.panel_error", { name: this.props.name, message: error.message }));
  }

  render() {
    if (!this.state.error) return this.props.children;
    if (this.state.error instanceof PageOutdated)
      return (
        <div className="panel-failed" role="alert">
          <div>
            <MessageText message={msg("W-PAGE-OUTDATED")} />
          </div>
          <Button onClick={() => window.location.reload()}>{t("ui.common.refresh")}</Button>
        </div>
      );
    return (
      <div className="panel-failed" role="alert">
        <div>
          <MessageText message={msg("E-PAGE-PANEL", { name: this.props.name, detail: this.state.error.message })} />
        </div>
        <Button tip={tipOf("consequence", t("ui.misc.panel_retry_tip"))} onClick={() => (retryLoads(), this.setState({ error: null }))}>
          {t("ui.common.retry")}
        </Button>
      </div>
    );
  }
}
