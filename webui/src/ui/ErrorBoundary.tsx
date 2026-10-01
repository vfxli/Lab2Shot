import { Component, type ReactNode } from "react";
import { msg } from "../messages/message";
import { notePageError } from "../platform/pageErrors";
import { retryLoads } from "../platform/lazyRetry";
import { Button } from "./Button";
import { MessageText } from "./MessageText";

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
    notePageError("panel", error.message, this.props.name, error.stack ?? "", `「${this.props.name}」${error.message}`);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="panel-failed" role="alert">
        <div>
          <MessageText message={msg("E-PAGE-PANEL", { name: this.props.name, detail: this.state.error.message })} />
        </div>
        <Button tip="重新画这一块（节点图和参数都还在；按需载入没载进来的部件再载一次）" onClick={() => (retryLoads(), this.setState({ error: null }))}>
          重试
        </Button>
      </div>
    );
  }
}
