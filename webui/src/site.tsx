// what every page of the site shares (each page brings its own rules with it, and a kit component brings its own)
import { lazyRetry } from "./platform/lazyRetry";
import "./styles/index.css";
import "./ui/glow.css"; // the decorative lights of the kit (the hover glow, the entry rim): not on the login page
import { Suspense } from "react";
import { AdminGate } from "./admin/Auth";
import { SiteNotices } from "./ui/Banner";
import { onPageError } from "./platform/pageErrors";
import { ErrorBoundary } from "./ui/ErrorBoundary";
import { Loading } from "./ui/Loading";
import { msg, say } from "./state/say";
import { useSessionWatch } from "./state/session";
import { addCatalogue } from "./messages/format";
import { CATALOGUE } from "./messages/generatedCatalogue";

/** The page behind the login (the gate loads it once the browser is logged in): /admin is the administrator's page
 * (the account menu opens it for an administrator); everything else is the node editor. The admin page is a file of its own that the server hands out only to the administrator
 * (server/access.py): it loads after the admin login. There is no help page; extension installation lives on the
 * admin page. */

const App = lazyRetry(() => import("./editor/App")); // the editor (and, when a 3D result is shown, three.js) loads with its page, not with the login
const AdminPage = lazyRetry(() => import("./admin/AdminApp"));

// the message templates served only after login (the gate carries just the few it uses): registered before any page
// renders
addCatalogue(CATALOGUE);

// every error the page catches goes into its log once (counted when it comes again): a part an ErrorBoundary caught by
// its name, anything else as a page error
onPageError((e, detail) => say(e.kind === "panel" ? msg("E-PAGE-PANEL", { name: e.where, detail: e.message }) : msg("E-PAGE-ERROR", { detail })));


/** One page of the site: its own error boundary (a failure in it never blanks the page's frame) and a loading line while
 * its code arrives. */
function Part({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <ErrorBoundary name={name}>
      <Suspense fallback={<Loading what={name} fill />}>{children}</Suspense>
    </ErrorBoundary>
  );
}

export default function Page() {
  useSessionWatch(); // the login every page behind the gate shows (state/session.ts)
  const page = window.location.pathname;
  if (page.startsWith("/admin"))
    return (
      <AdminGate what="管理页面" page="page.admin">
        {/* the administrator's notice is on every page; the server's own state is the admin page's own business —
            it is the page that restarts it (admin/Restart.tsx). The banner sits at the top of the page in normal
            document flow, never floating over the content: floating, it covers the view and reads as part of it. */}
        <SiteNotices restart={false} />
        <Part name="管理页面">
          <AdminPage />
        </Part>
      </AdminGate>
    );
  // every page but the admin page (which restarts the server and says so itself) notes a restart in progress
  return (
    <>
      {/* the topmost strip of the whole page, in normal document flow */}
      <SiteNotices />
      <Part name="编辑器">
        <App />
      </Part>
    </>
  );
}
