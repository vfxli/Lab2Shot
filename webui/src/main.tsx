import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Gate } from "./gate";
import { catchPageErrors } from "./platform/pageErrors";
import { installTips } from "./platform/tips";
import "./ui/tokens.css";
import "./styles/00-base.css";
import "./ui/field.css";
import "./ui/forms.css"; // 公共表单行（.who-row）多处使用，全站加载，不跟着某个组件走
import "./ui/tip.css";
import "./ui/glass.css";
import "./styles/login.css";

// The page's entry is only the gate (gate.tsx): the one thing anyone gets before the access code. The rest of the page
// (site.tsx: the editor, help, the admin pages) is another file the server hands out only after it (server/access.py).
catchPageErrors(); // before anything else, so an error on the login page is caught too
installTips();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <Gate />
  </StrictMode>,
);
