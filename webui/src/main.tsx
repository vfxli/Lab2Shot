// the kit's base styles first, so what a component brings with it (ui/menu.css's fixed .menu, which is also .glass)
// comes after them and wins where both say something
import "./ui/tokens.css";
import "./styles/00-base.css";
import "./ui/field.css";
import "./ui/forms.css"; // the shared form row (.who-row) is used in many places: loaded site-wide, not with any one component
import "./ui/glass.css";
import "./ui/tip.css"; // after glass: the tip is also .glass, and its own position (fixed) must win
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { Gate } from "./gate";
import { catchPageErrors } from "./platform/pageErrors";
import { installTips } from "./platform/tips";
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
