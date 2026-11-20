import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import LegacyApp from "./App";
import LitteringApp from "./LitteringApp";
import "./styles.css";
import "./littering.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {new URLSearchParams(window.location.search).get("mode") === "rodent"
      ? <LegacyApp />
      : <LitteringApp />}
  </StrictMode>,
);
