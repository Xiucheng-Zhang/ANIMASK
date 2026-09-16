import { useEffect, useState } from "react";
import ReadShelf from "./pages/ReadShelf";
import Reader from "./pages/Reader";
import { LangToggle, t, useLang } from "./i18n";
import { detectMode, getLLMConfig, setLLMConfig, LLMConfig } from "./playllm";

function useHashRoute(): string {
  const [hash, setHash] = useState(window.location.hash);
  useEffect(() => {
    const fn = () => setHash(window.location.hash);
    window.addEventListener("hashchange", fn);
    return () => window.removeEventListener("hashchange", fn);
  }, []);
  return hash.replace(/^#/, "");
}

export function nav(to: string) {
  window.location.hash = to;
}

export default function App() {
  useLang(); // re-render the whole tree on language switch
  const route = useHashRoute();
  const parts = route.split("/").filter(Boolean);
  const [mode, setMode] = useState<"server" | "static" | "">("");
  const [showKeys, setShowKeys] = useState(false);

  useEffect(() => { detectMode().then(setMode); }, []);

  let page: JSX.Element;
  if (parts[0] === "read" && parts[1]) {
    page = <Reader raw={decodeURIComponent(parts.slice(1).join("/"))} />;
  } else {
    page = <ReadShelf />;
  }

  return (
    <>
      <header className="top">
        <div className="top-in">
          <a className="brand" href="../">ANIMASK</a>
          <span className="page-name" onClick={() => nav("/")} style={{ cursor: "pointer" }}>
            {t("pageName")}
          </span>
          <span className="top-right">
            {mode === "static" && (
              <button className="ink-btn small" onClick={() => setShowKeys(true)}>
                🗝 {t("settings")}
              </button>
            )}
            <LangToggle />
          </span>
        </div>
      </header>
      <div className="page">
        {page}
        {showKeys && <KeySettings onClose={() => setShowKeys(false)} />}
      </div>
    </>
  );
}

/* BYOK settings — only shown on the static (public) build */
function KeySettings({ onClose }: { onClose: () => void }) {
  const [cfg, setCfg] = useState<LLMConfig>(getLLMConfig());
  const set = (k: keyof LLMConfig) =>
    (e: React.ChangeEvent<HTMLInputElement>) => setCfg({ ...cfg, [k]: e.target.value });
  return (
    <div className="play-overlay" onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div className="play-sheet" style={{ width: "min(460px, 96vw)" }}>
        <div className="play-head">
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ fontFamily: "var(--font-zh)", fontSize: 20, letterSpacing: 2, flex: 1 }}>
              🗝 {t("settingsTitle")}
            </div>
            <button className="ink-btn small" onClick={onClose}>{t("close")}</button>
          </div>
        </div>
        <div className="play-body">
          <p style={{ fontSize: 14, color: "var(--ink-soft)" }}>{t("settingsNote")}</p>
          <label style={{ display: "block", margin: "12px 0 4px", fontSize: 14 }}>
            {t("settingsBase")}
          </label>
          <input style={{ width: "100%" }} value={cfg.base}
            placeholder="https://api.openai.com" onChange={set("base")} />
          <label style={{ display: "block", margin: "12px 0 4px", fontSize: 14 }}>
            {t("settingsKey")}
          </label>
          <input style={{ width: "100%" }} type="password" value={cfg.key}
            placeholder="sk-…" onChange={set("key")} />
          <label style={{ display: "block", margin: "12px 0 4px", fontSize: 14 }}>
            {t("settingsModel")}
          </label>
          <input style={{ width: "100%" }} value={cfg.model}
            placeholder="gpt-5.5" onChange={set("model")} />
          <div style={{ display: "flex", gap: 10, marginTop: 18 }}>
            <button className="ink-btn primary"
              onClick={() => { setLLMConfig(cfg); onClose(); }}>
              {t("settingsSave")}
            </button>
            <button className="ink-btn"
              onClick={() => {
                const empty = { base: "", key: "", model: "" };
                setCfg(empty); setLLMConfig(empty);
              }}>
              {t("settingsClear")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
