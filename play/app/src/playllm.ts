/* LLM plumbing, dual-mode:
   - "server": the local python server proxies via animask/llm.py (.env creds)
   - "static": no backend — the browser calls an OpenAI-compatible API
     directly with the visitor's own key (BYOK, stored in localStorage only) */
import type { SandboxSession } from "./types";

/* ------------------------------------------------------------ mode probe */
let modeP: Promise<"server" | "static"> | null = null;
export function detectMode(): Promise<"server" | "static"> {
  if (!modeP) {
    modeP = fetch("/api/ping")
      .then((r): "server" | "static" => (r.ok ? "server" : "static"))
      .catch((): "server" | "static" => "static");
  }
  return modeP;
}

/* ------------------------------------------------------------------ BYOK */
export interface LLMConfig { base: string; key: string; model: string }

export function getLLMConfig(): LLMConfig {
  try {
    return { base: "", key: "", model: "", ...JSON.parse(localStorage.getItem("bpi.llm") || "{}") };
  } catch { return { base: "", key: "", model: "" }; }
}
export function setLLMConfig(c: LLMConfig) {
  localStorage.setItem("bpi.llm", JSON.stringify(c));
}

export class NeedKeyError extends Error {}

export async function directQuery(prompt: string, context: string): Promise<string> {
  const cfg = getLLMConfig();
  if (!cfg.base || !cfg.key || !cfg.model) throw new NeedKeyError("no key");
  const content = context ? `<context>\n${context}\n</context>\n\n${prompt}` : prompt;
  const r = await fetch(cfg.base.replace(/\/+$/, "") + "/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.key}` },
    body: JSON.stringify({
      model: cfg.model, temperature: 0, max_tokens: 4096,
      messages: [{ role: "user", content }],
    }),
  });
  if (!r.ok) throw new Error(`LLM API ${r.status}: ${(await r.text()).slice(0, 200)}`);
  const d = await r.json();
  const msg = d?.choices?.[0]?.message?.content;
  if (!msg) throw new Error("empty LLM response");
  return String(msg).trim();
}

/* ------------------------------------------------------- dual-mode calls */
async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const o = await r.json();
  if (o && o.error) throw new Error(o.error);
  return o as T;
}

/* --------------------------------------------- dual-mode saved chats */
function lsSessions(): SandboxSession[] {
  try { return JSON.parse(localStorage.getItem("bpi.sandbox") || "[]"); }
  catch { return []; }
}

export async function sandboxSave(s: SandboxSession): Promise<string> {
  if ((await detectMode()) === "server") {
    const r = await post<{ id: string }>("/api/sandbox/save", s);
    return r.id;
  }
  const all = lsSessions();
  const id = s.id || Math.random().toString(36).slice(2, 14);
  const entry = {
    ...s, id,
    saved_at: new Date().toISOString().slice(0, 16).replace("T", " "),
    n_messages: (s.messages || []).length,
  };
  const i = all.findIndex((x) => x.id === id);
  if (i >= 0) all[i] = entry; else all.unshift(entry);
  localStorage.setItem("bpi.sandbox", JSON.stringify(all.slice(0, 60)));
  return id;
}

export async function sandboxList(): Promise<SandboxSession[]> {
  if ((await detectMode()) === "server") {
    const r = await fetch("/api/sandbox/list").then((x) => x.json());
    return r.sessions || [];
  }
  return lsSessions();
}

export async function sandboxGet(id: string): Promise<SandboxSession | null> {
  if ((await detectMode()) === "server") {
    const r = await fetch(`/api/sandbox/${id}`).then((x) => x.json());
    return r && !r.error ? r : null;
  }
  return lsSessions().find((x) => x.id === id) || null;
}
