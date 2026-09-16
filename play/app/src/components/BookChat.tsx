import { useEffect, useMemo, useRef, useState } from "react";
import type { BookCharacter, BookDoc, SandboxSession } from "../types";
import { MODELS } from "../api";
import { bookChat, bookCharacters } from "../bookstore";
import { detectMode, NeedKeyError, sandboxSave } from "../playllm";
import { parseSegments, CHAR_COLORS } from "../parse";
import { Segments } from "./Segments";
import { DoodleAvatar, hashStr } from "../doodle";
import { getLang, t } from "../i18n";

interface ChatMsg { speaker: "user" | "char"; text: string; sample?: boolean }
interface Sample { q: string; a: string; q_zh?: string; a_zh?: string }

function charColor(name: string): string {
  return CHAR_COLORS[hashStr(name) % CHAR_COLORS.length];
}

function errText(e: unknown): string {
  return e instanceof NeedKeyError ? t("needKey") : String(e);
}

/* Talk to a character of the book, frozen at the reader's position. */
export default function BookChat({ book, upto, pct, session, onClose }: {
  book: BookDoc;
  upto: number;
  pct: number;
  session?: SandboxSession;
  onClose: () => void;
}) {
  const [mode, setMode] = useState<"server" | "static" | "">("");
  const [chars, setChars] = useState<BookCharacter[] | null>(null);
  const [scanning, setScanning] = useState(false);
  const [scanFailed, setScanFailed] = useState(false);
  const [charName, setCharName] = useState(session?.role_code || "");
  const [userName, setUserName] = useState(session?.user_name || "");
  const [model, setModel] = useState(
    session?.model || localStorage.getItem("bpi.model") || MODELS[0]);
  const [msgs, setMsgs] = useState<ChatMsg[]>(
    (session?.messages as ChatMsg[]) || []);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [savedId, setSavedId] = useState(session?.id || "");
  const [samples, setSamples] = useState<Record<string, Sample[]>>({});
  const bodyRef = useRef<HTMLDivElement>(null);

  /* ready-made conversations for the bundled books: no key needed */
  useEffect(() => {
    setSamples({});
    fetch(`data/samples__${encodeURIComponent(book.bid)}.json`)
      .then((r) => (r.ok ? r.json() : {}))
      .then((o) => setSamples((o && typeof o === "object" ? o : {}) as Record<string, Sample[]>))
      .catch(() => {});
  }, [book.bid]);

  useEffect(() => { detectMode().then(setMode); }, []);

  /* cached cast only — scanning costs an LLM call, so it's a button */
  useEffect(() => {
    bookCharacters(book, model, true).then(setChars).catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [book.bid]);

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: 1e9, behavior: "smooth" });
  }, [msgs, busy]);

  const readSoFar = useMemo(
    () => book.text.slice(0, Math.max(600, upto)), [book, upto]);
  const present = useMemo(
    () => (chars || []).filter((c) => readSoFar.includes(c.name)),
    [chars, readSoFar]);
  const brief = useMemo(
    () => (chars || []).find((c) => c.name === charName)?.brief || "",
    [chars, charName]);
  const sampleChars = Object.keys(samples);
  const asked = new Set(msgs.filter((m) => m.speaker === "user").map((m) => m.text));
  const inLang = (s: Sample): Sample =>
    getLang() === "zh" && s.q_zh && s.a_zh ? { q: s.q_zh, a: s.a_zh } : { q: s.q, a: s.a };
  const sampleQs = (samples[charName] || []).map(inLang).filter((s) => !asked.has(s.q));

  const askSample = (s: Sample) => {
    if (busy) return;
    setErr("");
    const next: ChatMsg[] = [...msgs, { speaker: "user", text: s.q }];
    setMsgs(next);
    setBusy(true);
    window.setTimeout(() => {
      setMsgs([...next, { speaker: "char", text: s.a, sample: true }]);
      setBusy(false);
    }, 700);
  };

  const scan = async () => {
    setScanning(true);
    setScanFailed(false);
    setErr("");
    try {
      const cs = await bookCharacters(book, model, false);
      // empty result -> back to null so the scan button stays available
      setChars(cs && cs.length ? cs : null);
      if (!cs || cs.length === 0) setScanFailed(true);
    } catch (e) {
      setErr(errText(e));
      setScanFailed(true);
    } finally {
      setScanning(false);
    }
  };

  const send = async () => {
    const text = draft.trim();
    const name = charName.trim();
    if (!text || !name || busy) return;
    setErr("");
    const next: ChatMsg[] = [...msgs, { speaker: "user", text }];
    setMsgs(next);
    setDraft("");
    setBusy(true);
    try {
      const reply = await bookChat(book, {
        char_name: name, char_brief: brief, upto,
        history: next, user_name: userName, model,
      });
      setMsgs([...next, { speaker: "char", text: reply }]);
    } catch (e) {
      setErr(errText(e));
      setMsgs(next);
    } finally {
      setBusy(false);
    }
  };

  const save = async () => {
    const id = await sandboxSave({
      id: savedId || undefined,
      kind: "bookchat", sid: book.bid, tag: "book", upto,
      title: t("bookDraftTitle", charName, book.title, pct),
      role_code: charName, user_name: userName, model,
      messages: msgs,
    });
    setSavedId(id);
  };

  const color = charName ? charColor(charName) : "var(--npc)";

  return (
    <div className="play-overlay" onClick={(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div className="play-sheet">
        <div className="sandbox-mark">{t("playhouseMark")}</div>

        <div className="play-head">
          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
            <DoodleAvatar id={charName || "?"} size={42} color={color} />
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontFamily: "var(--font-zh)", fontSize: 20, letterSpacing: 2 }}>
                {charName ? t("playChatTitle") : t("whoToTalk")}
              </div>
              <div style={{ fontSize: 13.5, color: "var(--ink-mute)" }}>
                {t("spoilerNote", pct)}
              </div>
            </div>
            <button className="ink-btn small" onClick={onClose}>{t("close")}</button>
          </div>

          {/* cast picker */}
          <div style={{ display: "flex", gap: 6, marginTop: 12, flexWrap: "wrap", alignItems: "center" }}>
            {present.map((c) => (
              <button key={c.name} title={c.brief}
                className={"char-chip" + (c.name === charName ? " on" : "")}
                style={{ "--chip-ink": charColor(c.name) } as React.CSSProperties}
                onClick={() => setCharName(c.name)}>
                <DoodleAvatar id={c.name} size={26} color={charColor(c.name)} />
                {c.name}
              </button>
            ))}
            {chars === null && !scanning && (
              <button className="ink-btn small" onClick={scan}>
                {t("scanChars")}
              </button>
            )}
            {scanning && (
              <span style={{ color: "var(--ink-mute)", fontSize: 14 }}>
                {t("scanningChars")}
              </span>
            )}
            {scanFailed && !scanning && (
              <span style={{ color: "var(--ink-mute)", fontSize: 14 }}>
                {t("scanCharsFail")}
              </span>
            )}
            <input placeholder={t("freeCharName")} value={charName}
              style={{ width: 160, padding: "4px 10px", fontSize: 14.5 }}
              onChange={(e) => setCharName(e.target.value)} />
          </div>
          {sampleChars.length > 0 && (
            <div className="sample-row">
              <span>{t("sampleLead")}</span>
              {sampleChars.map((n) => (
                <button key={n}
                  className={"char-chip" + (n === charName ? " on" : "")}
                  style={{ "--chip-ink": charColor(n) } as React.CSSProperties}
                  onClick={() => setCharName(n)}>
                  <DoodleAvatar id={n} size={22} color={charColor(n)} />
                  {n}
                </button>
              ))}
            </div>
          )}

          <div style={{ display: "flex", gap: 8, marginTop: 10, flexWrap: "wrap" }}>
            <input placeholder={t("yourName")} value={userName}
              style={{ width: 150 }}
              onChange={(e) => setUserName(e.target.value)} />
            {mode === "server" && (
              <select value={model} onChange={(e) => {
                setModel(e.target.value);
                localStorage.setItem("bpi.model", e.target.value);
              }}>
                {MODELS.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            )}
            <button className="ink-btn small" onClick={save} disabled={!msgs.length}>
              {savedId ? t("savedDraft") : t("saveDraft")}
            </button>
          </div>
        </div>

        <div className="play-body" ref={bodyRef}>
          {msgs.length === 0 && (
            <div className="ambient" style={{ marginTop: 40 }}>
              {t("bookChatEmpty")}
            </div>
          )}
          {msgs.map((m, i) =>
            m.speaker === "user" ? (
              <div key={i} className="beat right">
                <div className="avatar-col">
                  <DoodleAvatar id={"visitor:" + (userName || "you")}
                    color="var(--ink-soft)" guest />
                  <div className="who" style={{ color: "var(--ink-soft)" }}>
                    {userName || t("you")}
                  </div>
                </div>
                <div className="bubbles">
                  <div className="bubble"
                    style={{ "--bubble-ink": "var(--ink-soft)" } as React.CSSProperties}>
                    {m.text}
                  </div>
                </div>
              </div>
            ) : (
              <div key={i} className="beat">
                <div className="avatar-col">
                  <DoodleAvatar id={charName} color={color} />
                  <div className="who" style={{ color }}>{charName}</div>
                </div>
                <div className="bubbles">
                  <Segments segs={parseSegments(m.text)} inkColor={color} />
                  {m.sample && <div className="sample-tag">{t("sampleTag")}</div>}
                </div>
              </div>
            ),
          )}
          {busy && (
            <div className="loading-ink" style={{ padding: "16px 0" }}>
              {t("penUp", charName)}
            </div>
          )}
          {sampleQs.length > 0 && !busy && (
            <div className="sample-qs">
              <div className="sample-qs-lead">{t("sampleAsk")}</div>
              {sampleQs.map((s) => (
                <button key={s.q} className="ink-btn small" onClick={() => askSample(s)}>
                  {s.q}
                </button>
              ))}
            </div>
          )}
          {err && (
            <div className="event-box"
              style={{ borderColor: "var(--seal)", outlineColor: "var(--seal)" }}>
              <span className="label" style={{ color: "var(--seal)" }}>
                {t("errLabel")}
              </span>{err}
            </div>
          )}
        </div>

        <div className="play-foot">
          <textarea value={draft}
            placeholder={charName ? t("chatPlaceholder") : t("freeCharName")}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
            }} />
          <button className="ink-btn seal" onClick={send}
            disabled={busy || !draft.trim() || !charName.trim()}>
            {t("send")}
          </button>
        </div>
      </div>
    </div>
  );
}
