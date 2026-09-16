import { useEffect, useRef, useState } from "react";
import type { BookMeta, SandboxSession } from "../types";
import {
  addBook, getPos, listBooks, MAX_BOOK_MB, removeBook,
} from "../bookstore";
import { detectMode, sandboxList } from "../playllm";
import { nav } from "../App";
import { DoodleAvatar, hashStr, InkDivider } from "../doodle";
import { CHAR_COLORS } from "../parse";
import { t } from "../i18n";

interface Draft { title: string; author: string; text: string }

export default function ReadShelf() {
  const [books, setBooks] = useState<BookMeta[] | null>(null);
  const [chats, setChats] = useState<SandboxSession[]>([]);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [err, setErr] = useState("");
  const [mode, setMode] = useState<"server" | "static" | "">("");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { detectMode().then(setMode); }, []);

  const reload = () => {
    listBooks().then(setBooks).catch((e) => setErr(String(e)));
    sandboxList()
      .then((ss) => setChats(ss.filter((s) => s.kind === "bookchat")))
      .catch(() => {});
  };
  useEffect(reload, []);

  const onFile = async (f: File | undefined) => {
    if (!f) return;
    setErr("");
    if (f.size > MAX_BOOK_MB * 1_000_000) {
      setErr(t("uploadFailQuota"));
      return;
    }
    const text = await f.text();
    const title = f.name.replace(/\.txt$/i, "").replace(/[_-]+/g, " ").trim();
    setDraft({ title: title || "untitled", author: "", text });
  };

  const shelve = async () => {
    if (!draft) return;
    try {
      const bid = await addBook(draft.title.trim() || "untitled",
        draft.author.trim(), draft.text);
      setDraft(null);
      nav(`/read/${bid}`);
    } catch (e) {
      setErr(e instanceof DOMException && e.name === "QuotaExceededError"
        ? t("uploadFailQuota") : String(e));
    }
  };

  const remove = async (b: BookMeta) => {
    if (!window.confirm(t("confirmRemove", b.title))) return;
    await removeBook(b.bid);
    reload();
  };

  if (err && !books) return <div className="loading-ink">{t("errPrefix")} · {err}</div>;
  if (!books) return <div className="loading-ink">{t("loadingShelf")}</div>;

  const mine = books.filter((b) => b.source === "upload");
  const house = books.filter((b) => b.source === "builtin");

  return (
    <div className="fade-in">
      {/* ------------------------------------------------ my books ---- */}
      <div className="section-title">
        {t("myShelf")} <span className="en">{t("myShelfSub")}</span>
        <span style={{ flex: 1 }} />
        <button className="ink-btn" onClick={() => fileRef.current?.click()}>
          {t("addBookBtn")}
        </button>
      </div>
      <div style={{ color: "var(--ink-mute)", fontSize: 14, margin: "-6px 0 10px" }}>
        {mode === "server" ? t("addBookHintServer") : t("addBookHintStatic")}
      </div>
      <input ref={fileRef} type="file" accept=".txt,text/plain"
        style={{ display: "none" }}
        onChange={(e) => {
          onFile(e.target.files?.[0]);
          e.target.value = "";
        }} />
      {err && (
        <div className="event-box" style={{ borderColor: "var(--seal)", outlineColor: "var(--seal)", margin: "10px 0" }}>
          <span className="label" style={{ color: "var(--seal)" }}>{t("errLabel")}</span>{err}
        </div>
      )}

      {mine.length === 0 && !draft && (
        <div className="ambient" style={{ margin: "18px 0 8px", textAlign: "left" }}>
          {t("emptyMyShelf")}
        </div>
      )}
      <div className="shelf" style={{ marginTop: 14 }}>
        {mine.map((b, i) => (
          <ReadCover key={b.bid} book={b} index={i} onRemove={() => remove(b)} />
        ))}
      </div>

      {/* ------------------------------------------------ house books - */}
      <div className="section-title" style={{ marginTop: 44 }}>
        {t("houseShelf")} <span className="en">{t("houseShelfSub")}</span>
      </div>
      <div className="shelf" style={{ marginTop: 14 }}>
        {house.map((b, i) => (
          <ReadCover key={b.bid} book={b} index={i + mine.length} />
        ))}
      </div>

      {/* ------------------------------------------- saved book chats - */}
      {chats.length > 0 && (
        <>
          <div className="section-title" style={{ marginTop: 46 }}>
            {t("bookChatDrafts")} <span className="en">{t("bookChatDraftsSub")}</span>
          </div>
          <div className="wobble-card alt" style={{ maxWidth: 720 }}>
            <table className="tbl">
              <tbody>
                {chats.slice(0, 12).map((s) => (
                  <tr key={s.id} className="click"
                    onClick={() => nav(`/read/${s.sid}?chat=${s.id}`)}>
                    <td style={{ whiteSpace: "nowrap", color: "var(--indigo)" }}>
                      {t("kindChat")}
                    </td>
                    <td>{s.title || s.sid}</td>
                    <td style={{ whiteSpace: "nowrap", color: "var(--ink-mute)" }}>
                      {s.saved_at}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {draft && (
        <UploadSheet draft={draft} setDraft={setDraft}
          onConfirm={shelve} onCancel={() => setDraft(null)} />
      )}
    </div>
  );
}

/* --------------------------------------------------------- book cover -- */
function ReadCover({ book, index, onRemove }: {
  book: BookMeta; index: number; onRemove?: () => void;
}) {
  const rot = ((index * 37) % 5) * 0.4 - 0.8;
  const pos = getPos(book.bid);
  const pct = pos ? Math.round(pos.pct * 100) : 0;
  return (
    <div className="book-cover" style={{ transform: `rotate(${rot}deg)` }}
      onClick={() => nav(`/read/${book.bid}`)}>
      <div style={{ fontSize: 12.5, color: "var(--ink-faint)", letterSpacing: 1 }}>
        {book.published || book.added || "—"}
        {book.words
          ? ` · ${book.language === "zh"
            ? `${(book.words / 1000).toFixed(1)}k 字`
            : t("words", (book.words / 1000).toFixed(1))}`
          : ""}
      </div>
      <div className={"bk-title" + (book.language === "zh" ? " zh" : "")}>
        {book.title}
      </div>
      <div className="bk-author">{book.author}</div>
      <div style={{ margin: "8px 0 0" }}>
        <InkDivider seed={index + 3} />
      </div>
      {book.source === "builtin" && (
        <div style={{ marginTop: 10 }}>
          <span className="chip" style={{ color: "var(--ink-mute)" }}>
            {t("gutenbergNote")}
          </span>
        </div>
      )}
      <div className="bk-foot">
        <div className="bk-runs">
          {pos ? t("continueReading", pct) : t("startReading")}
        </div>
        {onRemove && (
          <button className="ink-btn small"
            style={{ color: "var(--ink-mute)" }}
            onClick={(e) => { e.stopPropagation(); onRemove(); }}>
            {t("removeBook")}
          </button>
        )}
        {!onRemove && (
          <DoodleAvatar id={"bk:" + book.bid} size={34}
            color={CHAR_COLORS[hashStr(book.bid) % CHAR_COLORS.length]} />
        )}
      </div>
      {pos && (
        <div className="read-progress">
          <div style={{ width: `${Math.max(2, pct)}%` }} />
        </div>
      )}
    </div>
  );
}

/* -------------------------------------------------------- upload sheet -- */
function UploadSheet({ draft, setDraft, onConfirm, onCancel }: {
  draft: Draft;
  setDraft: (d: Draft) => void;
  onConfirm: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="play-overlay" onClick={(e) => {
      if (e.target === e.currentTarget) onCancel();
    }}>
      <div className="play-sheet" style={{ width: "min(440px, 96vw)" }}>
        <div className="play-head">
          <div style={{ display: "flex", alignItems: "center" }}>
            <div style={{ fontFamily: "var(--font-zh)", fontSize: 20, letterSpacing: 2, flex: 1 }}>
              📚 {t("uploadFormTitle")}
            </div>
            <button className="ink-btn small" onClick={onCancel}>{t("close")}</button>
          </div>
        </div>
        <div className="play-body">
          <label style={{ display: "block", margin: "10px 0 4px", fontSize: 14 }}>
            {t("uploadTitleLabel")}
          </label>
          <input style={{ width: "100%" }} value={draft.title}
            onChange={(e) => setDraft({ ...draft, title: e.target.value })} />
          <label style={{ display: "block", margin: "14px 0 4px", fontSize: 14 }}>
            {t("uploadAuthorLabel")}
          </label>
          <input style={{ width: "100%" }} value={draft.author}
            onChange={(e) => setDraft({ ...draft, author: e.target.value })} />
          <div style={{ margin: "16px 0 0", color: "var(--ink-mute)", fontSize: 14 }}>
            {(draft.text.length / 1000).toFixed(0)}k chars
          </div>
          <div style={{ display: "flex", gap: 10, marginTop: 20 }}>
            <button className="ink-btn primary" onClick={onConfirm}>
              {t("uploadConfirm")}
            </button>
            <button className="ink-btn" onClick={onCancel}>
              {t("uploadCancel")}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
