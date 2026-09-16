import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { BookDoc, SandboxSession } from "../types";
import { chapterList, getBook, getPos, setPos, splitBlocks } from "../bookstore";
import { sandboxGet } from "../playllm";
import { nav } from "../App";
import BookChat from "../components/BookChat";
import { t } from "../i18n";

/* raw = "<bid>" or "<bid>?chat=<session id>" */
export default function Reader({ raw }: { raw: string }) {
  const [bid, chatId] = useMemo(() => {
    const q = raw.split("?");
    return [q[0], new URLSearchParams(q[1] || "").get("chat") || ""];
  }, [raw]);

  const [book, setBook] = useState<BookDoc | null>(null);
  const [err, setErr] = useState("");
  const [fs, setFs] = useState(() =>
    parseInt(localStorage.getItem("bpi.readerfs") || "17", 10));
  const [pct, setPct] = useState(0);
  const [chat, setChat] = useState<{ upto: number; session?: SandboxSession } | null>(null);

  const bodyRef = useRef<HTMLDivElement>(null);
  const offRef = useRef(0);              // current reading offset (chars)
  const saveTimer = useRef<number>();

  useEffect(() => {
    setBook(null);
    getBook(bid).then(setBook).catch((e) => setErr(String(e)));
  }, [bid]);

  /* reopen a saved conversation */
  useEffect(() => {
    if (!book || !chatId) return;
    sandboxGet(chatId).then((s) => {
      if (s) setChat({ upto: s.upto ?? book.text.length, session: s });
    });
  }, [book, chatId]);

  const blocks = useMemo(() => (book ? splitBlocks(book.text) : []), [book]);
  const chapters = useMemo(() => chapterList(blocks), [blocks]);

  /* restore position once the text is on screen */
  useEffect(() => {
    if (!book || !bodyRef.current) return;
    const saved = getPos(bid);
    offRef.current = saved?.off ?? 0;
    setPct(Math.round((saved?.pct ?? 0) * 100));
    if (saved && saved.off > 0) {
      const idx = blocks.findIndex((b) => b.off + b.text.length >= saved.off);
      const el = bodyRef.current.children[Math.max(0, idx)] as HTMLElement;
      if (el) window.scrollTo({ top: el.offsetTop - 90 });
    } else {
      window.scrollTo({ top: 0 });
    }
  }, [book, bid, blocks]);

  /* scroll → current offset, throttled; persisted on a short debounce */
  useEffect(() => {
    if (!book) return;
    const total = book.text.length || 1;
    let timer = 0;
    const measure = () => {
        timer = 0;
        const host = bodyRef.current;
        if (!host) return;
        const kids = host.children;
        const line = window.scrollY + window.innerHeight * 0.4;
        // last block whose top sits above the reading line
        let lo = 0, hi = kids.length - 1, best = 0;
        while (lo <= hi) {
          const mid = (lo + hi) >> 1;
          if ((kids[mid] as HTMLElement).offsetTop <= line) {
            best = mid; lo = mid + 1;
          } else hi = mid - 1;
        }
        const b = blocks[Math.min(best, blocks.length - 1)];
        if (!b) return;
        const off = b.off + b.text.length;
        const atEnd = window.scrollY + window.innerHeight
          >= document.documentElement.scrollHeight - 40;
        offRef.current = atEnd ? total : off;
        const p = Math.min(1, offRef.current / total);
        setPct(Math.round(p * 100));
        window.clearTimeout(saveTimer.current);
        saveTimer.current = window.setTimeout(
          () => setPos(bid, offRef.current, p), 400);
    };
    const onScroll = () => {   // ~7 measures/second is plenty
      if (!timer) timer = window.setTimeout(measure, 140);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    measure();
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.clearTimeout(timer);
      window.clearTimeout(saveTimer.current);
    };
  }, [book, bid, blocks]);

  const setFontSize = (n: number) => {
    const v = Math.min(22, Math.max(14, n));
    setFs(v);
    localStorage.setItem("bpi.readerfs", String(v));
  };

  if (err) return <div className="loading-ink">{t("errPrefix")} · {err}</div>;
  if (!book) return <div className="loading-ink">{t("loadingBookText")}</div>;

  return (
    <div className="fade-in">
      <div className="reader-top">
        <a onClick={() => nav("/")}
          style={{ cursor: "pointer", whiteSpace: "nowrap" }}>
          ← {t("crumbReadShelf")}
        </a>
        <span className="reader-title">
          {book.title}
          {book.author && (
            <span style={{ color: "var(--ink-mute)", fontSize: 13.5, marginLeft: 8 }}>
              {book.author}
            </span>
          )}
        </span>
        <span style={{ flex: 1 }} />
        {chapters.length > 1 && (
          <select
            value=""
            title={t("toc")}
            style={{ maxWidth: 170, padding: "3px 8px", fontSize: 14 }}
            onChange={(e) => {
              const i = parseInt(e.target.value, 10);
              const el = bodyRef.current?.children[i] as HTMLElement;
              if (el) window.scrollTo({ top: el.offsetTop - 90 });
            }}>
            <option value="" disabled>{t("tocJump")}</option>
            {chapters.map((c) => (
              <option key={c.block} value={c.block}>{c.title.slice(0, 40)}</option>
            ))}
          </select>
        )}
        <button className="ink-btn small" title={t("fontSmaller")}
          onClick={() => setFontSize(fs - 1)}>A−</button>
        <button className="ink-btn small" title={t("fontBigger")}
          onClick={() => setFontSize(fs + 1)}>A＋</button>
        <span className="chip" style={{ color: "var(--indigo)" }}>
          {t("readProgress", pct)}
        </span>
      </div>

      <div className="reader-body" ref={bodyRef}
        style={{ "--reader-fs": `${fs}px` } as React.CSSProperties}>
        {blocks.map((b, i) =>
          b.heading ? (
            <h2 key={i} className="reader-chap">{b.text}</h2>
          ) : (
            <p key={i} className="reader-para">{b.text}</p>
          ),
        )}
        <div className="ambient" style={{ marginTop: 60 }}>· 完 ·</div>
      </div>

      {/* portal: fixed-position UI must not sit under .fade-in — its entry
          animation's transform would re-anchor `fixed` to the page */}
      {createPortal(
        <>
          <button className="ink-btn seal reader-fab"
            onClick={() => setChat({ upto: offRef.current })}>
            {t("talkFab")}
          </button>
          {chat && (
            <BookChat book={book} upto={chat.upto}
              pct={Math.round(Math.min(1, chat.upto / (book.text.length || 1)) * 100)}
              session={chat.session}
              onClose={() => {
                setChat(null);
                if (chatId) nav(`/read/${bid}`);
              }} />
          )}
        </>,
        document.body,
      )}
    </div>
  );
}
