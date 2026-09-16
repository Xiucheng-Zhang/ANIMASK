/* Reader data layer, dual-mode like playllm.ts:
   - "server": play/server.py owns uploads (play/books/) and proxies the LLM
   - "static": built-ins ship as data/book__*.json; uploads live in the
     visitor's localStorage; LLM calls go direct with the visitor's key.
   Prompt builders are mirrored in play/server.py; keep the two in sync. */
import type { BookCharacter, BookDoc, BookMeta } from "./types";
import { detectMode, directQuery } from "./playllm";

const CHAR_SCAN_CHARS = 50000;
const BOOKCHAT_CTX_CHARS = 12000;
export const MAX_BOOK_MB = 8;

/* ------------------------------------------------------------ fetch glue */
async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  const o = await r.json();
  if (o && o.error) throw new Error(o.error);
  return o as T;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const o = await r.json();
  if (o && o.error) throw new Error(o.error);
  return o as T;
}

/* -------------------------------------------------- text canonical form */
/* mirrors server.py _normalize_book — offsets must agree everywhere */
export function normalizeBook(text: string): string {
  text = text.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  text = text.replace(/\[Illustration[^\]]*\]/gs, "");
  text = text.replace(/\n{3,}/g, "\n\n");
  return text.trim() + "\n";
}

export function detectBookLang(text: string): "en" | "zh" {
  const head = text.slice(0, 2000);
  let cjk = 0;
  for (const ch of head) if (ch >= "一" && ch <= "鿿") cjk++;
  return cjk > head.length * 0.15 ? "zh" : "en";
}

export function countWords(text: string, language: "en" | "zh"): number {
  if (language === "zh") {
    let n = 0;
    for (const ch of text) if (ch >= "一" && ch <= "鿿") n++;
    return n;
  }
  return text.split(/\s+/).filter(Boolean).length;
}

/* ------------------------------------------------- localStorage uploads */
const IDX_KEY = "bpi.mybooks";

function lsIndex(): BookMeta[] {
  try { return JSON.parse(localStorage.getItem(IDX_KEY) || "[]"); }
  catch { return []; }
}

function hashText(s: string): string {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0).toString(36) + s.length.toString(36);
}

/* --------------------------------------------------------------- books */
export async function listBooks(): Promise<BookMeta[]> {
  if ((await detectMode()) === "server") {
    return (await getJson<{ books: BookMeta[] }>("/api/books")).books;
  }
  const builtin = (await getJson<{ books: BookMeta[] }>("data/books.json")).books;
  return [...lsIndex(), ...builtin];
}

export async function getBook(bid: string): Promise<BookDoc> {
  if ((await detectMode()) === "server") {
    return getJson(`/api/book/${encodeURIComponent(bid)}`);
  }
  if (bid.startsWith("u")) {
    const meta = lsIndex().find((b) => b.bid === bid);
    const text = localStorage.getItem("bpi.mybook." + bid);
    if (!meta || text == null) throw new Error(`no such book: ${bid}`);
    return { ...meta, text };
  }
  return getJson(`data/book__${bid}.json`);
}

export async function addBook(
  title: string, author: string, rawText: string,
): Promise<string> {
  const text = normalizeBook(rawText);
  if (!text.trim()) throw new Error("empty text");
  const language = detectBookLang(text);
  if ((await detectMode()) === "server") {
    const r = await post<{ bid: string }>("/api/book/upload",
      { title, author, language, text: rawText });
    return r.bid;
  }
  const bid = "u" + hashText(text);
  const meta: BookMeta = {
    bid, title: title.slice(0, 120), author: author.slice(0, 80),
    language, words: countWords(text, language), source: "upload",
    added: new Date().toISOString().slice(0, 16).replace("T", " "),
  };
  localStorage.setItem("bpi.mybook." + bid, text);   // may throw QuotaExceeded
  const idx = lsIndex().filter((b) => b.bid !== bid);
  idx.unshift(meta);
  localStorage.setItem(IDX_KEY, JSON.stringify(idx));
  return bid;
}

export async function removeBook(bid: string): Promise<void> {
  if ((await detectMode()) === "server") {
    await post("/api/book/delete", { bid });
    return;
  }
  localStorage.removeItem("bpi.mybook." + bid);
  localStorage.removeItem("bpi.chars." + bid);
  localStorage.setItem(IDX_KEY,
    JSON.stringify(lsIndex().filter((b) => b.bid !== bid)));
}

/* ----------------------------------------------------- reading position */
export interface ReadPos { off: number; pct: number; ts: number }

function posMap(): Record<string, ReadPos> {
  try { return JSON.parse(localStorage.getItem("bpi.readpos") || "{}"); }
  catch { return {}; }
}

export function getPos(bid: string): ReadPos | null {
  return posMap()[bid] || null;
}

export function setPos(bid: string, off: number, pct: number) {
  const m = posMap();
  m[bid] = { off, pct, ts: Date.now() };
  localStorage.setItem("bpi.readpos", JSON.stringify(m));
}

/* -------------------------------------------------------- text blocks */
export interface Block { text: string; off: number; heading: boolean }

const HEAD_RES = [
  /^(chapter|book|part|volume|act|scene|stave|canto)\s+[ivxlcdm\d]+\b/i,
  /^第[一二三四五六七八九十百千万零〇0-9]+[章回卷节幕部]/,
  /^[IVXLCDM]+\.?$/,
];

function looksLikeHeading(t: string): boolean {
  if (t.length > 100) return false;
  // a chapter block may carry its title on a second line:
  // "CHAPTER I.\nDown the Rabbit-Hole"
  const lines = t.split("\n").map((s) => s.trim()).filter(Boolean);
  if (lines.length > 2) return false;
  if (HEAD_RES.some((re) => re.test(lines[0]))) return true;
  if (lines.length > 1) return false;
  // short shouty line, e.g. "PRIDE." — but never plain numbers/punctuation
  return t.length < 46 && /[A-Z]/.test(t) && !/[a-z]/.test(t);
}

export function splitBlocks(text: string): Block[] {
  const blocks: Block[] = [];
  const re = /\n\s*\n/g;
  let last = 0;
  let m: RegExpExecArray | null;
  const push = (raw: string, off: number) => {
    const lead = raw.length - raw.replace(/^\s+/, "").length;
    const t = raw.trim();
    if (t) blocks.push({ text: t, off: off + lead, heading: false });
  };
  while ((m = re.exec(text))) {
    push(text.slice(last, m.index), last);
    last = re.lastIndex;
  }
  push(text.slice(last), last);
  blocks.forEach((b) => { b.heading = looksLikeHeading(b.text); });
  // runs of 3+ consecutive "headings" are a table of contents — demote them
  let i = 0;
  while (i < blocks.length) {
    if (!blocks[i].heading) { i++; continue; }
    let j = i;
    while (j < blocks.length && blocks[j].heading) j++;
    if (j - i >= 3) for (let k = i; k < j; k++) blocks[k].heading = false;
    i = j;
  }
  return blocks;
}

export function chapterList(blocks: Block[]): { block: number; title: string }[] {
  return blocks
    .map((b, i) => ({ b, i }))
    .filter(({ b }) => b.heading)
    .map(({ b, i }) => ({ block: i, title: b.text.replace(/\s+/g, " ") }));
}

/* ------------------------------------------------------ cast extraction */
function charsPrompt(book: BookDoc): string {
  if (book.language === "zh") {
    return `你正在读《${book.title}》（${book.author || "佚名"}）的开头部分。请找出在这部分里真正出场过的主要角色（至多 10 个）。每个角色给出：
- "name"：文中最常用的称呼
- "brief"：一句话说明 TA 是谁（用原书的语言）

严格输出 JSON 数组，形如 [{"name": "…", "brief": "…"}]，不要输出其他任何内容。`;
  }
  return `You are reading the opening portion of "${book.title}" by ${book.author || "an unknown author"}. List the main characters who actually appear in this portion (at most 10). For each give:
- "name": the name the text uses most often
- "brief": one sentence on who they are, in the book's own language

Output STRICTLY a JSON array like [{"name": "...", "brief": "..."}]. Nothing else.`;
}

function parseChars(raw: string): BookCharacter[] {
  const m = raw.match(/\[[\s\S]*\]/);
  if (!m) return [];
  let arr: unknown;
  try { arr = JSON.parse(m[0]); } catch { return []; }
  if (!Array.isArray(arr)) return [];
  const out: BookCharacter[] = [];
  for (const it of arr) {
    if (it && typeof it === "object" && (it as { name?: unknown }).name) {
      const o = it as { name: unknown; brief?: unknown };
      out.push({ name: String(o.name).slice(0, 60),
        brief: String(o.brief ?? "").slice(0, 300) });
    }
  }
  return out.slice(0, 10);
}

/** cachedOnly=true never triggers an LLM call; null = not scanned yet */
export async function bookCharacters(
  book: BookDoc, model: string, cachedOnly = false,
): Promise<BookCharacter[] | null> {
  if ((await detectMode()) === "server") {
    const r = await post<{ characters: BookCharacter[] | null }>(
      "/api/book/characters",
      { bid: book.bid, model, cached_only: cachedOnly });
    return r.characters;
  }
  const key = "bpi.chars." + book.bid;
  try {
    const hit = localStorage.getItem(key);
    if (hit) return JSON.parse(hit);
  } catch { /* rescan */ }
  if (!book.bid.startsWith("u")) {
    // built-in books ship with a pre-extracted cast, so the chips show
    // before the visitor has entered any key
    try {
      const r = await fetch(`data/chars__${encodeURIComponent(book.bid)}.json`);
      if (r.ok) {
        const shipped = await r.json();
        if (Array.isArray(shipped) && shipped.length) return shipped;
      }
    } catch { /* none shipped */ }
  }
  if (cachedOnly) return null;
  const raw = await directQuery(
    charsPrompt(book), book.text.slice(0, CHAR_SCAN_CHARS));
  const chars = parseChars(raw);
  if (chars.length) localStorage.setItem(key, JSON.stringify(chars));
  return chars;
}

/* ------------------------------------------------------------ book chat */
type ChatMsg = { speaker: "user" | "char"; text: string };

function bookChatPrompt(
  lang: "en" | "zh", name: string, visitor: string, history: ChatMsg[],
): string {
  const convo = history.map((h) =>
    h.speaker === "user" ? `${visitor}: ${h.text}` : `${name}: ${h.text}`).join("\n");
  const lastUser = history.length && history[history.length - 1].speaker === "user"
    ? history[history.length - 1].text : "";
  if (lang === "zh") {
    return `这是一次故事内的对话。这本书在下面摘录的结尾处暂停了，一位书里从未出现过的访客「${visitor}」走进此刻的场景，来到 ${name} 面前，与其交谈。

你扮演 ${name}。严格依据原文体现的性格、说话方式与当下处境作答；人物只知道故事到此为止发生的事，不知道之后的任何情节，也不知道任何"书外"的信息。若访客问到人物不可能知道的事，人物应表现出困惑，或按自己的理解回应。

输出格式：内心独白用[…]，动作神态用（…），说出的话用「…」。只输出 ${name} 的下一个回合，不要代访客说话，不要任何解说。

对话至今：
${convo || "（访客刚刚到来，还未开口。）"}

${visitor} 刚说：${lastUser || "（沉默地看着你。）"}`;
  }
  return `This is an in-story conversation. The book is paused at the end of the excerpt below, and a visitor who never appears in it — "${visitor}" — steps into the scene and approaches ${name}.

You play ${name}. Stay strictly true to the personality, voice and situation the text shows. The character only knows what has happened in the story up to this point — nothing later, nothing outside the book. If the visitor asks about things the character cannot know, react with confusion or interpret it their own way.

Format: inner thoughts in [brackets], actions in (parentheses), spoken words in "quotes". Output only ${name}'s next turn. Never speak for the visitor. No commentary.

Conversation so far:
${convo || "(The visitor has just arrived and not yet spoken.)"}

${visitor} just said: ${lastUser || "(watches you silently.)"}`;
}

export async function bookChat(book: BookDoc, args: {
  char_name: string; char_brief: string; upto: number;
  history: ChatMsg[]; user_name: string; model: string;
}): Promise<string> {
  if ((await detectMode()) === "server") {
    const r = await post<{ reply: string }>("/api/bookchat",
      { bid: book.bid, ...args });
    if (r.reply.startsWith("<<ERROR")) throw new Error(r.reply);
    return r.reply;
  }
  const upto = Math.max(600, args.upto);
  const ctx = book.text.slice(0, upto).slice(-BOOKCHAT_CTX_CHARS);
  const visitor = args.user_name.trim() ||
    (book.language === "zh" ? "一位不速之客" : "a stranger");
  const card = [
    `BOOK: ${book.title} — ${book.author || ""}`,
    `CHARACTER: ${args.char_name}`,
    ...(args.char_brief ? [`ABOUT: ${args.char_brief}`] : []),
  ].join("\n");
  const stdin = card +
    "\n\n===== THE STORY SO FAR (the reader has read exactly this far, " +
    "and no further) =====\n\n" + ctx;
  return directQuery(
    bookChatPrompt(book.language, args.char_name, visitor, args.history),
    stdin);
}
