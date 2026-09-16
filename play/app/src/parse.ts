/* ------------------------------------------------ segment parsing ------ */
export type SegKind = "thought" | "action" | "speech" | "plain";
export interface Seg { kind: SegKind; text: string }

const SEG_RE =
  /\[([^\][]*)\]|【([^【】]*)】|（([^（）]*)）|\(([^()]*)\)|「([^「」]*)」|“([^“”]*)”|"([^"]*)"/g;

/** Split a character turn into comic segments: [thought] (action) "speech". */
export function parseSegments(text: string): Seg[] {
  const segs: Seg[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  SEG_RE.lastIndex = 0;
  const push = (kind: SegKind, t: string) => {
    t = t.trim();
    if (!t) return;
    const prev = segs[segs.length - 1];
    if (prev && prev.kind === kind && kind === "speech") {
      prev.text += "\n" + t;   // consecutive quoted lines join one bubble
    } else segs.push({ kind, text: t });
  };
  while ((m = SEG_RE.exec(text))) {
    const between = text.slice(last, m.index);
    if (between.trim()) push("plain", between);
    if (m[1] !== undefined || m[2] !== undefined)
      push("thought", m[1] ?? m[2] ?? "");
    else if (m[3] !== undefined || m[4] !== undefined)
      push("action", m[3] ?? m[4] ?? "");
    else push("speech", m[5] ?? m[6] ?? m[7] ?? "");
    last = m.index + m[0].length;
  }
  const tail = text.slice(last);
  if (tail.trim()) push("plain", tail);
  return segs.length ? segs : [{ kind: "plain", text }];
}

/* ------------------------------------------------ character colors ----- */
export const CHAR_COLORS = [
  "var(--char-0)", "var(--char-1)", "var(--char-2)",
  "var(--char-3)", "var(--char-4)", "var(--char-5)",
];
