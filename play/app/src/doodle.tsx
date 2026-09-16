import React, { useEffect, useMemo, useRef } from "react";
import rough from "roughjs";

/* seeded rng ------------------------------------------------------------ */
export function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}
function mulberry(seed: number) {
  let a = seed || 1;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* hand-drawn wobbly line (SVG path) ------------------------------------- */
export function wobblePath(
  x1: number, y1: number, x2: number, y2: number,
  seed: number, segments = 6, amp = 1.6,
): string {
  const rnd = mulberry(seed);
  let d = `M ${x1} ${y1}`;
  for (let i = 1; i <= segments; i++) {
    const t = i / segments;
    const mx = x1 + (x2 - x1) * t + (rnd() - 0.5) * amp * 2;
    const my = y1 + (y2 - y1) * t + (rnd() - 0.5) * amp * 2;
    d += ` L ${mx.toFixed(1)} ${my.toFixed(1)}`;
  }
  return d;
}

/* squiggle divider ------------------------------------------------------ */
export function InkDivider({ seed = 7, color = "var(--ink-faint)" }:
  { seed?: number; color?: string }) {
  const d = useMemo(() => {
    const rnd = mulberry(seed);
    let path = "M 0 6";
    for (let x = 8; x <= 400; x += 8) {
      path += ` Q ${x - 4} ${6 + (rnd() - 0.5) * 7}, ${x} ${6 + (rnd() - 0.5) * 4}`;
    }
    return path;
  }, [seed]);
  return (
    <svg viewBox="0 0 400 12" preserveAspectRatio="none" height="12" style={{ display: "block", width: "100%" }}>
      <path d={d} fill="none" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

/* rough.js framed panel (hero elements only — cheap CSS wobble elsewhere) */
export function RoughPanel({
  children, pad = 18, seed, stroke = "var(--line)", fill,
  style, className,
}: {
  children: React.ReactNode; pad?: number; seed?: number;
  stroke?: string; fill?: string; style?: React.CSSProperties; className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  useEffect(() => {
    const el = ref.current, svg = svgRef.current;
    if (!el || !svg) return;
    const draw = () => {
      const { width, height } = el.getBoundingClientRect();
      if (!width || !height) return;
      svg.setAttribute("width", String(width));
      svg.setAttribute("height", String(height));
      svg.innerHTML = "";
      const rc = rough.svg(svg);
      const node = rc.rectangle(2, 2, width - 4, height - 4, {
        seed: (seed ?? 3) + 1,
        roughness: 1.6,
        bowing: 1.2,
        stroke: "currentColor",
        strokeWidth: 1.6,
        fill: fill ? "currentColor" : undefined,
        fillStyle: "hachure",
        hachureGap: 7,
        fillWeight: 0.5,
      });
      svg.appendChild(node);
    };
    draw();
    const ro = new ResizeObserver(draw);
    ro.observe(el);
    return () => ro.disconnect();
  }, [seed, fill]);
  return (
    <div ref={ref} className={className}
      style={{ position: "relative", padding: pad, ...style }}>
      <svg ref={svgRef} aria-hidden
        style={{ position: "absolute", inset: 0, color: stroke, pointerEvents: "none" }} />
      <div style={{ position: "relative" }}>{children}</div>
    </div>
  );
}

/* doodle avatar --------------------------------------------------------- */
export function DoodleAvatar({ id, size = 46, color = "var(--ink)", guest = false }:
  { id: string; size?: number; color?: string; guest?: boolean }) {
  const svg = useMemo(() => {
    const rnd = mulberry(hashStr(id));
    const j = (n: number) => (rnd() - 0.5) * n;
    const parts: string[] = [];
    const sw = 2.2;
    // head: wobbly circle out of 4 arcs
    const cx = 24 + j(1.4), cy = 26 + j(1.4), r = 15 + j(2);
    let head = `M ${cx - r} ${cy}`;
    for (let q = 0; q < 4; q++) {
      const a0 = Math.PI + (q * Math.PI) / 2;
      const a1 = a0 + Math.PI / 2;
      const mx = cx + Math.cos((a0 + a1) / 2) * (r + j(2.4));
      const my = cy + Math.sin((a0 + a1) / 2) * (r + j(2.4));
      const ex = cx + Math.cos(a1) * (r + j(1.2));
      const ey = cy + Math.sin(a1) * (r + j(1.2));
      head += ` Q ${mx.toFixed(1)} ${my.toFixed(1)}, ${ex.toFixed(1)} ${ey.toFixed(1)}`;
    }
    parts.push(`<path d="${head}" fill="none"/>`);
    // hair — five styles
    const hs = Math.floor(rnd() * 5);
    if (hs === 0) { // spiky
      for (let i = 0; i < 5; i++) {
        const a = -Math.PI * (0.25 + 0.5 * (i / 4));
        const x0 = cx + Math.cos(a) * r, y0 = cy + Math.sin(a) * r;
        parts.push(`<path d="M ${x0} ${y0} L ${x0 + j(3)} ${y0 - 6 - rnd() * 4}" fill="none"/>`);
      }
    } else if (hs === 1) { // side part curve
      parts.push(`<path d="M ${cx - r + 2} ${cy - 4} Q ${cx - 4 + j(3)} ${cy - r - 5}, ${cx + r - 1} ${cy - 5 + j(2)}" fill="none"/>`);
    } else if (hs === 2) { // bun
      parts.push(`<circle cx="${cx + j(2)}" cy="${cy - r - 3}" r="${4 + rnd() * 2}" fill="none"/>`);
      parts.push(`<path d="M ${cx - r + 3} ${cy - 6} Q ${cx} ${cy - r - 2}, ${cx + r - 3} ${cy - 6}" fill="none"/>`);
    } else if (hs === 3) { // fringe scribble
      parts.push(`<path d="M ${cx - r + 2} ${cy - 5} q 3 -4 6 -1 q 3 -5 6 -1 q 3 -4 6 -1 q 3 -3 6 0" fill="none"/>`);
    } // hs===4: bald — nothing
    // eyes — dots, lines, or circles
    const es = Math.floor(rnd() * 3);
    const ey = cy - 2 + j(1.5), dx = 5.6 + j(1);
    if (es === 0) {
      parts.push(`<circle cx="${cx - dx}" cy="${ey}" r="1.4" fill="currentColor" stroke="none"/>`);
      parts.push(`<circle cx="${cx + dx}" cy="${ey}" r="1.4" fill="currentColor" stroke="none"/>`);
    } else if (es === 1) {
      parts.push(`<path d="M ${cx - dx - 2} ${ey} q 2 ${j(4) - 1} 4 0" fill="none"/>`);
      parts.push(`<path d="M ${cx + dx - 2} ${ey} q 2 ${j(4) - 1} 4 0" fill="none"/>`);
    } else {
      parts.push(`<circle cx="${cx - dx}" cy="${ey}" r="2.4" fill="none"/>`);
      parts.push(`<circle cx="${cx + dx}" cy="${ey}" r="2.4" fill="none"/>`);
    }
    // brows sometimes
    if (rnd() < 0.6) {
      const by = ey - 5 + j(1);
      const tilt = j(3);
      parts.push(`<path d="M ${cx - dx - 2.6} ${by + tilt} L ${cx - dx + 2.6} ${by - tilt}" fill="none"/>`);
      parts.push(`<path d="M ${cx + dx - 2.6} ${by - tilt} L ${cx + dx + 2.6} ${by + tilt}" fill="none"/>`);
    }
    // mouth — smile / flat / frown / o
    const ms = Math.floor(rnd() * 4);
    const my = cy + 7 + j(1.5);
    if (ms === 0) parts.push(`<path d="M ${cx - 4} ${my} q 4 ${3 + rnd() * 2} 8 0" fill="none"/>`);
    else if (ms === 1) parts.push(`<path d="M ${cx - 3.6} ${my + 1} l 7.2 ${j(2)}" fill="none"/>`);
    else if (ms === 2) parts.push(`<path d="M ${cx - 4} ${my + 2} q 4 ${-3 - rnd() * 2} 8 0" fill="none"/>`);
    else parts.push(`<circle cx="${cx + j(1)}" cy="${my}" r="1.8" fill="none"/>`);
    if (guest) {
      parts.push(`<path d="M 34 8 l 6 -5 l 2 6 z" fill="currentColor" stroke="none" opacity="0.5"/>`);
    }
    return `<g stroke="currentColor" stroke-width="${sw}" stroke-linecap="round">${parts.join("")}</g>`;
  }, [id, guest]);
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" aria-hidden
      style={{ color, display: "inline-block", verticalAlign: "middle" }}
      dangerouslySetInnerHTML={{ __html: svg }} />
  );
}

/* tiny inline glyphs for the timeline ---------------------------------- */
export function FlagGlyph({ color = "var(--seal)" }: { color?: string }) {
  return (
    <svg width="14" height="16" viewBox="0 0 14 16" style={{ color }}>
      <g stroke="currentColor" strokeWidth="1.6" fill="none" strokeLinecap="round">
        <path d="M 3 15 L 3 2" />
        <path d="M 3 2 Q 8 0.5 12 3 L 12 8 Q 8 6 3 8" fill="currentColor" fillOpacity="0.25" />
      </g>
    </svg>
  );
}
