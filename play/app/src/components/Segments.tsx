import React from "react";
import type { Seg } from "../parse";
import { t } from "../i18n";

/* One character turn as comic segments: thought clouds, stage directions,
   speech bubbles. */
export function Segments({ segs, inkColor }: { segs: Seg[]; inkColor: string }) {
  return (
    <>
      {segs.map((s, i) => {
        if (s.kind === "thought")
          return (
            <div key={i} className="bubble thought">
              <span className="q">{t("thinks")}</span>
              {s.text}
            </div>
          );
        if (s.kind === "action")
          return <div key={i} className="action-line">{s.text}</div>;
        return (
          <div key={i} className="bubble"
            style={{ "--bubble-ink": inkColor } as React.CSSProperties}>
            {s.text}
          </div>
        );
      })}
    </>
  );
}
