export interface BookMeta {
  bid: string;
  title: string;
  author: string;
  published?: string;
  language: "en" | "zh";
  words: number | null;
  source: "builtin" | "upload";
  added?: string;
}

export interface BookDoc extends BookMeta {
  text: string;
}

export interface BookCharacter {
  name: string;
  brief: string;
}

/* a saved conversation with a character, at a reading position */
export interface SandboxSession {
  id?: string;
  kind: "bookchat";
  sid: string;          // book id
  tag: string;
  upto: number;         // character offset the reader had reached
  title: string;
  role_code?: string;   // character name
  user_name?: string;
  model?: string;
  messages: unknown[];
  saved_at?: string;
  n_messages?: number;
}
