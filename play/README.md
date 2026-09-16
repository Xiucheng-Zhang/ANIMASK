# Talk to a character (the play page)

The second page of the project site, at
https://xiucheng-zhang.github.io/ANIMASK/play/. Pick one of the bundled
public-domain books or bring a plain `.txt`, read, and open the chat from
the corner of the page. A language model plays the character from the text
read so far and nothing after it, so the conversation cannot spoil the
book. It is the paper's idea, a character frozen at a point in its story,
reduced to a single conversation.

The interface is English by default; the toggle in the top-right corner
switches to Chinese and remembers the choice. Book text is never
translated.

## Two ways to run the same frontend

**Static (no backend).** The build in `docs/play/` works on any file host,
including GitHub Pages. Reading is fully offline. Each bundled book ships
a few ready-made questions and replies for two of its characters
(`public/data/samples__<bid>.json`, written for the opening chapters), so
a visitor can try a conversation without any key. Free conversation calls
an OpenAI-compatible endpoint straight from the browser with a key the
visitor enters under the key button in the header. The key stays in
the visitor's own `localStorage` and never leaves the browser except to
that endpoint. Uploaded books and saved chats also live in `localStorage`.

**Local server.** From the repository root:

```bash
python3 play/server.py
```

Open http://127.0.0.1:8123. The server is standard library only. Model
calls go through `animask/llm.py`, so the provider keys in `.env` apply and
the model can be chosen in the chat sheet. Uploads are stored under
`play/books/uploads/` and saved chats under `play/sandbox/`.

The frontend probes `/api/ping` to tell the two modes apart.

## Building

```bash
cd play/app
npm ci
npm run build      # writes docs/play/
```

`npm run dev` starts a hot-reloading dev server that proxies `/api` to the
local server on port 8123.

## Layout

```
play/
  server.py             local server (book endpoints, static files)
  app/                  Vite + React frontend
    public/data/        bundled books (book__<bid>.json), their casts
                        (chars__<bid>.json) and ready-made conversations
                        (samples__<bid>.json), listed in books.json
    src/bookstore.ts    books, reading position, cast extraction, chat
    src/playllm.ts      mode probe, bring-your-own-key client, saved chats
    src/pages/          shelf and reader
    src/components/     chat sheet, comic segments
  books/chars/          cast cache written by the local server
```

The cast-extraction and chat prompts exist twice, in `server.py` and in
`src/bookstore.ts`, one per mode. The text normalization
(`_normalize_book` and `normalizeBook`) must stay identical, since reading
positions are character offsets shared by both sides.
