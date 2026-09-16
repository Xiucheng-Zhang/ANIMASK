import { useSyncExternalStore } from "react";

/* ------------------------------------------------------------------ store */
export type Lang = "zh" | "en";

/* English first; the corner toggle switches and remembers the choice */
let lang: Lang = (localStorage.getItem("bpi.lang") as Lang) || "en";

const subs = new Set<() => void>();

export function setLang(l: Lang) {
  lang = l;
  localStorage.setItem("bpi.lang", l);
  subs.forEach((f) => f());
}

export function getLang(): Lang {
  return lang;
}

export function useLang(): Lang {
  return useSyncExternalStore(
    (cb) => { subs.add(cb); return () => subs.delete(cb); },
    () => lang,
  );
}

/* ------------------------------------------------------------- dictionary */
/* UI chrome only — story content stays in the book's own language. */
const DICT = {
  pageName: ["和书中人聊聊", "Talk to a character"],
  sampleLead: ["先找这两位", "Start with"],
  sampleAsk: ["试着问问", "Try asking"],
  sampleTag: ["示例", "sample"],

  /* ---- reader shelf ---- */
  myShelf: ["我的书", "My Books"],
  myShelfSub: ["", ""],
  houseShelf: ["馆藏", "House Books"],
  houseShelfSub: ["公版经典", "public domain classics"],
  addBookBtn: ["＋ 上传一本书", "+ add a book"],
  addBookHintStatic: ["纯文本 .txt，只存在你的浏览器里", "Plain .txt, kept in your browser"],
  addBookHintServer: ["纯文本 .txt，存在本机", "Plain .txt, kept on this machine"],
  uploadFormTitle: ["给这本书上个架", "Put it on the shelf"],
  uploadTitleLabel: ["书名", "title"],
  uploadAuthorLabel: ["作者（可留空）", "author (optional)"],
  uploadConfirm: ["上架 → 开始读", "shelve it → start reading"],
  uploadCancel: ["算了", "never mind"],
  uploadFailQuota: ["浏览器空间不够了，删掉几本再试。", "Your browser is out of space. Remove a book and try again."],
  emptyMyShelf: ["还没有书。上传一个 .txt 就能开始。", "Nothing here yet. Add a .txt to start."],
  gutenbergNote: ["公版 · Project Gutenberg", "public domain · Project Gutenberg"],
  continueReading: ["读到 {0}% · 继续", "at {0}% · continue"],
  startReading: ["从头开始读", "start reading"],
  removeBook: ["拿走", "remove"],
  confirmRemove: ["把《{0}》从书架上拿走？", "Remove “{0}” from the shelf?"],
  bookChatDrafts: ["聊过的天", "Saved Conversations"],
  bookChatDraftsSub: ["", ""],
  labEntry: [
    "🎭 实验剧场 —— 看角色们自己把书演完（研究档案）",
    "🎭 the lab — watch characters play whole books out on their own (research archive)",
  ],
  crumbReadShelf: ["书架", "Shelf"],
  labTitle: ["实验剧场", "The Lab"],

  /* ---- reader ---- */
  toc: ["目录", "contents"],
  tocJump: ["跳到 …", "jump to …"],
  readProgress: ["已读 {0}%", "{0}% read"],
  talkFab: ["✎ 找个角色聊聊", "✎ talk to a character"],
  loadingBookText: ["正在翻开这本书……", "opening the book…"],
  fontSmaller: ["字小些", "smaller type"],
  fontBigger: ["字大些", "bigger type"],

  /* ---- book chat ---- */
  whoToTalk: ["找谁聊？", "Who do you want to talk to?"],
  presentChars: ["已出场的角色", "on stage so far"],
  scanChars: ["🔍 认一认书里的角色", "🔍 meet the cast"],
  scanningChars: ["正在认人……", "reading the cast…"],
  scanCharsFail: ["没找到，直接输入名字也行。", "Couldn't find anyone. Type a name instead."],
  freeCharName: ["或输入任意角色名…", "or type any character's name…"],
  spoilerNote: ["只知道你读到的地方（{0}%）之前的事", "knows the story as far as you've read ({0}%)"],
  bookChatEmpty: ["故事停在你读到的地方。说点什么吧。", "The story is paused where you left it. Say something."],
  bookDraftTitle: ["与{0}聊《{1}》· 读到 {2}%", "chat with {0} · “{1}” at {2}%"],
  shelf: ["书架", "The Shelf"],
  shelfSub: ["the shelf", "书架"],
  shelfCount: ["{0} 本书 · 角色们自己演完了 {1} 场", "{0} books · {1} performances, played by the characters themselves"],
  playhouseDrafts: ["小剧场手稿", "Playhouse Drafts"],
  playhouseDraftsSub: ["your saved plays", "你存下的手稿"],
  kindChat: ["对话", "chat"],
  kindSteer: ["引导", "steer"],
  words: ["{0}k 词", "{0}k words"],
  perfCount: ["{0} 场演出", "{0} performances"],
  foundEndingBadge: ["演回过原著", "found the ending"],
  foundEndingBadgeTip: ["有一场演出走回了原著的结局", "one performance found its way back to the book's own ending"],
  variantOracle: ["全书人格", "whole-book persona"],
  variantPrefreeze: ["冻结前人格", "pre-freeze persona"],
  variantTip: ["角色的性格从整本书提取，记忆只到冻结点", "personalities drawn from the whole book; memories stop at the freeze point"],

  crumbShelf: ["书架", "Shelf"],
  crumbPerf: ["演出", "performance"],
  personaLabel: ["角色人格：{0}", "personas: {0}"],
  tensionNote: ["故事停在这个悬念上", "The story stops on this question"],
  cast: ["角色", "The Cast"],
  castSub: ["the cast", "角色"],
  performances: ["历次演出", "The Performances"],
  performancesSub: ["the performances", "历次演出"],
  performancesHint: [
    "同一本书、同一个冻结时刻，每次都可能演出不一样的故事",
    "same book, same frozen moment — a different story may play out each time",
  ],
  expandWorld: ["（展开世界设定）", "(expand the world)"],
  thRun: ["场次", "Run"], thEnding: ["结局", "Ending"], thStoryTime: ["故事时长", "Story time"],
  thChapters: ["回数", "Chapters"], thPanels: ["格数", "Panels"], thNotes: ["评语", "Notes"],

  runChips: ["{0} 回 · {1} 格 · 故事里 {2}", "{0} chapters · {1} panels · {2} in-story"],
  narratorNotes: ["说书人批注", "narrator's notes"],
  hideVerdict: ["收起评语", "hide the verdict"],
  showVerdict: ["看评语", "the verdict"],
  theEndBanner: [
    "— 全文完 · 把鼠标停在任何一格上，就能从那一刻走进这个故事 —",
    "— The End · hover any panel to step into the story from that moment —",
  ],
  loadingShelf: ["正在翻开书架……", "opening the shelf…"],
  loadingBook: ["正在翻开这本书……", "opening the book…"],
  loadingRun: ["正在铺开这场演出……", "unrolling the performance…"],
  errPrefix: ["出错了", "something went wrong"],
  bookMissing: ["书不存在", "no such book"],

  thinks: ["内心", "thinks"],
  guest: ["客串", "guest"],
  btnTalk: ["✎ 对话", "✎ talk"],
  btnSteer: ["⤳ 引导", "⤳ steer"],
  btnTalkTip: ["与此刻的角色对话", "talk to the character as they are right now"],
  btnSteerTip: ["从这一刻引导剧情", "steer the plot from this moment"],
  scene: ["◎ 现场", "◎ scene"],
  worldEvent: ["❍ 世界事件", "❍ world event"],
  timeSkip: ["时光快进 +{0}", "time skips ahead +{0}"],
  intermission: ["— 幕间休息 {0} —", "— intermission {0} —"],
  intermissionAbout: ["· 约 {0} ", "· ~{0} "],
  chapterN: ["第 {0} 回", "Chapter {0}"],
  markerGoals: ["· 角色们定下心愿 ·", "· the characters set their hearts ·"],
  markerStart: ["· 开演 ·", "· curtain up ·"],
  narrator: ["说书人", "narrator"],
  plotMoved: ["剧情推进了", "the plot moved"],
  treadingWater: ["原地踏步", "treading water"],
  theEnd: ["剧终", "The End"],
  epilogue: ["尾声", "Epilogue"],
  runtimeError: ["运行错误", "runtime error"],
  anchorLine: [
    "⚑ 到了原著结局的时刻 — 记下此刻的定格，故事继续",
    "⚑ the book's own ending-time arrives — a snapshot is taken, the play goes on",
  ],

  nowChapter: ["此刻 · 第 {0} 回", "Now · Chapter {0}"],
  storyTimePassed: ["故事里过了：", "story time: "],
  stagnant: ["（原地踏步 ×{0}）", "(treading water ×{0})"],
  hangingQuestion: ["核心悬念", "The Hanging Question"],
  answered: ["· 已见分晓 ✓", "· answered ✓"],
  openPromises: ["未了之约 · {0}", "Open Promises · {0}"],
  noPromises: ["（此刻没有未了的约定）", "(no open promises right now)"],
  kAppointment: ["约定", "appointment"], kDeadline: ["期限", "deadline"],
  kReply: ["待回复", "awaiting reply"], kObligation: ["承诺", "promise"],
  madeInCh: ["{0}，第 {1} 回立下", "{0}, made in ch. {1}"],
  nMore: ["…还有 {0} 条", "…{0} more"],
  whatTheyWant: ["角色们此刻想要什么", "What They Want Right Now"],

  stFoundEnding: ["演回原著", "found the ending"],
  stOwnWay: ["另辟蹊径", "went its own way"],
  stFoundEndingTip: ["角色们自己演到了和原著相同的结局", "the characters found their way to the book's own ending"],
  stOwnWayTip: ["角色们把故事演向了别处", "the characters took the story somewhere else"],
  stCoherence: ["连贯 {0}/5", "coherence {0}/5"],
  stCoherenceTip: ["剧情顺不顺、有没有前后矛盾", "does the plot hold together, no contradictions"],
  stTooHarmonious: ["一团和气", "made peace too easily"],
  stTooHarmoniousTip: ["本该针锋相对的矛盾被角色们悄悄抹平了", "conflicts that should have clashed were quietly smoothed over"],
  whereItWent: ["这一场演到了哪里", "Where this performance went"],
  matchesBook: ["和原著对上了", "Matches the book"],
  departsBook: ["和原著岔开了", "Departs from the book"],
  notes: ["点评：", "notes: "],
  storyTimeFold: ["故事里过了多久", "How much story-time passed"],
  bookEndTimeFold: ["原著结局的时刻", "When the book itself ends"],
  snapshotAt: ["走到这一刻时的定格：第 {0} 回 · 故事里 {1}", "the snapshot at that moment: chapter {0} · {1} in-story"],
  vetoFold: ["时间快进被拦下 · {0} 次", "Time-skips vetoed · {0}"],
  vetoExplain: [
    "每当故事想快进，会先悄悄问过每个角色——有人心里还有事没办完，就会拦下时间。",
    "before time skips ahead, each character is quietly asked — anyone with unfinished business can hold time back.",
  ],
  vetoWho: ["谁拦的", "Who"], vetoHow: ["想跳多久", "How far"],
  vetoWhy: ["TA 心里的小算盘", "Their private reason"],
  clockAxis: ["一回一格 →（纵轴：故事里过了多久）", "one chapter per step → (y: story time)"],
  tipSkipped: ["快进", "skipped"], tipNewPromises: ["立下新约 {0}", "new promises {0}"],

  playChatTitle: ["时空对话", "Talk Across Time"],
  playSteerTitle: ["引导剧情", "Steer the Plot"],
  pausedAt: ["故事停在第 {0} 格 · 小剧场，不影响正片", "paused at panel {0} · playhouse — the real record is untouched"],
  chatWith: ["与此刻的 {0} 交谈 —— TA 只知道故事至今发生的事", "talk to {0} as they are right now — they only know the story so far"],
  steerSub: ["往故事里丢一个新事件，看角色们怎么接 —— 每次续演约 5 格", "toss a new event into the story and watch them catch it — ~5 panels per turn"],
  yourName: ["你的名字（可不填）", "your name (optional)"],
  saveDraft: ["存下来", "save"],
  savedDraft: ["已保存 ✓", "saved ✓"],
  close: ["合上 ✕", "close ✕"],
  chatEmpty: [
    "（你走进了暂停的故事。开口吧 —— 角色会以此刻的记忆与心境回应。）",
    "(You have stepped into the paused story. Speak — they will answer with the memories and mood of this very moment.)",
  ],
  steerEmpty: [
    "（写下一个事件 —— 一场大火、一封来信、一位不速之客……然后看故事怎么长出去。）",
    "(Write an event — a fire, a letter, an uninvited guest… then watch the story grow.)",
  ],
  penUp: ["{0} 正在写……", "{0} is writing…"],
  worldTurning: ["世界正在转动……", "the world is turning…"],
  errLabel: ["出错", "error"],
  chatPlaceholder: ["说点什么……", "Say something…"],
  steerPlaceholderFirst: ["例：一场大火从钟楼烧起 / 一位陌生贵妇登门指名要见她……", "e.g. a fire breaks out in the bell tower / a strange noblewoman arrives asking for her…"],
  steerPlaceholderMore: ["追加新的指令，或留空让它顺其自然……", "add a new directive, or leave empty to let it flow…"],
  send: ["发送 ➤", "send ➤"],
  action: ["开演 ⤳", "action ⤳"],
  continueAction: ["续演 ⤳", "continue ⤳"],
  directorConfused: ["导演没读懂，换个说法试试", "the director didn't get it — try phrasing it differently"],
  you: ["你", "you"],
  playhouseMark: ["小剧场", "PLAYHOUSE"],
  draftChatTitle: ["与 {0} 对话 · 第 {1} 格", "chat with {0} · panel {1}"],
  draftSteerTitle: ["引导剧情 · 从第 {0} 格演起", "steering · from panel {0}"],

  ocAchieved: ["圆满收场", "resolved, and well"],
  ocFailed: ["事与愿违", "resolved, badly"],
  ocTime: ["抵达原著时刻", "reached the book's ending-time"],
  ocRounds: ["篇幅用尽", "ran out of pages"],
  ocDeadlock: ["剧情卡住", "the plot got stuck"],
  ocMotives: ["心愿尘埃落定", "hearts settled"],
  ocOngoing: ["未完待续", "to be continued"],

  hMin: ["{0} 分钟", "{0} min"],
  hHour: ["{0} 小时", "{0} h"],
  hDay: ["{0} 天", "{0} days"],

  settings: ["API key", "API key"],
  settingsTitle: ["填写 API key", "Your API key"],
  settingsNote: [
    "支持任何 OpenAI 兼容接口，比如 api.openai.com 配 gpt-5.5，或 api.deepseek.com 配 deepseek-v4-flash。key 只存在你的浏览器里。",
    "Any OpenAI-compatible endpoint works, for example api.openai.com with gpt-5.5 or api.deepseek.com with deepseek-v4-flash. The key stays in your browser.",
  ],
  settingsBase: ["接口地址（不含 /v1）", "Base URL, without /v1"],
  settingsKey: ["API key", "API key"],
  settingsModel: ["模型", "Model"],
  settingsSave: ["保存", "Save"],
  settingsClear: ["清除", "Clear"],
  needKey: ["想自己提问，先在右上角填一个 API key。", "To ask your own questions, add an API key in the top corner."],
} as const;

export type StrKey = keyof typeof DICT;

export function t(key: StrKey, ...args: (string | number)[]): string {
  const pair = DICT[key];
  let s: string = pair[lang === "zh" ? 0 : 1];
  args.forEach((a, i) => { s = s.split(`{${i}}`).join(String(a)); });
  return s;
}

/* hours formatter — unit words follow the UI language */
export function fmtHours(h: number | null | undefined): string {
  if (h == null || isNaN(h)) return "—";
  if (h < 1) return t("hMin", Math.round(h * 60));
  if (h < 48) return t("hHour", h % 1 ? h.toFixed(1) : h);
  const d = h / 24;
  return t("hDay", d % 1 ? d.toFixed(1) : d);
}

export function outcomeLabel(o: string): string {
  const m: Record<string, StrKey> = {
    "resolved:achieved": "ocAchieved", "resolved:failed": "ocFailed",
    time_reached: "ocTime", rounds_exhausted: "ocRounds",
    deadlock: "ocDeadlock", motivations_settled: "ocMotives", ongoing: "ocOngoing",
  };
  return m[o] ? t(m[o]) : (o || "—");
}

export function kindLabel(k: string): string {
  const m: Record<string, StrKey> = {
    appointment: "kAppointment", deadline: "kDeadline",
    reply: "kReply", obligation: "kObligation",
  };
  return m[k] ? t(m[k]) : k;
}

export function variantLabel(v: string): string {
  if (v === "oracle") return t("variantOracle");
  if (v === "prefreeze") return t("variantPrefreeze");
  return v;
}

/* section title helper: main text in current lang, sub in the other */
export function sectionPair(key: StrKey, subKey: StrKey): [string, string] {
  return [t(key), t(subKey)];
}

export function LangToggle() {
  const l = useLang();
  return (
    <button
      className="ink-btn small"
      title={l === "zh" ? "Switch to English" : "切换为中文"}
      onClick={() => setLang(l === "zh" ? "en" : "zh")}
      style={{ letterSpacing: 1 }}
    >
      {l === "zh" ? "EN" : "中"}
    </button>
  );
}
