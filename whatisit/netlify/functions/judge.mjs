// POST /api/judge: the game's AI judge, paid for by the site, so players need no AI account.
//
// Who pays: the provider whose key you put in the site's environment variables. The cheapest is a
// free Gemini key from Google AI Studio (no card; Google stops it at its free quota, so it can't run
// up a bill). Netlify AI Gateway can also supply keys, billed to the team's Netlify credits; on the
// Free plan those credits also pay for deploys and bandwidth, and every project on the team pauses
// when they run out. So the gateway is only used if you opt in with JUDGE_USE_NETLIFY_CREDITS=1.
//
// Environment variables (all optional):
//   GEMINI_API_KEY | ANTHROPIC_API_KEY | OPENAI_API_KEY   your own key; the first one found is used
//   JUDGE_PROVIDER      gemini | anthropic | openai, to choose when several keys are set
//   JUDGE_MODEL         default per provider below
//   JUDGE_DAILY_CAP     model calls per UTC day, whole site (default 200)
//   JUDGE_MONTHLY_CAP   model calls per calendar month, whole site (default 3000)
//   JUDGE_PER_VISITOR   model calls per visitor per day (default 60)
//   JUDGE_USE_NETLIFY_CREDITS  1 to let Netlify AI Gateway's keys (and your Netlify credits) pay
//   JUDGE_OFF           1 to switch the AI judge off; the game falls back to its free judges
//   JUDGE_PASSCODE      tester passes, comma-separated: when set, only players who opened a tester
//                       link (https://your-site/?pass=CODE) get AI verdicts; everyone else gets the
//                       answer key and the robot
//
// Most guesses never reach a model: the browser and this function first look in the mystery's answer
// key (example guesses scored in advance), then in a shared cache of earlier verdicts.
import { getStore } from "@netlify/blobs";
import { answerKey, BadRequest, buildRequest, cacheText, judge } from "../lib/judge-core.mjs";

const PROVIDERS = {
  gemini: { keys: ["GEMINI_API_KEY", "GOOGLE_API_KEY"], gateway: "GOOGLE_GEMINI_BASE_URL", model: "gemini-3.1-flash-lite" },
  anthropic: { keys: ["ANTHROPIC_API_KEY"], gateway: "ANTHROPIC_BASE_URL", model: "claude-haiku-4-5" },
  openai: { keys: ["OPENAI_API_KEY"], gateway: "OPENAI_BASE_URL", model: "gpt-5-nano" },
};
const yes = (v) => /^(1|true|yes)$/i.test(v || "");
// Netlify only sets a provider's base URL when it injects its own (gateway) key for that provider.
function usable(name) {
  const p = PROVIDERS[name];
  if (!p || !p.keys.some((k) => process.env[k])) return false;
  return !process.env[p.gateway] || yes(process.env.JUDGE_USE_NETLIFY_CREDITS);
}
const PROVIDER = (process.env.JUDGE_PROVIDER || "").toLowerCase() || Object.keys(PROVIDERS).find(usable) || "gemini";
const MODEL = process.env.JUDGE_MODEL || (PROVIDERS[PROVIDER] || PROVIDERS.gemini).model;
const CAPS = {
  day: Number(process.env.JUDGE_DAILY_CAP || 200),
  month: Number(process.env.JUDGE_MONTHLY_CAP || 3000),
  visitor: Number(process.env.JUDGE_PER_VISITOR || 60),
};
const PROMPT_VERSION = "2"; // bump when the prompts change, so old cached verdicts are not reused

const switchedOn = () => !yes(process.env.JUDGE_OFF) && usable(PROVIDER);

let client = null;
async function complete(prompt, mode) {
  const maxTokens = mode === "score" ? 200 : 900;
  if (PROVIDER === "openai") {
    const { default: OpenAI } = await import("openai");
    client = client || new OpenAI();
    const res = await client.chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      ...(MODEL.startsWith("gpt-5") ? { reasoning_effort: "minimal", max_completion_tokens: maxTokens + 400 } : { max_tokens: maxTokens }),
    });
    return res.choices[0].message.content || "";
  }
  if (PROVIDER === "gemini") {
    const { GoogleGenAI } = await import("@google/genai");
    client = client || new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY });
    const res = await client.models.generateContent({
      model: MODEL,
      contents: prompt,
      config: { responseMimeType: "application/json", maxOutputTokens: maxTokens, temperature: 0.2 },
    });
    return res.text || "";
  }
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  client = client || new Anthropic();
  const msg = await client.messages.create({
    model: MODEL,
    max_tokens: maxTokens,
    messages: [{ role: "user", content: prompt }],
  });
  if (msg.stop_reason === "refusal") throw new Error("The model declined to judge this.");
  return msg.content.filter((b) => b.type === "text").map((b) => b.text).join("");
}

/** Why a model call failed, in terms the game can act on. */
function failure(e) {
  const status = Number((e && (e.status || (e.response && e.response.status))) || 0);
  const msg = String((e && e.message) || "").toLowerCase();
  if (status === 402 || /credit|billing|insufficient|payment required|spend limit|usage limit/.test(msg)) return "credit";
  if (status === 429 || /rate limit|quota|resource_exhausted|too many requests/.test(msg)) return "busy";
  return "error";
}

async function sha(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 40);
}

const reply = (status, body) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
const resting = (why) => reply(503, { error: "The AI judge is resting. The answer key and the robot are judging instead.", reason: why });

export default async (req, context) => {
  if (req.method !== "POST") return reply(405, { error: "POST a JSON body" });
  let body;
  try {
    body = await req.json();
  } catch {
    return reply(400, { error: "Send JSON" });
  }
  let request;
  try {
    request = buildRequest(body);
  } catch (e) {
    return reply(e instanceof BadRequest ? 400 : 500, { error: e.message });
  }

  // 1. The answer key: free, no storage needed.
  const free = answerKey(request);
  if (free) return reply(200, free);
  if (!switchedOn()) return resting("off");

  // Tester passes: friends and family get AI verdicts, strangers get the free judges.
  const passes = (process.env.JUDGE_PASSCODE || "").split(",").map((s) => s.trim()).filter(Boolean);
  if (passes.length && !passes.includes((req.headers.get("x-judge-pass") || "").trim())) {
    return reply(403, { error: "The AI judge is open to invited testers for now; the answer key and the robot are judging.", reason: "pass" });
  }

  let store = null;
  try {
    store = getStore("judge");
  } catch { /* no Blobs outside Netlify: no cache or caps */ }

  // 2. The shared cache of earlier verdicts.
  const key = "cache/" + (await sha([PROMPT_VERSION, PROVIDER, MODEL, cacheText(request)].join("\n")));
  try {
    const hit = store && (await store.get(key, { type: "json" }));
    if (hit) return reply(200, { ...hit, cached: true });
  } catch { /* cache is best effort */ }

  // 3. Caps: whole site per day and month, and per visitor per day.
  const now = new Date().toISOString();
  const ip = (context && context.ip) || req.headers.get("x-nf-client-connection-ip") || "unknown";
  const counters = {
    day: "count/" + now.slice(0, 10),
    month: "month/" + now.slice(0, 7),
    visitor: "visitor/" + now.slice(0, 10) + "/" + (await sha(now.slice(0, 10) + ip)),
  };
  const used = { day: 0, month: 0, visitor: 0 };
  if (store) {
    try {
      await Promise.all(Object.keys(counters).map(async (k) => { used[k] = Number((await store.get(counters[k], { type: "text" })) || 0); }));
    } catch { /* counts are best effort */ }
  }
  if (used.day >= CAPS.day || used.month >= CAPS.month) return resting("cap");
  if (used.visitor >= CAPS.visitor) return reply(429, { error: "You've used today's AI verdicts. The answer key and the robot are judging instead.", reason: "visitor" });

  let out;
  try {
    out = await judge(body, complete);
  } catch (e) {
    const why = failure(e);
    console.error("judge failed:", why, (e && e.status) || "", (e && e.message) || e);
    if (why === "credit") return resting("credit");
    if (why === "busy") return reply(429, { error: "The AI judge is busy. Try again in a minute.", reason: "busy" });
    return reply(502, { error: "The AI judge couldn't answer just now.", reason: "error" });
  }
  out = { ...out, model: MODEL };
  if (store) {
    try {
      await Promise.all([
        ...Object.keys(counters).map((k) => store.set(counters[k], String(used[k] + 1))),
        store.setJSON(key, out),
      ]);
    } catch { /* best effort */ }
  }
  return reply(200, out);
};

export const config = {
  path: "/api/judge",
  method: "POST",
  rateLimit: { windowLimit: 20, windowSize: 60, aggregateBy: ["ip", "domain"] },
};
