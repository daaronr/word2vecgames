// POST /api/judge: the game's AI judge, paid for by the site, so players need no AI account.
//
// Runs through Netlify AI Gateway: Netlify injects the provider keys (ANTHROPIC_API_KEY and
// ANTHROPIC_BASE_URL, OPENAI_*, GEMINI_*) and bills the calls to the team's Netlify credits. Set your
// own provider key in the site's environment variables instead and the provider bills you directly.
//
// Environment variables (all optional):
//   JUDGE_PROVIDER   anthropic (default) | openai | gemini
//   JUDGE_MODEL      default per provider below; pick one Netlify's AI Gateway lists
//   JUDGE_DAILY_CAP  most model calls per UTC day before the site falls back to the free judge (default 300)
//
// Guards: 20 requests a minute per visitor, a daily cap, and a cache so the same question is only
// paid for once.
import { getStore } from "@netlify/blobs";
import { BadRequest, buildRequest, judge } from "../lib/judge-core.mjs";

const PROVIDER = (process.env.JUDGE_PROVIDER || "anthropic").toLowerCase();
const DEFAULT_MODEL = { anthropic: "claude-haiku-4-5", openai: "gpt-5-nano", gemini: "gemini-3.1-flash-lite" };
const MODEL = process.env.JUDGE_MODEL || DEFAULT_MODEL[PROVIDER] || DEFAULT_MODEL.anthropic;
const DAILY_CAP = Number(process.env.JUDGE_DAILY_CAP || 300);

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
    client = client || new GoogleGenAI({});
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

async function sha(text) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 40);
}

const reply = (status, body) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });

export default async (req) => {
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

  let store = null;
  try {
    store = getStore("judge");
  } catch { /* no Blobs outside Netlify: no cache or cap */ }
  const key = "cache/" + (await sha(PROVIDER + MODEL + "\n" + request.prompt));
  try {
    const hit = store && (await store.get(key, { type: "json" }));
    if (hit) return reply(200, { ...hit, cached: true });
  } catch { /* cache is best effort */ }

  const day = "count/" + new Date().toISOString().slice(0, 10);
  let used = 0;
  try {
    used = store ? Number((await store.get(day, { type: "text" })) || 0) : 0;
  } catch { /* count is best effort */ }
  if (used >= DAILY_CAP) return reply(503, { error: "The AI judge has had its fill for today. The free judge is standing in." });

  let out;
  try {
    out = await judge(body, complete);
  } catch (e) {
    console.error("judge failed:", e && e.message);
    return reply(502, { error: "The AI judge couldn't answer just now." });
  }
  out = { ...out, model: MODEL };
  try {
    if (store) await Promise.all([store.set(day, String(used + 1)), store.setJSON(key, out)]);
  } catch { /* best effort */ }
  return reply(200, out);
};

export const config = {
  path: "/api/judge",
  method: "POST",
  rateLimit: { windowLimit: 20, windowSize: 60, aggregateBy: ["ip", "domain"] },
};
