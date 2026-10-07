// One model call for the AI judge, shared by the Netlify function and eval/run_models.mjs.
//
// Keys and base URLs come from the environment, the way Netlify AI Gateway injects them
// (ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL, OPENAI_*, GEMINI_API_KEY + GOOGLE_GEMINI_BASE_URL), or
// from your own provider keys when no base URL is set.

export const PROVIDERS = {
  gemini: { keys: ["GEMINI_API_KEY", "GOOGLE_API_KEY"], gateway: "GOOGLE_GEMINI_BASE_URL", model: "gemini-3.1-flash-lite" },
  anthropic: { keys: ["ANTHROPIC_API_KEY"], gateway: "ANTHROPIC_BASE_URL", model: "claude-haiku-4-5" },
  openai: { keys: ["OPENAI_API_KEY"], gateway: "OPENAI_BASE_URL", model: "gpt-5-nano" },
};

// US dollars per million tokens (input, output), standard tier, checked 2026-10-07. Netlify AI
// Gateway bills these at 180 credits per dollar. Used only to report and cap spending.
export const PRICES = {
  "gemini-2.5-flash-lite": [0.1, 0.4],
  "gemini-3.1-flash-lite": [0.25, 1.5],
  "gemini-3.5-flash-lite": [0.3, 2.5],
  "claude-haiku-4-5": [1, 5],
  "gpt-5-nano": [0.05, 0.4],
};
export const CREDITS_PER_DOLLAR = 180;

/** Netlify credits for one call's usage ({input, output} tokens); unknown models count as Haiku. */
export function credits(model, usage) {
  const [i, o] = PRICES[model] || PRICES["claude-haiku-4-5"];
  return ((usage.input * i + usage.output * o) / 1e6) * CREDITS_PER_DOLLAR;
}

const clients = {};

/** Ask the model; returns {text, usage: {input, output}}. mode "score" gets a short reply budget. */
export async function complete(provider, model, prompt, mode) {
  const maxTokens = mode === "score" ? 200 : 900;
  if (provider === "openai") {
    const { default: OpenAI } = await import("openai");
    clients.openai = clients.openai || new OpenAI();
    const res = await clients.openai.chat.completions.create({
      model,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_object" },
      ...(model.startsWith("gpt-5") ? { reasoning_effort: "minimal", max_completion_tokens: maxTokens + 400 } : { max_tokens: maxTokens }),
    });
    const u = res.usage || {};
    return { text: res.choices[0].message.content || "", usage: { input: u.prompt_tokens || 0, output: u.completion_tokens || 0 } };
  }
  if (provider === "gemini") {
    const { GoogleGenAI } = await import("@google/genai");
    const base = process.env.GOOGLE_GEMINI_BASE_URL;
    clients.gemini =
      clients.gemini ||
      new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY, ...(base ? { httpOptions: { baseUrl: base } } : {}) });
    const res = await clients.gemini.models.generateContent({
      model,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        maxOutputTokens: maxTokens,
        temperature: 0.2,
        // No thinking: it costs output tokens and this is a quick call.
        ...(model.startsWith("gemini-2.5") ? { thinkingConfig: { thinkingBudget: 0 } } : { thinkingConfig: { thinkingLevel: "minimal" } }),
      },
    });
    const u = res.usageMetadata || {};
    return { text: res.text || "", usage: { input: u.promptTokenCount || 0, output: (u.candidatesTokenCount || 0) + (u.thoughtsTokenCount || 0) } };
  }
  const { default: Anthropic } = await import("@anthropic-ai/sdk");
  clients.anthropic = clients.anthropic || new Anthropic();
  const msg = await clients.anthropic.messages.create({ model, max_tokens: maxTokens, messages: [{ role: "user", content: prompt }] });
  if (msg.stop_reason === "refusal") throw new Error("The model declined to judge this.");
  const u = msg.usage || {};
  return { text: msg.content.filter((b) => b.type === "text").map((b) => b.text).join(""), usage: { input: u.input_tokens || 0, output: u.output_tokens || 0 } };
}
