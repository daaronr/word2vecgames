// Build "What Is It?" into self-contained pages.
//
//   node whatisit/tools/build.mjs            -> whatisit/site/index.html     (Netlify / any static host)
//                                              whatisit/site/artifact.html  (claude.ai Artifact body)
//   SHARE_URL=https://... node whatisit/tools/build.mjs   (link added to the Daily's share text)
//
// Everything (styles, scripts, content, the Anthropic SDK for the API-key judge) is inlined, so the
// Netlify page is one HTML file. The robot judge's word vectors (site/judge-vectors.bin, made by
// tools/build_vectors.py) are fetched when first needed: next to the page if it is there, else from
// the GitHub repository.
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => readFileSync(join(ROOT, p), "utf8");
const RAW = "https://raw.githubusercontent.com/daaronr/word2vecgames";
const BRANCH = process.env.WII_BRANCH || "claude/quirky-brahmagupta-l9gamm";

// ---------- content ----------
const REQUIRED = ["id", "cat", "prompt", "ask", "truth", "key", "clues", "decoys", "rating"];
const CATS = new Set(["web", "brand", "plate", "patent", "search", "double"]);
const items = [];
for (const f of readdirSync(join(ROOT, "content")).filter((f) => f.endsWith(".json")).sort()) {
  for (const it of JSON.parse(read("content/" + f))) {
    if (it.status === "unverified-draft") continue;
    const missing = REQUIRED.filter((k) => it[k] === undefined || it[k] === "");
    if (missing.length) throw new Error(`${f}: ${it.id || it.prompt} is missing ${missing.join(", ")}`);
    if (!CATS.has(it.cat)) throw new Error(`${f}: ${it.id} has unknown category ${it.cat}`);
    if (!Array.isArray(it.key) || !it.key.length) throw new Error(`${f}: ${it.id} needs key words`);
    items.push(it);
  }
}
const ids = new Set();
for (const it of items) {
  if (ids.has(it.id)) throw new Error("Duplicate id " + it.id);
  ids.add(it.id);
}
const contentJson = JSON.stringify(items).replace(/</g, "\\u003c");

// ---------- code ----------
const css = read("src/style.css");
const judge = read("src/judge.js") + "\n" + read("src/pick.js");
const appParts = ["core", "daily", "party", "byo", "bluff", "app"].map((n) => `// ---- ${n}.js ----\n` + read(`src/${n}.js`));
const app = `(function () {\n${appParts.join("\n")}\n})();`;
const sdk = read("vendor/anthropic-sdk-0.131.0.min.mjs");
for (const bad of ["</script", "<!--"]) {
  if (sdk.includes(bad) || app.includes(bad) || judge.includes(bad)) throw new Error("Script text contains " + bad);
}

const FONTS = "https://fonts.googleapis.com/css2?family=Bungee&family=IBM+Plex+Mono:wght@400;500;600&family=Oswald:wght@600&family=Public+Sans:wght@400;600;700;800&display=swap";
const TITLE = "What Is It?";
const DESC = "Guess what a web address, brand, licence plate or patent really is, then find out. Four trial versions: Daily five, Party board, Bring your own, Bluff.";

const config = (env) => {
  const c = {
    env,
    vectorUrls: env === "artifact" ? ["judge-vectors.wasm"] : [
      "judge-vectors.bin",
      `${RAW}/main/whatisit/site/judge-vectors.bin`,
      `${RAW}/${BRANCH}/whatisit/site/judge-vectors.bin`,
    ],
    shareUrl: env === "artifact" ? "" : process.env.SHARE_URL || "",
  };
  return `<script>const CONFIG = ${JSON.stringify(c)};</script>`;
};

const forms = ["wii-rating", "wii-trial", "wii-suggestion", "wii-judge"].map((name) =>
  `<form name="${name}" data-netlify="true" netlify-honeypot="bot-field" hidden><input name="bot-field"><input name="kind"><input name="rid"><input name="id"><input name="stars"><textarea name="text"></textarea><textarea name="data"></textarea></form>`
).join("\n");

const body = (env) => [
  `<div id="app"><p style="padding:24px;font-family:system-ui,sans-serif">Loading What Is It?…</p></div>`,
  env === "netlify" ? forms : "",
  `<script type="application/json" id="wii-content">${contentJson}</script>`,
  env === "netlify" ? `<script type="text/plain" id="anthropic-sdk-src">${sdk}</script>` : "",
  config(env),
  `<script>\n${judge}\n</script>`,
  `<script>\n${app}\n</script>`,
].filter(Boolean).join("\n");

const netlify = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>${TITLE}</title>
<meta name="description" content="${DESC}">
<meta property="og:title" content="${TITLE}">
<meta property="og:description" content="${DESC}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="${FONTS}">
<style>
${css}
</style>
</head>
<body>
${body("netlify")}
</body>
</html>
`;

const artifact = `<title>${TITLE}</title>
<link rel="stylesheet" href="${FONTS}">
<style>
${css}
</style>
${body("artifact")}
`;

writeFileSync(join(ROOT, "site/index.html"), netlify);
writeFileSync(join(ROOT, "site/artifact.html"), artifact);
const byCat = {};
for (const it of items) byCat[it.cat] = (byCat[it.cat] || 0) + 1;
console.log(`site/index.html ${(netlify.length / 1024).toFixed(0)} KB, site/artifact.html ${(artifact.length / 1024).toFixed(0)} KB, ${items.length} items`, byCat);
