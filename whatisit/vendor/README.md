# Vendored code

`anthropic-sdk-0.131.0.min.mjs` is the official Anthropic TypeScript SDK (`@anthropic-ai/sdk`
0.131.0, MIT licence, see `anthropic-sdk-LICENSE`) plus its `helpers/json-schema` module, bundled
for the browser. It is inlined into `site/index.html` and loaded only when a player sets their own
API key for the AI judge. Rebuild:

```bash
npm install @anthropic-ai/sdk@0.131.0 esbuild
printf 'export { default as Anthropic } from "@anthropic-ai/sdk";\nexport { jsonSchemaOutputFormat } from "@anthropic-ai/sdk/helpers/json-schema";\n' > entry.mjs
npx esbuild entry.mjs --bundle --format=esm --minify --platform=browser --target=es2020 --legal-comments=none --outfile=anthropic-sdk-0.131.0.min.mjs
```
