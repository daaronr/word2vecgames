/*
 * GPT-2's tokenizer (byte-level BPE), for the Tokens page and the token view of the court.
 * Runs in the browser (window.BocceTokens) and in Node (module.exports), like engine.js.
 *
 *   text → pieces (split at spaces, digits and punctuation; a piece keeps its leading space)
 *        → UTF-8 bytes → merges, applied in the order GPT-2 learned them → token IDs
 *
 * The IDs follow from the merge list alone: 0–255 are single bytes, 256 + i is merge i, and
 * 50256 is <|endoftext|>. So tokens/gpt2-merges.txt (OpenAI's vocab.bpe, MIT) is all we ship.
 */
(function (root) {
  "use strict";

  // GPT-2's pre-split: contractions, words, numbers, punctuation runs, whitespace.
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  // The merges file writes each byte as a printable character (a space is "Ġ", a newline "Ċ").
  const BYTE_CHAR = [], CHAR_BYTE = new Map();
  (() => {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    bs.forEach((b, i) => { BYTE_CHAR[b] = String.fromCharCode(cs[i]); CHAR_BYTE.set(BYTE_CHAR[b], b); });
  })();
  // ...and the token IDs for single bytes follow that same order.
  const BYTE_ORDER = [...CHAR_BYTE.keys()];

  const utf8 = new TextEncoder();
  const strict = new TextDecoder("utf-8", { fatal: true });
  const hex = (b) => b.toString(16).toUpperCase().padStart(2, "0");

  class Tokenizer {
    constructor(mergesText) {
      const lines = mergesText.split("\n").filter((l) => l && !l.startsWith("#version"));
      this.ranks = new Map();
      this.pieces = BYTE_ORDER.slice(); // id → token in the merges file's spelling
      lines.forEach((l, i) => {
        const [a, b] = l.split(" ");
        this.ranks.set(a + " " + b, i);
        this.pieces.push(a + b);
      });
      this.pieces.push("<|endoftext|>");
      this.ids = new Map(this.pieces.map((p, i) => [p, i]));
      this.size = this.pieces.length; // 50,257 for GPT-2
      this.cache = new Map();
    }

    /** Split one pre-split piece (in byte characters) into tokens, merging the best-ranked pair first. */
    bpe(word) {
      if (this.cache.has(word)) return this.cache.get(word);
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1, bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; best = i; }
        }
        if (best < 0) break;
        const a = parts[best], b = parts[best + 1], out = [];
        for (let i = 0; i < parts.length;) {
          if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) { out.push(a + b); i += 2; }
          else out.push(parts[i++]);
        }
        parts = out;
      }
      if (this.cache.size < 20000) this.cache.set(word, parts);
      return parts;
    }

    /** Token IDs for `text`. */
    encode(text) {
      const out = [];
      for (const m of String(text).matchAll(PAT)) {
        const word = Array.from(utf8.encode(m[0]), (b) => BYTE_CHAR[b]).join("");
        for (const p of this.bpe(word)) out.push(this.ids.get(p));
      }
      return out;
    }

    bytes(id) { return Uint8Array.from(Array.from(this.pieces[id] || ""), (c) => CHAR_BYTE.get(c)); }

    /** Text from token IDs (pieces of a character are joined back up before decoding). */
    decode(ids) {
      const all = [];
      for (const id of ids) all.push(...this.bytes(id));
      return new TextDecoder().decode(Uint8Array.from(all));
    }

    /**
     * How to show one token: spaces as "␣", newlines as "↵", tabs as "⇥". A token that is only part
     * of a character (an emoji is 4 bytes; GPT-2 often stores it as 2 or 3 tokens) shows its bytes.
     * `partial` is true for those.
     */
    label(id) {
      if (id === this.size - 1) return { text: "<|endoftext|>", partial: false };
      const bytes = this.bytes(id);
      let s;
      try { s = strict.decode(bytes); } catch (e) {
        const lead = bytes.findIndex((b) => b !== 0x20);
        return { text: "␣".repeat(lead) + "‹" + Array.from(bytes.subarray(lead), hex).join(" ") + "›", partial: true };
      }
      return { text: s.replace(/ /g, "␣").replace(/\n/g, "↵").replace(/\t/g, "⇥").replace(/\r/g, "␍"), partial: false };
    }

    /** Tokens with their labels, for showing: [{ id, text, partial }]. */
    tokens(text) { return this.encode(text).map((id) => ({ id, ...this.label(id) })); }
  }

  async function load(url = "tokens/gpt2-merges.txt") {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.url}: HTTP ${r.status}`);
    return new Tokenizer(await r.text());
  }

  const api = { Tokenizer, load };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.BocceTokens = api;
})(typeof self !== "undefined" ? self : this);
