GPT-2 tokenizer data for Word Bocce's Tokens tab

gpt2-merges.txt is OpenAI's GPT-2 byte-pair merge list (vocab.bpe, 50,000
merges), from https://github.com/openai/gpt-2 (Copyright (c) 2019 OpenAI,
modified MIT licence), taken unchanged from the gpt-3-encoder npm package
(MIT). web/tokens.js rebuilds GPT-2's token IDs from it: 0-255 are single
bytes, 256 + i is merge i, and 50256 is <|endoftext|>.

quiz.json holds the "Guess the split" rounds. Each lists the pieces its text
splits into; tests/tokens.test.js checks them against the tokenizer.
