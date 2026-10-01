AI tokens word set for Word Bocce

vectors.bin, vocab.txt and pools.json in this folder are derived from the
token embedding table (wte) of GPT-2 small, by OpenAI (Radford et al.,
"Language Models are Unsupervised Multitask Learners", 2019),
https://github.com/openai/gpt-2, modified MIT licence (Copyright (c) 2019
OpenAI). Weights fetched from https://huggingface.co/openai-community/gpt2.

The bundle keeps 49,745 of GPT-2's 50,257 tokens (no pieces of characters,
control characters or blocklisted words), mean-centred, reduced to 128 of 768
dimensions with PCA, and quantised to int8. vocab.txt holds each token's label
as web/tokens.js shows it ("␣" = a leading space).

Rebuild: python3 tools/build_token_data.py
