Common-sense word set for Word Bocce

vectors.bin, vocab.txt and pools.json in this folder are derived from
ConceptNet Numberbatch 19.08 (English), by Robyn Speer, Joshua Chin and
Catherine Havasi ("ConceptNet 5.5: An Open Multilingual Graph of General
Knowledge", AAAI 2017), https://github.com/commonsense/conceptnet-numberbatch

Numberbatch is licensed under Creative Commons Attribution-ShareAlike 4.0
(CC BY-SA 4.0), https://creativecommons.org/licenses/by-sa/4.0/
This derived bundle (21,114 everyday English words, quantised to int8) is
shared under the same licence.

Rebuild (see the repository README):
  python tools/build_web_data.py numberbatch-en-19.08.txt.gz \
      --vocab-from web/data/vocab.txt --everyday 0.9 --lists embeddings --out web/data-sense
