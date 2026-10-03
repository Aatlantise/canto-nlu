# Tokenizer analysis: Cantonese vs. Mandarin

This document was drafted by Claude.

How well do the tokenizers used in this work represent Cantonese, compared with Mandarin
(standard written Chinese, SWC)?

| Tokenizer | Source | Type |
|---|---|---|
| bert-base-chinese | `google-bert/bert-base-chinese` | WordPiece, character-level for CJK |
| bert-base-cantonese | `hon9kon9ize/bert-base-cantonese` | bert-base-chinese + 500 added tokens |
| chinese-modernbert-wwm | `feynmanzhao/chinese-modernbert-large-wwm` | BPE, multi-character words |
| yue-tokenizer (ours) | `models/yue-tokenizer` | Unigram, 20k, byte fallback |

Run from the repo root: `USE_TF=0 python3 appendix/tokenizer_analysis/analyze_tokenizers.py`.
External inputs are downloaded to `resources/` (gitignored) on first run. All numbers come from
the CSVs in this folder.

## Setup

* **Character classes** come from the [cantonesedetect](https://github.com/CanCLID/cantonesedetect)
  lists (MIT): 68 Cantonese-only characters, 9 Cantonese sentence-final particles, 26 SWC-only
  characters. Cantonese characters are weighted by [words.hk](https://words.hk/faiman/analysis/)
  frequency, SWC ones by UD Chinese-HK frequency.
* **Running text:** [UD Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK) and
  [UD Chinese-HK](https://github.com/UniversalDependencies/UD_Chinese-HK) (r2.18, CC BY-SA 4.0). They are sentence-aligned translations, 1,004 sentences each, both in
  traditional characters. Each is also run in simplified characters (`hanziconv`) to separate a
  script gap from a dialect gap. OpenRice reviews (`data/sentiment`) are in
  `corpus_oov_fertility.csv`.
* **OOV** means `[UNK]` or byte-fallback tokens, since ours has no `[UNK]`.
* **Function words:** the 115 types per treebank tagged PART, AUX, PRON, DET, ADP, SCONJ or
  CCONJ with at least 3 occurrences, evaluated on gold UD word spans. 52 of the Cantonese ones
  never appear in Chinese-HK (嘅 佢 我哋 咗 喺 嗰 㗎 喎 畀 嚟 冇 …) and are called
  *Cantonese-specific*.

## Summary

| | Cantonese-only chars covered* | OOV %, UD Cantonese-HK | OOV %, UD Chinese-HK | Tokens per char (Cantonese) | Multi-char function words as one token |
|---|---|---|---|---|---|
| bert-base-chinese | 98.3% | 2.05 | 0.04 | 0.99 | 0% |
| bert-base-cantonese | 100.0% | 0.10 | 0.01 | 0.99 | 0% |
| chinese-modernbert-wwm | **25.9%** | **10.86** | 1.08 | 0.91 | 28.9% |
| yue-tokenizer (ours) | 98.3% | 0.36 | 0.46 | **0.71** | **94.1%** |

\* Weighted by words.hk frequency. By type the figures are much lower (30.9, 82.4, 2.9, 89.7%)
because the list includes rare and supplementary-plane characters (𨳍 𨳊 𨋢 …).
Source: `char_class_coverage.csv`, `corpus_oov_fertility.csv`, `function_words.csv`.

## Findings

1. **bert-base-chinese covers everyday Cantonese but has a tail of colloquial gaps.**
   The frequent characters are there, but 噉 is its most common OOV (161 times in UD
   Cantonese-HK) and the particles 囖 唻 嘑 噃 啩 嗱 are all `[UNK]`. Result: 2.05% of
   Cantonese characters are OOV, vs. 0.04% on the parallel Mandarin.

2. **bert-base-cantonese's 500 added tokens target exactly those gaps** (噉 嬲 嗱 嗌 掟 閂 戇
   撳 冧 …, `bert_base_cantonese_added_tokens.csv`), bringing Cantonese OOV to 0.10%. It is
   otherwise bert-base-chinese: character by character, 0.99 tokens per character.

3. **chinese-modernbert-wwm has a real Cantonese gap, not just a script gap.**
   * It covers 25.9% of Cantonese-only character tokens and 0.7% of particle tokens.
   * Same sentences, same script: 10.9% OOV for Cantonese vs. 1.1% for Mandarin. In
     simplified it is 9.5% vs. 0.1%, because 佢 咗 喺 啲 嗰 嚟 嘢 咁 哋 㗎 喎 have no
     simplified form, so converting fixes Mandarin but not Cantonese.
   * 52.7% of Cantonese-specific function-word occurrences are `[UNK]`: pronouns, aspect
     markers and sentence-final particles are invisible to the model.

4. **Ours compresses Cantonese best and keeps non-compositional function words whole.**
   * 0.71 tokens per character on Cantonese and 0.76 on Mandarin, vs. 0.84–0.99 for the
     others, with similar OOV on both (0.36% vs. 0.46%).
   * **Multi-character function words** (乜嘢 但係 同埋 我哋 …), which a character-level
     tokenizer must split, are a single token 94.1% of the time (89.6% for the 29
     Cantonese-specific ones) and are almost never broken up (1.1%). Their misses are mostly
     joins with a neighbor (好多 60% own token, 呢啲 71%, 佢哋 71%).
   * **Single-character function words** are joined with a neighbor 32.5% of the time
     (67% stay on their own), and this includes frequent auxiliaries and pronouns, not just
     particles: 想 11% own token, 會 21%, 畀 24%, 嗰 33%, 係 37%, 咗 45%, 佢 55%, 我 57%. This
     is joining into larger units, not breaking of words. Whether it matters depends on the
     task: probably not for classification, possibly for POS and dependency parsing. Not
     tested yet.

   | Cantonese-HK function words, ours | occ. | own token | broken, edges intact | joined |
   |---|---|---|---|---|
   | Multi-character (50 types) | 837 | 94.1% | 1.1% | 4.8% |
   | Single-character (65 types) | 3,085 | 66.9% | 0.6%* | 32.5% |

   \* Byte fallback for OOV characters, since a single character can't be split.

   **Limitations.**
   * *Traditional only.* Only 99 simplified-only characters are in the vocabulary, so
     simplified input falls back to bytes for 23–29% of CJK characters. This is fine while
     all inputs are traditional, as in every dataset in this repo.
   * *Missing Cantonese spellings.* The vocabulary has 曬 揾 響 but not 晒 搵 响 (or 吓),
     which are byte-encoded. 晒, 搵 and 吓 are among the most common OOVs in UD Cantonese-HK.
     Adding them to the vocabulary or normalizing the input would fix this. Taiwan and
     Mainland spellings are also missing but out of scope.
   * *Mandarin is split up.* Only 56.8% of Mandarin multi-character function words are one
     token, and 42.8% are broken into pieces.
