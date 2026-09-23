# Part-of-speech tagging

This is the part-of-speech (POS) tagging subset of CantoNLU,
built from the Cantonese Universal Dependencies treebank
([UD_Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK)).

## Dataset info
The dataset consists of 1004 sentences: 903 for train, 101 for test
(a 90/10 split, taken in file order without shuffling).
Each example is a token sequence paired with its UPOS tag sequence, e.g.
```json
{"id": 0, "tokens": ["你", "喺度", "搵", "乜嘢", "呀", "？"], "upos": ["PRON", "ADV", "VERB", "PRON", "PART", "PUNCT"]}
```
Labels are the standard [Universal POS tags](https://universaldependencies.org/u/pos/).

## Requirements
Run `python3 convert_pos.py`. The script downloads the source CoNLL-U file
(`yue_hk-ud-test.conllu`) from the UD_Cantonese-HK `dev` branch into this
directory if it is not already present, then parses it.

## Artifacts
`pos_train.jsonl` and `pos_test.jsonl`.

There is no sample .csv file here; instead, use the first ten sentences to annotate on an
annotation interface like [Arborator](https://gucorpling.org/arborator/q.cgi).

## Note
`deps` (`../deps`) is built from the same source CoNLL-U file with the same
split, so `pos_{split}.jsonl` and `deps_{split}.jsonl` share `id` and `tokens`
for corresponding rows.