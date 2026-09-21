# Dependency parsing

This is the dependency parsing (DEPS) subset of CantoNLU,
built from the Cantonese Universal Dependencies treebank
([UD_Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK)).

## Dataset info
The dataset consists of 1004 sentences: 903 for train, 101 for test
(a 90/10 split, taken in file order without shuffling).
Each example is a token sequence paired with a dependency relation and a
head index (1-based, into `tokens`; `0` marks the root) for every token, e.g.
```json
{"id": 0, "tokens": ["你", "喺度", "搵", "乜嘢", "呀", "？"], "deprels": ["nsubj", "advmod", "root", "obj", "discourse:sp", "punct"], "heads": [3, 3, 0, 3, 3, 3]}
```
Labels are [Universal Dependencies relation types](https://universaldependencies.org/u/dep/).

## Requirements
Run `python3 convert_deps.py`. The script downloads the source CoNLL-U file
(`yue_hk-ud-test.conllu`) from the UD_Cantonese-HK `dev` branch into this
directory if it is not already present, then parses it.

## Artifacts
`deps_train.jsonl` and `deps_test.jsonl`.

## Note
`pos` (`../pos`) is built from the same source CoNLL-U file with the same
split, so `deps_{split}.jsonl` and `pos_{split}.jsonl` share `id` and `tokens`
for corresponding rows.
