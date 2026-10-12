# Language identification

This is the language identification (LID) subset of CantoNLU: a 3-way
classification task distinguishing sentences in Cantonese, Mandarin, and others (code-mixed sentences/sentences acceptable in both languages), sampled from a manually constructed grammar.

## Dataset info
Sentences are sampled from a manually constructed, context-free grammar of Mandarin and Cantonese. The grammar consists of two types of productions: grammar rules and lexical items. Each rule contains a sequence of symbols, which may be expanded or may define a fixed word. Beginning from a random sentence rule, symbols are recursively expanded until a well-formed sentence is obtained (or a dead end is reached, where the sentence is discarded and a new sentence is sampled). Each production is labelled as either `canto` (exclusive to Cantonese), `mando` (exclusive to Mandarin), or `shared` (shared by both languages).

Each example is a single sentence with a 3-way label:
* `0`: Cantonese sentence, comprising some `canto` productions and no `mando` productions.
* `1`: Mandarin sentence, comprising some `mando` productions and no `canto` productions.
* `2`: Sentences containing both `canto` and `mando` productions, or neither.

### Example sentence
`嗰個警察去開嗰間茶餐廳嘅`
"That police officer usually goes to that cha chaan teng (a type of restaurant)."

**Derivation:**
```
S → NP 去開 嗰間 PLACE 嘅 (canto)
    NP → DEM 個 N_person (shared)
        DEM → 嗰 (canto)
        N_person → 警察 (shared)
    PLACE → 茶餐廳 (canto)
```
Resulting label: `0` (canto)


## Requirements
Run `python3 grammar.py [--n_train 10000] [--n_test 2000] [--start S]`.
The script requires only the Python standard library.

## Artifacts
`lid_train.jsonl` and `lid_test.jsonl`, with sizes as defined by the `--n_train` and `--n_test` flags. All sentences in the test set are constructed using at least one held out rule (defined in `HOLDOUT_RULES`) and at least one held out lexical item (defined in `HOLDOUT_LEX`). Conversely, sentences in the train set do not contain held out rules nor held out lexical items.

`sample_lid.py` extracts a class-balanced sample (`lid_sample.csv`, 101 rows)
from `lid_test.jsonl` for spot-checking/human annotation.
