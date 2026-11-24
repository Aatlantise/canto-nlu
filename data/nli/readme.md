# Natural language inference

This is the natural language inference (NLI) subset of CantoNLU: binary
sentence-pair classification into
* `entailment`
* `not_entailment`

## Data sources
* **Train:** the `train` split of [`hon9kon9ize/yue-all-nli`](https://huggingface.co/datasets/hon9kon9ize/yue-all-nli)
  (SNLI + MNLI machine-translated into Cantonese by Gemini), downloaded by `download.py`
  to `data/yue-nli-local`. Each triplet (anchor, positive/entailing, negative/contradicting
  hypothesis) is unrolled into two pairs: `entailment` and `not_entailment`.
* **Validation / test:** [`aatlantise/canto-xnli`](https://huggingface.co/datasets/aatlantise/canto-xnli)
  (Canto-XNLI; 2,490 validation and 5,010 test pairs), downloaded by `download.py` to
  `data/canto-xnli-local` at a pinned commit. It is the [XNLI](https://github.com/facebookresearch/XNLI)
  (Conneau et al., 2018) dev and test sets, translated from English into Cantonese by Fable
  (same rows, order and labels as XNLI), with the original 3-way labels
  (`entailment` / `neutral` / `contradiction`, balanced 1:1:1). Fields: `pairID`, `genre`,
  `label`, and the Cantonese text (`yue_premise` / `yue_hypothesis`). License: CC BY-NC 4.0.

Because the training data has no neutral pairs, `neutral` is **dropped** from validation
and test, and `contradiction` maps to `not_entailment`. This leaves 1,660 validation and
3,340 test pairs, balanced 50/50.

```json
{"pairID": "4", "genre": "facetoface", "label": "contradiction", "yue_premise": "嗯，我根本冇諗過嗰樣嘢，但係我好忟，搞到最後我又同佢傾咗偈。", "yue_hypothesis": "我仲未再同佢傾過偈。"}
```

## Sample
Run `python3 sample_nli.py` (requires `datasets`). It draws 50 `entailment` + 50
`not_entailment` pairs from the Canto-XNLI test split at the same pinned commit
(neutral excluded, fixed seed 42) and writes
`sample_nli.jsonl` and `nli_sample.csv`: the same 100 examples, one per format, with
`id` = the source `pairID`.
