# Linguistic acceptability judgment

This is the linguistic acceptability judgment (LAJ) subset of CantoNLU,
adapted from the Cantonese subset of SiniticMTError
([Liu et al. 2026](https://aclanthology.org/2026.lrec-1.683/)),
a dataset of machine translation error annotations.

## Dataset info
The dataset consists of 1402 comparison examples (which sentence is more acceptable?)
or 2804 single-sentence examples (is this sentence acceptable or not?).
The label space is as follows:
* `0`: not acceptable
* `1`: acceptable

## Requirements
Requires a [cantonese.jsonl](https://github.com/hannliu/SiniticMTError/blob/main/cantonese.jsonl) file.
Download the file, place it in this directory and run `build_laj.sh`.

## Artifacts
* An intermediate full-dataset file `laj_dataset.jsonl`
* Test and train splits and a human annotation sample for CoLA-style ([Warstadt et al. 2019](https://aclanthology.org/Q19-1040/)) fine-tuned evaluation in `lab_finetune`
* Test split for BLiMP-style and a human annotation sample for BLiMP-style ([Warstadt et al. 2020](https://aclanthology.org/2020.tacl-1.25/)) comparison evaluation in `laj_comparison`
