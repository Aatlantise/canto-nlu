# Sentiment analysis

This is the sentiment analysis subset of CantoNLU,
adapted from the Cantonese subset of SiniticMTError
([toastynews/openrice-senti](https://github.com/toastynews/openrice-senti)),
a dataset of scraped Hong Kong restaurant reviews.

## Dataset info
The dataset consists of 11997 examples: 9999 for train, 999 for validation and testing.
The label space is as follows:
* smile
* ok
* cry

## Requirements
Download .tsv files from [toastynews/openrice-senti](https://github.com/toastynews/openrice-senti),
then run
`python3 convert_to_jsonl.py`.

## Artifacts
A total of 3 .jsonl files, one for each split.

## Training and eval
Use `evaluate_sentiment.py`.
