# Cantonese NLP Resources

---

## Benchmarks

| Resource | Link | Paper | Description |
|---|---|---|---|
| HKCanto-Eval | [github.com/hon9kon9ize/hkeval2025](https://github.com/hon9kon9ize/hkeval2025) | [Cheng et al., 2025 (CoNLL)](https://aclanthology.org/2025.conll-1.1/) | Benchmark for Cantonese language understanding and Hong Kong cultural knowledge in LLMs. Includes a translated MMLU, OpenRice sentiment, and small phonology/orthography sets. |
| Yue-Benchmark | [github.com/jiangjyjy/Yue-Benchmark](https://github.com/jiangjyjy/Yue-Benchmark) | [Jiang et al., 2025 (NAACL Findings)](https://aclanthology.org/2025.findings-naacl.253/) | Benchmark of LLM reasoning, knowledge, and logic in Cantonese (Yue-MMLU, Yue-GSM8K, Yue-ARC-C, Yue-TruthfulQA, etc.), partly translated from other languages. |
| HKMMLU | — | [Cao et al., 2025 (arXiv)](https://arxiv.org/abs/2505.02177) | 26,698 multiple-choice questions over 66 subjects on Hong Kong knowledge, plus 90,550 Mandarin–Cantonese translation items. |
| Cantonese / Japanese / Turkish cross-lingual benchmark | — | [Xia et al., 2025 (arXiv)](https://arxiv.org/abs/2511.10664) | LLM evaluation on open-domain QA, summarization, English→Cantonese translation, and culturally grounded dialogue. |
| Multimodal-Yue-Benchmark | [huggingface.co/datasets/J017athan/Multimodal-Yue-Benchmark](https://huggingface.co/datasets/J017athan/Multimodal-Yue-Benchmark) | — | Spoken version of Yue-Benchmark items (MMLU and GSM8K style) made with TTS in 3 voices; about 15k rows. |
| YUE-PUB-Speech | — | Wen et al., 2026 (Interspeech, to appear) | Speech-based benchmark for Cantonese pragmatic understanding. |
| SIB-200 | [huggingface.co/datasets/Davlan/sib200](https://huggingface.co/datasets/Davlan/sib200) | [Adelani et al., 2024 (EACL)](https://aclanthology.org/2024.eacl-long.14/) | Topic classification in 200+ languages, including a Cantonese portion. |

## Task datasets

| Resource | Link | Paper | Description |
|---|---|---|---|
| SiniticMTError | — | [Liu et al., 2026 (LREC)](https://aclanthology.org/2026.lrec-1.683/) | Machine translation dataset with error-span annotations for Sinitic languages, including Cantonese. |
| yue-all-nli | [huggingface.co/datasets/hon9kon9ize/yue-all-nli](https://huggingface.co/datasets/hon9kon9ize/yue-all-nli) | — | SNLI and MNLI machine-translated into Cantonese as triplets (premise, entailment, contradiction), about 570k examples. Built for sentence embeddings. |
| yue-stsb | [huggingface.co/datasets/hon9kon9ize/yue-stsb](https://huggingface.co/datasets/hon9kon9ize/yue-stsb) | — | STS-B semantic textual similarity, machine-translated into Cantonese. |
| yue-logiqa | [huggingface.co/datasets/hon9kon9ize/yue-logiqa](https://huggingface.co/datasets/hon9kon9ize/yue-logiqa) | — | LogiQA logical reading comprehension, machine-translated into Cantonese (8.7k items, not checked by hand). |
| OpenRice sentiment | [github.com/toastynews/openrice-senti](https://github.com/toastynews/openrice-senti); source site [openrice.com](https://www.openrice.com/en/hongkong) | [Zhang et al., 2011 (Expert Systems with Applications)](https://doi.org/10.1016/j.eswa.2010.12.147) | Hong Kong restaurant reviews in Cantonese with positive/neutral/negative labels, about 12k examples. |
| Cantonese_sentiment | [huggingface.co/datasets/sepidmnorozy/Cantonese_sentiment](https://huggingface.co/datasets/sepidmnorozy/Cantonese_sentiment) | — | About 41.6k Cantonese restaurant reviews with binary sentiment labels (probably from OpenRice). |
| yue-openrice-review / yue-lihkg-topic | [yue-openrice-review](https://huggingface.co/datasets/izhx/yue-openrice-review), [yue-lihkg-topic](https://huggingface.co/datasets/izhx/yue-lihkg-topic) | — | OpenRice review classification and topic classification of LIHKG forum threads (20 categories). |
| Cantonese emotion lexicon | — | [Zhang et al., 2024 (arXiv)](https://arxiv.org/abs/2410.11526) | Cantonese emotion lexicon built with both LLM and human annotation. |
| UD Cantonese-HK | [github.com/UniversalDependencies/UD_Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK) | [Wong et al., 2017 (Depling)](https://aclanthology.org/W17-6530/) | Universal Dependencies treebank of about 1k sentences / 14k tokens, with POS tags and dependency relations (17 relations specific to Cantonese). Part of a Cantonese–Mandarin parallel treebank. |
| Cantonese ParGram treebank | — | [Lam & Uí Dhonnchadha, 2026 (arXiv)](https://arxiv.org/abs/2608.07283); [Lam, 2026 (EMNLP Findings)](https://arxiv.org/abs/2608.23448) | LFG (ParGram) grammar and treebank resources for Cantonese, built with the help of LLMs. |
| PunCantonese | — | [Li et al., 2023 (Interspeech)](https://www.isca-archive.org/interspeech_2023/li23z_interspeech.html) | Benchmark corpus for restoring punctuation in Cantonese speech transcripts. |
| wikipedia-zh-yue-qa / -summaries | [QA](https://huggingface.co/datasets/indiejoseph/wikipedia-zh-yue-qa), [summaries](https://huggingface.co/datasets/indiejoseph/wikipedia-zh-yue-summaries) | — | About 35k question–answer pairs, and article summaries, generated from Cantonese Wikipedia. |
| gsm8k_cantonese | [huggingface.co/datasets/jed351/gsm8k_cantonese](https://huggingface.co/datasets/jed351/gsm8k_cantonese) | — | GSM8K math word problems translated into Cantonese. |
| Cantonese_HarmfulBehaviors | [huggingface.co/datasets/cantonesesra/Cantonese_HarmfulBehaviors](https://huggingface.co/datasets/cantonesesra/Cantonese_HarmfulBehaviors) | — | 416 Cantonese prompts describing harmful behavior, for safety evaluation. |

## Instruction-tuning data

| Resource | Link | Paper | Description |
|---|---|---|---|
| yue-alpaca / yue-alpaca-chat | [yue-alpaca](https://huggingface.co/datasets/hon9kon9ize/yue-alpaca), [yue-alpaca-chat](https://huggingface.co/datasets/hon9kon9ize/yue-alpaca-chat) | — | About 18.6k Alpaca-style Cantonese instructions generated with Gemini Pro (CC BY-NC 4.0). |
| lingnaam-cantonese-cot-qa | [huggingface.co/datasets/CanCLID/lingnaam-cantonese-cot-qa](https://huggingface.co/datasets/CanCLID/lingnaam-cantonese-cot-qa) | — | About 11k question–answer pairs on Lingnan culture, with reasoning chains, across 17 domains. |

## Corpora

| Resource | Link | Paper | Description |
|---|---|---|---|
| Cantonese Wikipedia (zh-yue) | [huggingface.co/datasets/wikimedia/wikipedia (20231101.zh-yue)](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.zh-yue) | — | Wikipedia dump: about 137k articles, 40M characters. |
| Cantonese Sentences | [huggingface.co/datasets/raptorkwok/cantonese_sentences](https://huggingface.co/datasets/raptorkwok/cantonese_sentences) | — | About 30M Cantonese sentences (660M characters), compiled by Kwok (2024). |
| YueData / YueTung-7B | — | [Jiang et al., 2025 (EMNLP Findings)](https://aclanthology.org/2025.findings-emnlp.102/) | Cantonese corpus of 2B+ tokens for LLM pre-training and fine-tuning; contains a lot of non-Cantonese text. YueTung-7B is Qwen-2.5-7B trained further on it. |
| HK Content Corpus | [huggingface.co/datasets/IKMLab-team/hk_content_corpus](https://huggingface.co/datasets/IKMLab-team/hk_content_corpus), [Zenodo](https://zenodo.org/records/16882352) | — | 10.1 GB of cleaned text from 8 Hong Kong sources, including LIHKG, OpenRice, Apple Daily, Stand News, and zh-hk Wikipedia (CC BY 4.0). |
| Cantonese_Common_Crawl_Filtered | [huggingface.co/datasets/jed351/Cantonese_Common_Crawl_Filtered](https://huggingface.co/datasets/jed351/Cantonese_Common_Crawl_Filtered) | — | About 5.6M rows (10.8 GB) of Cantonese text pulled from Common Crawl with a Cantonese detector. |
| LIHKG | [huggingface.co/datasets/AlienKevin/LIHKG](https://huggingface.co/datasets/AlienKevin/LIHKG) | — | Scraped threads from the LIHKG forum (14.4 GB). |
| Hong Kong Cantonese Corpus (HKCanCor) | [github.com/fcbond/hkcancor](https://github.com/fcbond/hkcancor) | [Luke & Wong, 2015 (Journal of Chinese Linguistics)](https://www.jstor.org/stable/26455290) | Transcribed spoken Cantonese (conversations and radio, 1997–98) with segmentation, POS tags, and Jyutping. |
| CHILDES Cantonese (Lee/Wong/Leung) | [childes.talkbank.org](https://childes.talkbank.org/access/Chinese/Cantonese/LeeWongLeung.html) | — | Longitudinal, POS-tagged transcripts of 8 Cantonese-speaking children (171 transcripts). |
| CHILDES Yip/Matthews bilingual corpus | [talkbank.org](https://talkbank.org/childes/access/Biling/YipMatthews.html) | — | Longitudinal transcripts of Cantonese–English bilingual children. |
| BabyBabelLM (Cantonese portion) | — | [Jumelet et al., 2025 (arXiv)](https://arxiv.org/abs/2510.10159) | Multilingual training data meant to resemble what children are exposed to; includes Cantonese. |
| Spoken Cantonese–Written Chinese parallel corpus | — | [Lee, 2011 (IJCNLP)](https://aclanthology.org/I11-1174/) | Hong Kong TV transcripts aligned with their Standard Chinese subtitles. |
| Cantonese–Mandarin parallel corpus (Dai et al.) | — | [Dai et al., 2025 (LoResLM)](https://aclanthology.org/2025.loreslm-1.32/) | Parallel corpus from work on Cantonese-to-Mandarin translation with LLMs. |
| yue_nmt parallel data | [github.com/evelynkyl/yue_nmt](https://github.com/evelynkyl/yue_nmt) | [Liu, 2022 (VarDial)](https://aclanthology.org/2022.vardial-1.4/) | Mandarin–Cantonese parallel sentences, grown to about 36k pairs by mining. |
| Written Chinese–Cantonese NMT corpus | — | [Mak & Lee, 2021 (NLPIR; arXiv 2025)](https://arxiv.org/abs/2505.17816) | About 100k Standard Chinese–Cantonese sentence pairs, from linguistic sources and mined from Wikipedia. |
| cantonese-chinese-parallel-corpus | [huggingface.co/datasets/HKAllen/cantonese-chinese-parallel-corpus](https://huggingface.co/datasets/HKAllen/cantonese-chinese-parallel-corpus) | — | Cantonese–Mandarin parallel sentence pairs. |
| cantonese-mandarin-translations | [huggingface.co/datasets/botisan-ai/cantonese-mandarin-translations](https://huggingface.co/datasets/botisan-ai/cantonese-mandarin-translations) | — | About 24k Cantonese→Simplified Chinese pairs, machine-translated from HKCanCor and Common Voice text. |
| Tatoeba | [tatoeba.org](https://tatoeba.org) | — | Crowdsourced multilingual sentence and translation collection that includes Cantonese. |
| NLLB / FLORES-200 | [github.com/facebookresearch/flores](https://github.com/facebookresearch/flores) | [NLLB Team et al., 2024 (Nature)](https://doi.org/10.1038/s41586-024-07335-x) | Massively multilingual MT project (200 languages, including Cantonese) and its evaluation data. |

## Speech and multimodal

| Resource | Link | Paper | Description |
|---|---|---|---|
| WenetSpeech-Yue | [github.com/ASLP-lab/WenetSpeech-Yue](https://github.com/ASLP-lab/WenetSpeech-Yue) | [Li et al., 2025 (arXiv)](https://arxiv.org/abs/2509.03959) | 21,800 hours of Cantonese speech with rich annotation. Also releases the WSYue-eval ASR/TTS benchmark and Cantonese ASR and TTS models (Whisper, SenseVoice, Conformer, CosyVoice2). |
| MDCC | — | [Yu et al., 2022 (LREC)](https://aclanthology.org/2022.lrec-1.696/) | 73.6 hours of read speech from Hong Kong audiobooks in several domains. The paper also surveys earlier Cantonese ASR datasets. |
| Common Voice (zh-HK / yue) | [commonvoice.mozilla.org](https://commonvoice.mozilla.org/en/datasets) | [Ardila et al., 2020 (LREC)](https://aclanthology.org/2020.lrec-1.520/) | Crowdsourced read speech that includes Hong Kong Cantonese. The usual fine-tuning set for Cantonese Whisper models. |
| CI-AVSR | [github.com/HLTCHKUST/CI-AVSR](https://github.com/HLTCHKUST/CI-AVSR) | [Dai et al., 2022 (LREC)](https://aclanthology.org/2022.lrec-1.731/) | Cantonese audio-visual speech of in-car commands (8.3 hours, 30 speakers). |
| CantoMap | — | [Winterstein et al., 2020 (LREC)](https://aclanthology.org/2020.lrec-1.355/) | Hong Kong Cantonese MapTask dialogue corpus (40 speakers, about 13 hours). |
| MCE (Mixed Cantonese and English) | [github.com/Shelton1013/Whisper_MCE](https://github.com/Shelton1013/Whisper_MCE) | [Xie & Chen, 2023 (arXiv)](https://arxiv.org/abs/2310.17953) | 34.8 hours of Cantonese–English code-switched speech, with a fine-tuned Whisper. |
| CUMIX | — | [Chan et al., 2005 (Interspeech)](https://www.isca-archive.org/interspeech_2005/chan05b_interspeech.html) | Cantonese–English code-mixing read-speech corpus from CUHK. |
| CAVES | — | [Chong et al., 2024 (Behavior Research Methods)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11289252/) | Cantonese audio-visual emotional speech: 10 speakers, 6 emotions plus neutral. |
| EM2LDL | [github.com/xingfengli/EM2LDL](https://github.com/xingfengli/EM2LDL) | [Li et al., 2025 (arXiv)](https://arxiv.org/abs/2511.20106) | Mixed-emotion speech in English, Mandarin, and Cantonese, with code-switching. |
| Guangzhou Cantonese Conversational Speech Corpus | [magichub.com](https://magichub.com/datasets/guangzhou-cantonese-conversational-speech-corpus/) | — | 4.25 hours of spontaneous Guangzhou Cantonese conversation (CC BY-NC-ND 4.0). |
| Cantonese Whisper fine-tunes | e.g. [whisper-small-cantonese](https://huggingface.co/alvanlii/whisper-small-cantonese), [whisper-large-v2-cantonese](https://huggingface.co/simonl0909/whisper-large-v2-cantonese) | — | Community Whisper ASR models fine-tuned on Cantonese from Common Voice. |

## Lexical resources and dictionaries

| Resource | Link | Paper | Description |
|---|---|---|---|
| CC-Canto | [cantonese.org](https://cantonese.org) | — | Open Cantonese–English dictionary in the style of CC-CEDICT. |
| Words.hk | [words.hk](https://words.hk) | [Lau et al., 2022 (DCLRL @ LREC)](https://aclanthology.org/2022.dclrl-1.7/) | Large crowdsourced Cantonese dictionary with definitions, translations, and romanized examples. |
| CC-CEDICT | [cedict.org](https://cedict.org) | — | Open Chinese–English dictionary. |
| Kaifangcidian 粵語辭典 | [kaifangcidian.com](https://kaifangcidian.com) | — | Open Cantonese dictionary. |
| rime-cantonese | [github.com/rime/rime-cantonese](https://github.com/rime/rime-cantonese) | — | Cantonese input-method schema for Rime, with a large word list in LSHK Jyutping. |
| Hong Kong Supplementary Character Set (HKSCS) | [ccli.gov.hk/en/hkscs](https://www.ccli.gov.hk/en/hkscs/what_is_hkscs.html) | — | Standard set of extra characters used in Hong Kong, including Cantonese-specific characters. |

## Models

| Resource | Link | Paper | Description |
|---|---|---|---|
| hon9kon9ize Cantonese BERT models | [bert-base-cantonese](https://huggingface.co/hon9kon9ize/bert-base-cantonese), [bert-large-cantonese](https://huggingface.co/hon9kon9ize/bert-large-cantonese), [-nli](https://huggingface.co/hon9kon9ize/bert-large-cantonese-nli), [-sts](https://huggingface.co/hon9kon9ize/bert-large-cantonese-sts) | — | Cantonese BERT encoders. The base model is bert-base-chinese with continued pre-training on closed-source Cantonese news, social media, and web text. The large model (about 326M parameters) is trained on Cantonese. The -nli and -sts versions are sentence-embedding models tuned on yue-all-nli and yue-stsb. |
| CantoneseLLM v2 | [huggingface.co/collections/hon9kon9ize/cantonesellm-v20](https://huggingface.co/collections/hon9kon9ize/cantonesellm-v20) | [Cheng et al., 2026 (arXiv)](https://arxiv.org/abs/2609.06970) | Qwen3 8B and 30B-A3B models with continued pre-training on 784M Cantonese tokens, then SFT, DPO, and RLVR for reasoning in Cantonese. |
| CantoneseLLMChat-v1.0 | [7B](https://huggingface.co/hon9kon9ize/CantoneseLLMChat-v1.0-7B), [72B](https://huggingface.co/hon9kon9ize/CantoneseLLMChat-v1.0-72B) | — | Qwen2.5 with continued pre-training on about 600M Cantonese/HK tokens, then instruction tuning. |
| CantoneseLLM-6B (preview) | [huggingface.co/hon9kon9ize/CantoneseLLM-6B-preview202402](https://huggingface.co/hon9kon9ize/CantoneseLLM-6B-preview202402) | — | Yi-6B with further pre-training on 800M Cantonese tokens. |
| cantonese-gemma-2 | [huggingface.co/hon9kon9ize/cantonese-gemma-2-9b-lora-preview20240929](https://huggingface.co/hon9kon9ize/cantonese-gemma-2-9b-lora-preview20240929) | — | Gemma-2 LoRA fine-tunes (2B/9B) for Cantonese. |
| HKGAI-V1 | — | [Han et al., 2025 (arXiv)](https://arxiv.org/abs/2507.11502) | LLM built for Hong Kong, fluent in Cantonese and aligned to local norms. |
| Llama-3.1-Cantonese-8B-Instruct | [huggingface.co/lordjia/Llama-3.1-Cantonese-8B-Instruct](https://huggingface.co/lordjia/Llama-3.1-Cantonese-8B-Instruct) | — | Community LoRA instruction tune of Llama 3.1 on Cantonese data. |
| mt5-translate-zh-yue | [huggingface.co/botisan-ai/mt5-translate-zh-yue](https://huggingface.co/botisan-ai/mt5-translate-zh-yue) | — | mT5-base fine-tuned for Mandarin→Cantonese translation. |
| CantonMT | [github.com/kenrickkung/CantoneseTranslation](https://github.com/kenrickkung/CantoneseTranslation) | Hong et al., 2024: [EAMT](https://aclanthology.org/2024.eamt-1.49/), [arXiv](https://arxiv.org/abs/2405.08172), [AMTA](https://aclanthology.org/2024.amta-presentations.9/) | Cantonese→English NMT models (fine-tuned OpusMT, NLLB, and mBART) with a web app to compare them. Also releases Cantonese–English parallel data and synthetic back-translation data. The three papers cover the platform, the back-translation and model-switch study, and a human evaluation. |

## Tools

| Resource | Link | Paper | Description |
|---|---|---|---|
| PyCantonese | [github.com/jacksonllee/pycantonese](https://github.com/jacksonllee/pycantonese) | [Lee et al., 2022 (LREC)](https://aclanthology.org/2022.lrec-1.711/) | NLTK-style Python library for Cantonese: corpus access, Jyutping, word segmentation, POS tagging. |
| ToJyutping / to-jyutping | [Python](https://github.com/CanCLID/ToJyutping), [JS](https://github.com/CanCLID/to-jyutping) | — | Converts Cantonese characters to Jyutping automatically (CanCLID). |
| PyJyutping | [github.com/MacroYau/PyJyutping](https://github.com/MacroYau/PyJyutping) | — | Python text-to-Jyutping converter based on the official Cantonese pronunciation list. |
| pinyin-jyutping | [github.com/Vocab-Apps/pinyin-jyutping](https://github.com/Vocab-Apps/pinyin-jyutping) | — | Converts Chinese text to Pinyin or Jyutping. |
| g2p-mix | [github.com/pengzhendong/g2p-mix](https://github.com/pengzhendong/g2p-mix) | — | Grapheme-to-phoneme for mixed Mandarin/Cantonese and English text. |
| CharsiuG2P | [github.com/lingjzhu/charsiug2p](https://github.com/lingjzhu/charsiug2p) | — | Multilingual grapheme-to-phoneme in 100 languages, including Cantonese. |
| cantoseg | [github.com/ayaka14732/cantoseg](https://github.com/ayaka14732/cantoseg) | — | Cantonese word segmenter built on jieba. |
| canto-filter | [github.com/CanCLID/canto-filter](https://github.com/CanCLID/canto-filter) | — | Rule-based filter that separates Cantonese, Standard Chinese, and mixed text. |
| GlotLID | [huggingface.co/cis-lmu/glotlid](https://huggingface.co/cis-lmu/glotlid) | [Kargaran et al., 2023 (EMNLP Findings)](https://aclanthology.org/2023.findings-emnlp.410/) | Language identifier for 2,000+ labels, including Cantonese. |
| fastlangid | [github.com/currentslab/fastlangid](https://github.com/currentslab/fastlangid) | — | Language ID that tells Cantonese apart from Simplified and Traditional Chinese. |
| OpenCC | [github.com/byvoid/opencc](https://github.com/byvoid/opencc) | — | Simplified/Traditional converter with Hong Kong variant profiles. |
| HanziConv | [github.com/berniey/hanziconv](https://github.com/berniey/hanziconv) | — | Converter between Simplified and Traditional Chinese. |
| SimAlign | [github.com/cisnlp/simalign](https://github.com/cisnlp/simalign) | [Jalili Sabet et al., 2020 (EMNLP Findings)](https://aclanthology.org/2020.findings-emnlp.147/) | Word aligner based on embeddings that needs no parallel training data. Useful for aligning Cantonese–Mandarin pairs. |
| Cantonese punctuation restoration | [github.com/fanolabs/cantonese-punctuation-restoration](https://github.com/fanolabs/cantonese-punctuation-restoration) | [Suen et al., 2025 (Interspeech)](https://www.isca-archive.org/interspeech_2025/suen25_interspeech.html) | Punctuation restoration models trained on LLM-annotated spoken Cantonese. |

## Papers (Cantonese NLP / linguistics)

| Paper | Description |
|---|---|
| **Surveys** | |
| [Xiang, Liao & Li (2024). *Cantonese Natural Language Processing in the Transformers Era.* SIGHAN-10](https://aclanthology.org/2024.sighan-1.8/) | Survey of Cantonese NLP. Covers data scarcity and reliance on Mandarin transfer. |
| [Xiang et al. (2022). *When Cantonese NLP Meets Pre-training: Progress and Challenges.* AACL-IJCNLP Tutorials](https://aclanthology.org/2022.aacl-tutorials.3/) | Tutorial on pre-training for Cantonese NLP. |
| [Liu & Best (2025). *A Survey of NLP Progress in Sino-Tibetan Low-Resource Languages.* NAACL](https://aclanthology.org/2025.naacl-long.396/) | Survey of Sino-Tibetan NLP; Cantonese is the most-studied language in it. |
| **LLMs and evaluation** | |
| [Vij et al. (2026). *Improving Methodologies for LLM Evaluations Across Global Languages.* arXiv](https://arxiv.org/abs/2601.15706) | Safety evaluation of LLMs across ten languages, including Cantonese. |
| [Xu (2026). *What Drives Dialectal Jailbreaks? An Ablation of Surface Form, Cultural Framing, and Strategy Banks.* EMNLP](https://arxiv.org/abs/2609.31664) | Red-teaming LLMs in Cantonese and Shanghainese. |
| **Written Cantonese identification** | |
| [Lau, Lau & To (2024). *The Extraction and Fine-grained Classification of Written Cantonese Materials through Linguistic Feature Detection.* EURALI @ LREC-COLING](https://aclanthology.org/2024.eurali-1.4/) | Identifies written Cantonese and separates it from Standard Chinese in web text. |
| **Machine translation** | |
| [Dare et al. (2023). *Unsupervised Mandarin-Cantonese Machine Translation.* arXiv](https://arxiv.org/abs/2301.03971) | Unsupervised MT using a new corpus of about 1M Cantonese sentences. |
| [Suen, Chow & Lam (2024). *Leveraging Mandarin as a Pivot Language for Low-Resource Machine Translation between Cantonese and English.* LoResMT](https://aclanthology.org/2024.loresmt-1.8/) | Cantonese–English MT that pivots through Mandarin. |
| [Zhang (1998). *Dialect MT: A Case Study between Cantonese and Mandarin.* COLING-ACL](https://aclanthology.org/P98-2238/) | Early rule-based Cantonese–Mandarin machine translation. |
| **Sentiment and social media** | |
| [Fu et al. (2024). *Efficacy of ChatGPT in Cantonese Sentiment Analysis: Comparative Study.* JMIR](https://www.jmir.org/2024/1/e51069) | Compares GPT-3.5 and GPT-4 with lexicon-based and machine-learning sentiment methods on Cantonese text. |
| [Xiang, Jiao & Lu (2019). *Sentiment Augmented Attention Network for Cantonese Restaurant Review Analysis.* WISDOM @ KDD](https://sentic.net/wisdom2019xiang.pdf) | Attention-based sentiment model for Cantonese reviews. |
| [Li et al. (2022). *Sentiment Analysis of Political Posts on Hong Kong Local Forums Using Fine-Tuned mBERT.* IEEE BigData](https://scholars.hkbu.edu.hk/en/publications/sentiment-analysis-of-political-posts-on-hong-kong-local-forums-u/) | Sentiment of Cantonese political forum posts. |
| [Chen et al. (2024). *A deep semantic-aware approach for Cantonese rumor detection in social networks with graph convolutional network.* Expert Systems with Applications](https://www.sciencedirect.com/science/article/abs/pii/S0957417423035091) | Cantonese rumor detection that combines a Cantonese BERT with a graph network. |
| **Speech and phonology** | |
| [Li, Wang & Beigi (2019). *Cantonese Automatic Speech Recognition Using Transfer Learning from Mandarin.* arXiv](https://arxiv.org/abs/1911.09271) | Cantonese speech recognition with transfer from Mandarin. |
| [Luo et al. (2021). *Cross-Language Transfer Learning and Domain Adaptation for End-to-End Automatic Speech Recognition.* ICME](https://doi.org/10.1109/ICME51207.2021.9428334) | ASR transfer that includes Cantonese. |
| Li (2024). *Fine-tuning Guangdong Cantonese based on wav2vec 2.0 XLSR model pretrained on Mandarin Chinese to improve ASR performance.* MA thesis, University of Groningen | Cantonese ASR adapted from a Mandarin model. |
| [Chen et al. (2025). *CantoASR: Prosody-Aware ASR-LALM Collaboration for Low-Resource Cantonese.* arXiv](https://arxiv.org/abs/2511.04139) | Corrects Whisper ASR output with an audio LLM that uses prosody. |
| [Mak, Suen & Lam (2025). *Speech-guided Grapheme-to-Phoneme Conversion for Cantonese Text-to-Speech.* Interspeech](https://www.isca-archive.org/interspeech_2025/mak25_interspeech.html) | Uses speech to choose among possible readings of a character, improving G2P for TTS. |
| [Zhang et al. (2026). *Toward a Cross-Lingual Romanization Ecosystem for Sinitic Languages: A Paired Mandarin-Cantonese Case Study.* ISCSLP](https://arxiv.org/abs/2608.29170) | A shared romanization scheme for Mandarin and Cantonese, applied to ASR. |
| **Grammar and linguistics** | |
| [Lam (2020). *Forms and Meanings of Lexical Reduplications in Cantonese: A Corpus Study.* PACLIC](https://aclanthology.org/2020.paclic-1.64/) | Corpus study of Cantonese reduplication. |
| Matthews & Yip (2011). *Cantonese: A Comprehensive Grammar* (2nd ed.). Routledge | Reference grammar of Cantonese. |
| Snow (2004). *Cantonese as Written Language: The Growth of a Written Chinese Vernacular.* Hong Kong University Press | History and status of written Cantonese. |
| Snow (2008). *Cantonese as written standard?* Journal of Asian Pacific Communication 18(2) | Whether written Cantonese functions as a standard. |
