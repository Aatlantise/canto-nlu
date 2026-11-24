from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

# Hugging Face model ID -> local directory name under --models_root
MODELS = {
    "google-bert/bert-base-chinese": "bert-base-chinese",
    "hon9kon9ize/bert-base-cantonese": "bert-base-cantonese",
    "feynmanzhao/chinese-modernbert-large-wwm": "chinese-modernbert",
    "google-bert/bert-base-multilingual-cased": "mbert-cased",
    "google-bert/bert-base-multilingual-uncased": "mbert-uncased",
    "FacebookAI/xlm-roberta-base": "xlmr-base",
}


# Canto-XNLI (NLI validation/test), pinned to a fixed commit so reported results stay reproducible.
CANTO_XNLI_ID = "aatlantise/canto-xnli"
CANTO_XNLI_REVISION = "a4146b47902ee02cb1abc5d78879173389ce6fdf"


def download_datasets():
    print("Downloading Cantonese NLI training data (yue-all-nli)...")
    ds = load_dataset("hon9kon9ize/yue-all-nli")
    ds.save_to_disk("./data/yue-nli-local")

    print("Downloading Cantonese NLI evaluation data (canto-xnli)...")
    ds = load_dataset(CANTO_XNLI_ID, revision=CANTO_XNLI_REVISION)
    ds.save_to_disk("./data/canto-xnli-local")


def download_models(models_root):
    for hub_id, local_name in MODELS.items():
        save_dir = Path(models_root) / local_name
        print(f"Downloading {hub_id} to {save_dir}...")
        # Copy the repo as-is rather than save_pretrained(), so custom tokenizer code
        # (chinese-modernbert) is saved locally and the directory loads offline.
        # Skip duplicate TF/Flax/PyTorch-pickle weights; all three ship model.safetensors.
        snapshot_download(
            repo_id=hub_id,
            local_dir=save_dir,
            ignore_patterns=["*.h5", "*.msgpack", "*.bin", "*.ot", "onnx/*", "coreml/*"],
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--models_root", default="./models",
                        help="Directory under which each model and its tokenizer are saved")
    args = parser.parse_args()
    download_datasets()
    download_models(args.models_root)
