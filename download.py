from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

# Hugging Face model ID -> local directory name under --models_root
MODELS = {
    "google-bert/bert-base-chinese": "bert-base-chinese",
    "hon9kon9ize/bert-base-cantonese": "bert-base-cantonese",
    "feynmanzhao/chinese-modernbert-large-wwm": "chinese-modernbert",
}


def download_datasets():
    print("Downloading Cantonese NLI dataset...")
    ds = load_dataset("hon9kon9ize/yue-all-nli")
    ds.save_to_disk("./data/yue-nli-local")


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
