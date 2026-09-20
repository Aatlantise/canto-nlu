from datasets import load_dataset
from transformers import BertTokenizerFast, BertForMaskedLM
from argparse import ArgumentParser
import os
import requests

def download(args):
    print("Downloading datasets and models...")
    if args.lang == "yue":
        if not os.path.exists('./data/canto-wiki'):
            print('Downloading Yue Wikipedia...')
            ds = load_dataset('ming030890/wiki-yue-high-quality', split='train')
            ds.save_to_disk('./data/canto-wiki')
        if not os.path.exists('./data/canto-wordshk'):
            print('Downloading words.hk...')
            ds = load_dataset('AlienKevin/wordshk_cantonese_speech', split='train')
            ds.save_to_disk('./data/canto-wordshk')
        if not os.path.exists('./data/canto-lihkg'):
            print('Downloading LIHKG...')
            ds = load_dataset('IKMLab-team/hk_content_corpus', 'lihkg', split='train')
            ds.save_to_disk('./data/canto-lihkg')
        if not os.path.exists('./data/canto-appledaily'):
            print('Downloading Apple Daily...')
            ds = load_dataset('IKMLab-team/hk_content_corpus', 'appledaily', split='train')
            ds.save_to_disk('./data/canto-appledaily')
        if not os.path.exists('./data/canto-hkcancor'):
            print('Downloading HKCanCor...')
            ds = load_dataset('AlienKevin/hkcancor')
            ds.save_to_disk('./data/canto-hkcancor')
        if not os.path.exists('./data/canto-zoengjyutgaai'):
            print('Downloading Zoeng Jyut Gaai...')
            ds = load_dataset('CanCLID/zoengjyutgaai')
            ds.save_to_disk('./data/canto-zoengjyutgaai')
        if not os.path.exists('./data/canto-babylmyue'):
            print('Downloading BabyLM Yue...')
            ds = load_dataset('BabyLM-community/babylm-yue', split='train')
            ds.save_to_disk('./data/canto-babylmyue')
        if not os.path.exists('./data/canto-cantomap.tsv'):
            print('Downloading CantoMap...')
            response = requests.get('https://huggingface.co/datasets/safecantonese/cantomap/resolve/main/transcript/yue/combined.tsv')
            with open('./data/canto-cantomap.tsv', 'wb') as f:
                f.write(response.content)
        if not os.path.exists('./data/canto-tatoeba'):
            print('Downloading Tatoeba Cantonese...')
            ds = load_dataset('tatoeba', lang1='en', lang2='yue')
            ds.save_to_disk('./data/canto-tatoeba')
        # if not os.path.exists('./data/canto-youtube'):
        #     print('Downloading YouTube Cantonese...')
        #     ds = load_dataset('alvanlii/cantonese-youtube')
        #     ds.save_to_disk('./data/canto-youtube')

        print("Downloading BERT tokenizer and model...")
        BertTokenizerFast.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)
#        print("Download SOTA Cantonese BERT model...")
        #BertForMaskedLM.from_pretrained('hon9kon9ize/bert-base-cantonese').save_pretrained(args.model_dir)
        #BertTokenizerFast.from_pretrained('hon9kon9ize/bert-base-cantonese').save_pretrained(args.model_dir)

    elif args.lang == "wuu":
        print("Downloading Wu Wiki dataset...")
        ds = load_dataset("wikimedia/wikipedia", "20231101.wuu")
        ds.save_to_disk("./data/wuu-wiki-local")
        print("Downloading BERT tokenizer...")
        BertTokenizerFast.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)
    else:
        print(f"{args.lang} is not supported. Please choose from: yue, wuu")
        return
    print("Downloading BERT model...")
    BertForMaskedLM.from_pretrained('bert-base-chinese').save_pretrained(args.model_dir)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue", choices=["yue", "wuu"],
                        help="Language to download data for")
    parser.add_argument("--model_dir", default="./models/bert-base-chinese-local",
                        help="Directory to save the model and tokenizer")
    args = parser.parse_args()
    download(args)
