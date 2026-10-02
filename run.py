from main import (WuPreTrainer, CantoPreTrainer, CantoTransferPreTrainer, CantoNLIFineTuner, CantoPOSFineTuner,
                  CantoDEPSFineTuner, CantoTokenClassificationFineTuner, CantoAcceptabilityFineTuner)
from argparse import ArgumentParser
import os
from transformers import Trainer

def run(args):
    if args.pretrain:
        if args.lang == "yue":
            assert args.data in ["wiki",
                                "cantonese-sentences",
                                "canto-corpus"], f"{args.data} is not a valid dataset. Choose between 'wiki', 'cantonese-sentences', 'canto-corpus'"
            print(f"!!!!!!{args.data}")
            if args.transfer:
                model = CantoTransferPreTrainer(model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir,
                                                data=args.data, arch=args.arch, init=args.init,
                                                stage1_steps=args.stage1_steps, t2s=not args.no_t2s)
            else:
                model = CantoPreTrainer(model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir, scratch=args.scratch, data=args.data,
                                        arch=args.arch)
            # print class name of model
            print(f"Pre-training model class: {model.__class__.__name__}")
            model.train()
        elif args.lang == "wuu":
            model = WuPreTrainer(model_dir=args.model_dir)
            model.train()
        else:
            print(f"{args.lang} pre-training is not supported. Please choose from: yue, wuu")
    if args.finetune:
        if args.lang == "yue":
            if args.task == "nli":
                model = CantoNLIFineTuner(args.lang, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)
                model.finetune()
            elif args.task == "pos":
                model = CantoPOSFineTuner(args.lang, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)
                model.finetune()
            elif args.task == "deps":
                model = CantoDEPSFineTuner(args.lang, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)
                model.finetune()
            elif args.task == "accept":
                model = CantoAcceptabilityFineTuner(args.lang, model_dir=args.model_dir, tokenizer_dir=args.tokenizer_dir)
                model.finetune()
            else:
                print(f"{args.task} fine-tuning is not supported. Please choose from: pos, nli, deps, accept")

        else:
            print(f"{args.lang} fine-tuning is not supported. Please choose from: yue")
    if args.eval_only:
        if args.lang == "yue":
            model = CantoNLIFineTuner(args.lang, model_dir=args.model_dir, eval_only=True)

            trainer = Trainer(
                model=model.model,
                args=model.training_args,
                eval_dataset=model.finetune_dataset["test"]
            )

            model.eval(trainer)
        else:
            print(f"{args.lang} evaluating is not supported. Please choose from: yue")

def check_transfer_args(parser, args):
    # fail before any tokenizing or model loading, not halfway into preprocess_data
    if args.lang != "yue":
        parser.error("--transfer is only supported for --lang yue.")
    if args.scratch:
        parser.error("--transfer and --scratch are mutually exclusive: transfer starts from --model_dir.")
    if args.data != "canto-corpus":
        parser.error("--transfer requires --data canto-corpus.")
    if os.path.abspath(args.tokenizer_dir) == os.path.abspath(args.model_dir):
        parser.error("--transfer needs --tokenizer_dir pointing at the Cantonese tokenizer, "
                     "not the source model's directory.")
    if args.stage1_steps is not None and args.stage1_steps < 0:
        parser.error("--stage1_steps must be >= 0.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue")
    parser.add_argument("--model_dir", default="./models/bert-base-chinese-local")
    parser.add_argument("--tokenizer_dir", default="./models/bert-base-chinese-local")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--scratch", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="wiki")
    parser.add_argument("--arch", type=str, default="bert", choices=["bert", "modernbert"])
    parser.add_argument("--task", type=str, default="")
    # tokenizer transfer + two-stage LAPT
    parser.add_argument("--transfer", action="store_true", default=False,
                        help="Swap in --tokenizer_dir's tokenizer and run two-stage LAPT from --model_dir.")
    parser.add_argument("--init", type=str, default="fvt", choices=["fvt", "random"],
                        help="Embedding initialisation for the new vocabulary.")
    parser.add_argument("--stage1_steps", type=int, default=None,
                        help="Frozen-body steps (default: CantoTransferPreTrainer.STAGE1_STEPS; 0 skips stage 1).")
    parser.add_argument("--no_t2s", action="store_true", default=False,
                        help="Disable the Traditional->Simplified lookup during FVT initialisation.")
    args = parser.parse_args()

    """
    Add your custom arguments for IDE tests here
    """

    if args.task:
        args.finetune = True
    if args.scratch:
        args.pretrain = True
    if args.transfer:
        args.pretrain = True
        check_transfer_args(parser, args)

    print(args)
    run(args)