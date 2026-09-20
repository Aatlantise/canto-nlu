from main import WuPreTrainer, CantoPreTrainer, CantoNLIFineTuner, CantoPOSFineTuner, CantoDEPSFineTuner, CantoTokenClassificationFineTuner, CantoAcceptabilityFineTuner
from argparse import ArgumentParser
from transformers import Trainer

def run(args):
    if args.pretrain:
        if args.lang == "yue":
            assert args.data in ["wiki",
                                "cantonese-sentences",
                                "canto-corpus"], f"{args.data} is not a valid dataset. Choose between 'wiki', 'cantonese-sentences', 'canto-corpus'"
            model = CantoPreTrainer(model_dir=args.model_dir, scratch=args.scratch, data=args.data)
            model.train()
        elif args.lang == "wuu":
            model = WuPreTrainer(model_dir=args.model_dir)
            model.train()
        else:
            print(f"{args.lang} pre-training is not supported. Please choose from: yue, wuu")
    if args.finetune:
        if args.lang == "yue":
            if args.task == "nli":
                model = CantoNLIFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            elif args.task == "pos":
                model = CantoPOSFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            elif args.task == "deps":
                model = CantoDEPSFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            elif args.task == "accept":
                model = CantoAcceptabilityFineTuner(args.lang, model_dir=args.model_dir)
                model.finetune()
            else:
                print(f"{args.task} fine-tuning is not supported. Please choose from: pos, nli, deps")

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lang", default="yue")
    parser.add_argument("--model_dir", default="./models/bert-base-chinese-local")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--scratch", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="wiki")
    parser.add_argument("--task", type=str, default="")
    args = parser.parse_args()

    """
    Add your custom arguments for IDE tests here
    """

    if args.task:
        args.finetune = True
    if args.scratch:
        args.pretrain = True

    print(args)
    run(args)