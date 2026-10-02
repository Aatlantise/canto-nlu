from main import (
    CantoPreTrainer,
    CantoSequenceClassificationFineTuner,
    CantoNLIFineTuner,
    CantoPOSFineTuner,
    CantoDEPSFineTuner,
    CantoSentimentFineTuner,
    CantoLangDetectFineTuner,
    CantoLAJFineTuner,
)
from argparse import ArgumentParser
from transformers import Trainer

FINETUNE_TASKS = {
    "nli": CantoNLIFineTuner,
    "pos": CantoPOSFineTuner,
    "deps": CantoDEPSFineTuner,
    "sentiment": CantoSentimentFineTuner,
    "ld": CantoLangDetectFineTuner,
    "laj": CantoLAJFineTuner,
}

def run(args):
    if args.pretrain:
        if args.lang == "yue":
            assert args.data in ["wiki",
                                "cantonese-sentences"], f"{args.data} is not a valid dataset. Choose between 'wiki', 'cantonese-sentences'"
            model = CantoPreTrainer(model_dir=args.model_dir, scratch=args.scratch, data=args.data)
            model.train()
        else:
            print(f"{args.lang} pre-training is not supported. Please choose from: yue")
    if args.finetune:
        if args.lang == "yue":
            fine_tuner_cls = FINETUNE_TASKS.get(args.task)
            if fine_tuner_cls is None:
                print(f"{args.task} fine-tuning is not supported. Please choose from: {', '.join(FINETUNE_TASKS)}")
            else:
                model = fine_tuner_cls(args.lang, model_dir=args.model_dir, per_device_batch_size=args.batch_size)
                model.finetune()
        else:
            print(f"{args.lang} fine-tuning is not supported. Please choose from: yue")
    if args.eval_only:
        if args.lang == "yue":
            fine_tuner_cls = FINETUNE_TASKS.get(args.task)
            if fine_tuner_cls is None or not issubclass(fine_tuner_cls, CantoSequenceClassificationFineTuner):
                eval_only_tasks = [t for t, cls in FINETUNE_TASKS.items() if issubclass(cls, CantoSequenceClassificationFineTuner)]
                print(f"{args.task} evaluating is not supported. Please choose from: {', '.join(eval_only_tasks)}")
            else:
                model = fine_tuner_cls(args.lang, model_dir=args.model_dir, eval_only=True,
                                       per_device_batch_size=args.batch_size)

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
    parser.add_argument("--model_dir", default="./models/yue-monolingual")
    parser.add_argument("--pretrain", action="store_true", default=False)
    parser.add_argument("--scratch", action="store_true", default=False)
    parser.add_argument("--finetune", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--data", type=str, default="wiki")
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size. E.g. for a 20GB VRAM card with bert-base, use 64.")
    args = parser.parse_args()

    """
    Add your custom arguments for IDE tests here
    """

    if args.task:
        args.finetune = True
    if args.scratch:
        args.pretrain = True
    if args.eval_only:
        args.finetune = False

    print(args)
    run(args)
