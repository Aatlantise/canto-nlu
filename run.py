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
from transformers import AutoConfig, Trainer, set_seed

FINETUNE_TASKS = {
    "nli": CantoNLIFineTuner,
    "pos": CantoPOSFineTuner,
    "deps": CantoDEPSFineTuner,
    "sentiment": CantoSentimentFineTuner,
    "ld": CantoLangDetectFineTuner,
    "laj": CantoLAJFineTuner,
}

def is_modernbert(model_dir):
    """True if model_dir holds a ModernBERT-architecture model, read from its config.json."""
    try:
        model_type = AutoConfig.from_pretrained(model_dir, trust_remote_code=True).model_type
    except (OSError, ValueError):
        model_type = ""
    return "modernbert" in model_type.lower() or "modernbert" in model_dir.lower()


def run(args):
    # Seed before any model is built: Trainer only seeds once it is constructed, after the
    # task-specific heads have already been randomly initialized.
    set_seed(args.seed)
    if args.pretrain:
        if args.lang == "yue":
            assert args.data in ["wiki",
                                "cantonese-sentences"], f"{args.data} is not a valid dataset. Choose between 'wiki', 'cantonese-sentences'"
            model = CantoPreTrainer(model_dir=args.model_dir, scratch=args.scratch, data=args.data, seed=args.seed)
            model.train()
        else:
            print(f"{args.lang} pre-training is not supported. Please choose from: yue")
    if args.finetune:
        if args.lang == "yue":
            fine_tuner_cls = FINETUNE_TASKS.get(args.task)
            if fine_tuner_cls is None:
                print(f"{args.task} fine-tuning is not supported. Please choose from: {', '.join(FINETUNE_TASKS)}")
            else:
                model = fine_tuner_cls(args.lang, model_dir=args.model_dir, per_device_batch_size=args.batch_size,
                                       seed=args.seed, gradient_accumulation_steps=args.grad_accum)
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
                                       per_device_batch_size=args.batch_size, seed=args.seed)

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
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Per-device batch size. Defaults to 64 (base-sized BERT on a 20GB-VRAM device), "
                             "or 32 for ModernBERT, which runs out of memory at 64.")
    parser.add_argument("--grad_acc", type=int, default=None,
                        help="Gradient accumulation steps. Defaults to 1, or 2 for ModernBERT, keeping the "
                             "effective batch size at 64 so every model gets the same number of optimizer steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for model head initialization, data shuffling and dropout. "
                             "Run with several seeds to measure cross-seed variance.")
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

    modernbert = is_modernbert(args.model_dir)
    if args.batch_size is None:
        args.batch_size = 32 if modernbert else 64
    if args.grad_accum is None:
        args.grad_accum = 2 if modernbert else 1

    print(args)
    run(args)
