"""Training a LLM to play chess.

Use with:
```python
poetry run python -m src.train.llm
```
"""

import argparse
from typing import Optional

import loguru
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainingArguments,
)

from src.utils.llm_dataset import CustomCollator, LlmDataset, LogCallback

parser = argparse.ArgumentParser("llm")
parser.add_argument(
    "--training", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument("--n-epochs", type=int, default=1)
parser.add_argument("--logging-steps-ratio", type=float, default=0.01)
parser.add_argument("--eval-steps-ratio", type=float, default=0.5)
parser.add_argument(
    "--train-batch-size", type=int, default=50
)  # 275 A100 60 other
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument(
    "--eval-batch-size", type=int, default=500
)  # 600 A100 150 Other
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--dataloader-num-workers", type=int, default=4)
parser.add_argument("--resume-from-checkpoint", type=int, default=None)

args = parser.parse_args()

NAME = f"llm_{args.n_epochs}_{args.train_batch_size}_{args.lr}"
OUTPUT_DIR = f"output/weights/{NAME}"
LOGGING_DIR = f"output/logging/{NAME}"

if args.resume_from_checkpoint is not None:
    resume_from_checkpoint: Optional[str] = (
        f"{OUTPUT_DIR}/checkpoint-{args.resume_from_checkpoint}"
    )
else:
    resume_from_checkpoint = None

print("[INFO] Loading datasets")

eval_dataset = LlmDataset(
    "./data/test_stockfish_5000_sar.jsonl", n_parts=args.dataloader_num_workers
)
train_dataset = LlmDataset(
    "./data/train_stockfish_262k_sar.jsonl",
    n_parts=args.dataloader_num_workers,
)
print(len(eval_dataset))
print(len(train_dataset))


model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
collator = CustomCollator(tokenizer=tokenizer)
train_dataset_len = train_dataset.n_lines


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    logging_dir=LOGGING_DIR,
    overwrite_output_dir=True,
    logging_strategy="steps",
    logging_steps=int(
        args.logging_steps_ratio * train_dataset_len / args.train_batch_size
    ),
    evaluation_strategy="steps",
    eval_steps=int(
        args.eval_steps_ratio * train_dataset_len / args.train_batch_size
    ),
    save_strategy="steps",
    save_steps=int(
        args.eval_steps_ratio * train_dataset_len / args.train_batch_size
    ),
    per_device_eval_batch_size=args.eval_batch_size,
    per_device_train_batch_size=args.train_batch_size,
    num_train_epochs=args.n_epochs,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.lr,
    run_name="latest",
    fp16=True,
    remove_unused_columns=False,
    disable_tqdm=True,
    include_tokens_per_second=False,
    dataloader_num_workers=args.dataloader_num_workers,
    dataloader_pin_memory=True,
    resume_from_checkpoint=resume_from_checkpoint,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collator,
)
trainer.remove_callback(ProgressCallback)
trainer.remove_callback(PrinterCallback)
trainer.add_callback(LogCallback)

if args.training:
    loguru.logger.info("Training")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
else:
    loguru.logger.info("Evaluating")
    evaluation = trainer.evaluate()
    loguru.logger.info(evaluation)
