import subprocess
from dataclasses import dataclass

import chess
import jsonlines
import loguru
import torch
import tqdm
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizerBase, TrainerCallback


def preprocess_games(in_path, out_path):
    with jsonlines.open(in_path) as reader:
        with jsonlines.open(out_path, "w") as writer:
            for obj in tqdm.tqdm(reader):
                state_action = []
                parsed_moves = [
                    m for m in obj["moves"].split() if not m.endswith(".")
                ]
                board = chess.Board()
                for m in parsed_moves:
                    fen = board.fen()
                    move = board.push_san(m)
                    state_action.append({"fen": fen, "move": move.uci()})
                outcome = board.outcome()
                if outcome is None:
                    result = "-"
                else:
                    result = outcome.result()
                writer.write_all(
                    [{**sa, "result": result} for sa in state_action]
                )


class LlmDataset(IterableDataset):
    def __init__(
        self,
        file_path,
        n_parts=1,
    ):
        super().__init__()
        self.file_path = file_path
        *rest, ext = file_path.split(".")
        self.base_name = ".".join(rest)
        self.ext = ext

        c_out = subprocess.Popen(
            ["wc", "-l", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = c_out.communicate()
        out = stdout.decode().split()
        self.n_lines = int(out[0].split()[0])
        self.n_parts = n_parts
        if n_parts > 1:
            per_part_lines = self.n_lines // n_parts
            c_out = subprocess.Popen(
                [
                    "split",
                    "-l",
                    str(per_part_lines),
                    file_path,
                    self.base_name,
                    "--additional-suffix=." + ext,
                    "-d",
                    "-a",
                    "1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            stdout, stderr = c_out.communicate()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return iter(jsonlines.open(self.file_path))
        else:
            worker_id = worker_info.id
            return iter(
                jsonlines.open(f"{self.base_name}{worker_id}.{self.ext}")
            )

    def __len__(self):
        return self.n_lines


@dataclass
class CustomCollator:
    tokenizer: PreTrainedTokenizerBase
    padding = True
    return_tensors: str = "pt"

    def __call__(self, batch):
        tokenized_batch = self.tokenizer(
            [f"FEN: {exp['fen']}\nMOVE: {exp['move']}" for exp in batch],
            return_tensors=self.return_tensors,
            padding=self.padding,
        )

        labels = tokenized_batch["input_ids"].clone()
        ignore_tokens = self.tokenizer(
            [f"FEN: {exp['fen']}\nMOVE:" for exp in batch]
        )["input_ids"]
        for i, exp in enumerate(ignore_tokens):
            labels[i, : len(exp)] = -100
        tokenized_batch["labels"] = labels
        return tokenized_batch


class LogCallback(TrainerCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            loguru.logger.info(logs)
