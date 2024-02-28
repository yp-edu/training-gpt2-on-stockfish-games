"""Make a leaderboard hosted on wandb.

Run with:
```bash
poetry run python -m scripts.wandb_leaderboard
```
"""

import argparse

import wandb

#######################################
# HYPERPARAMETERS
#######################################
parser = argparse.ArgumentParser("wandb_leaderboard")
parser.add_argument(
    "--make_leaderboard", action=argparse.BooleanOptionalAction, default=False
)
parser.add_argument(
    "--retrieve_leaderboard",
    action=argparse.BooleanOptionalAction,
    default=False,
)
parser.add_argument("--run_id", type=str, default="")
parser.add_argument(
    "--add_debug_row", action=argparse.BooleanOptionalAction, default=False
)
#######################################

ARGS = parser.parse_args()

if ARGS.make_leaderboard:
    leaderboard = wandb.Table(  # type: ignore
        columns=["username", "winin", "pgn"], data=[]
    )
    with wandb.init(  # type: ignore
        project="gpt2-stockfish-debug", entity="yp-edu"
    ) as run:
        run.log({"leaderboard": leaderboard})

if ARGS.retrieve_leaderboard:
    with wandb.init(  # type: ignore
        project="gpt2-stockfish-debug", entity="yp-edu"
    ) as run:
        artifact = run.use_artifact(f"run-{ARGS.run_id}-leaderboard:latest")
        leaderboard = artifact.get("leaderboard")
        print(list(leaderboard.iterrows()))
        leaderboard.add_data("debug", 0, "pgn")
        artifact.add(leaderboard, "leaderboard")
        artifact.save()
