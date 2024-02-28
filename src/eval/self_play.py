"""Simple self-play script for testing the model.
"""

import chess
from transformers import AutoModelForCausalLM, AutoTokenizer


def next_move(model, tokenizer, fen):
    input_ids = tokenizer(f"FEN: {fen}\nMOVE:", return_tensors="pt")
    input_ids = {k: v.to(model.device) for k, v in input_ids.items()}
    out = model.generate(
        **input_ids,
        max_new_tokens=10,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.1,
    )
    out_str = tokenizer.batch_decode(out)[0]
    return out_str.split("MOVE:")[-1].replace("<|endoftext|>", "").strip()


board = chess.Board()
model = AutoModelForCausalLM.from_pretrained("Xmaster6y/gpt2-stockfish-debug")
tokenizer = AutoTokenizer.from_pretrained(
    "Xmaster6y/gpt2-stockfish-debug"
)  # or "gpt2"
tokenizer.pad_token = tokenizer.eos_token
for i in range(100):
    fen = board.fen()
    move_uci = next_move(model, tokenizer, fen)
    try:
        print(move_uci)
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise chess.IllegalMoveError
        board.push(move)
    except chess.IllegalMoveError:
        print(board)
        print("Illegal move", i)
        break
