"""Score generated texts: GPT-2 log-perplexity (primary), distinct-word ratio
and denoising steps (secondary)."""
import argparse
import json
from pathlib import Path


def distinct_ratio(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


def steps_from_consumed(consumed: int) -> int:
    """Denoising steps the confidence sampler ran: 4-byte seed, then one 1024-byte
    canvas initially and one per step."""
    return (consumed - 4) // 1024 - 1


def score_line(line: dict, nll_fn) -> dict:
    text = line.get("text") or ""
    usable = line.get("error") is None and text.strip() != ""
    return {
        **line,
        "gpt2_nll": float(nll_fn(text)) if usable else None,
        "distinct_ratio": distinct_ratio(text),
        "steps": steps_from_consumed(line["consumed"]) if line.get("consumed") else None,
    }


class Gpt2Scorer:
    """Mean per-token negative log-likelihood (nats) under gpt2 (124M)."""

    def __init__(self):
        import torch
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = GPT2TokenizerFast.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device).eval()

    def __call__(self, text: str) -> float:
        ids = self.tok(text, return_tensors="pt", truncation=True, max_length=1024).input_ids
        ids = ids.to(self.device)
        with self.torch.no_grad():
            return float(self.model(ids, labels=ids).loss)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("batch_jsonl")
    p.add_argument("out")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    lines = [json.loads(l) for l in Path(args.batch_jsonl).read_text().splitlines() if l.strip()]
    if args.limit is not None:
        lines = lines[: args.limit]
    scorer = Gpt2Scorer()
    with open(args.out, "w") as out:
        for line in lines:
            scored = score_line(line, scorer)
            out.write(json.dumps(scored) + "\n")
    n_null = sum(s is None for s in (json.loads(l)["gpt2_nll"] for l in open(args.out)))
    print(f"scored {len(lines)} samples on {scorer.device}; {n_null} without gpt2_nll")


if __name__ == "__main__":
    main()
