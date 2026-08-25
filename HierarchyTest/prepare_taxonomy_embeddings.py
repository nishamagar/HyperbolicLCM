from __future__ import annotations

import argparse
import csv

import torch
from torch import amp
from transformers import AutoModel, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--taxonomy", required=True)
    p.add_argument("--output", default="taxonomy_input_embeddings.pt")
    p.add_argument("--model", default="microsoft/deberta-v3-small")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    with open(args.taxonomy, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "concept_id" not in rows[0] or "concept_text" not in rows[0]:
        raise ValueError("Taxonomy CSV must contain concept_id and concept_text columns")
    ids = [str(row["concept_id"]).strip() for row in rows]
    texts = [str(row["concept_text"]).strip() for row in rows]
    if any(not x for x in ids) or any(not x for x in texts):
        raise ValueError("concept_id and concept_text must be non-empty")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    encoder = AutoModel.from_pretrained(args.model).to(device).eval()
    vectors = []
    use_amp = device.type == "cuda"
    dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    for start in range(0, len(texts), args.batch_size):
        batch = tokenizer(
            texts[start:start + args.batch_size],
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp.autocast(device_type="cuda", enabled=use_amp, dtype=dtype):
            cls = encoder(**batch).last_hidden_state[:, 0, :]
        vectors.append(cls.detach().cpu().float())

    result = {
        "concept_ids": ids,
        "embeddings": torch.cat(vectors, dim=0),
        "encoder_name": args.model,
        "max_length": args.max_length,
        "concept_text": texts,
    }
    torch.save(result, args.output)
    print(f"Saved {len(ids)} concept vectors to {args.output}")


if __name__ == "__main__":
    main()
