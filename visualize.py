import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from model import TransformerClassifier


PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def tokenize(text: str):
    return text.strip().lower().split()


def encode_text(text: str, vocab: dict[str, int], max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab.get(UNK_TOKEN, 1)) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab.get(PAD_TOKEN, 0)] * (max_len - length)
    return ids, tokens[:max_len]


def load_vocab(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_meta(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_sentence_from_test(processed_dir: Path):
    test_data = torch.load(processed_dir / "test.pt")
    return test_data["texts"][0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="")
    parser.add_argument("--sentence", type=str, default="")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    results_dir = Path(args.results_dir)
    vocab = load_vocab(processed_dir / "vocab.json")
    meta = load_meta(processed_dir / "meta.json")

    if args.model:
        model_path = Path(args.model)
    else:
        candidates = sorted(results_dir.glob("model_Transformer*.pt"))
        if not candidates:
            raise FileNotFoundError("Khong tim thay model_Transformer*.pt trong results/. Hay chay train.py truoc.")
        model_path = candidates[0]

    stem = model_path.stem.replace("model_", "")
    import re
    match = re.search(r"d(\d+)_ff(\d+)", stem)
    if match:
        d_model, d_ff = int(match.group(1)), int(match.group(2))
    else:
        d_model, d_ff = 64, 128

    model = TransformerClassifier(
        vocab_size=meta["vocab_size"],
        d_model=d_model,
        d_ff=d_ff,
        max_len=meta["max_len"],
        num_classes=meta["num_classes"],
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    sentence = args.sentence if args.sentence else pick_sentence_from_test(processed_dir)
    input_ids, tokens = encode_text(sentence, vocab, meta["max_len"])

    with torch.no_grad():
        logits = model(torch.tensor([input_ids], dtype=torch.long))
        pred = logits.argmax(dim=-1).item()
        weights = model.last_attention_weights[0, : len(tokens), : len(tokens)].cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(weights)
    plt.colorbar()
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right")
    plt.yticks(range(len(tokens)), tokens)
    plt.title(f"Attention heatmap | pred={meta['label_names'][pred]}")
    plt.tight_layout()

    out_path = results_dir / "attention_heatmap.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Sentence: {sentence}")
    print(f"Predicted label: {meta['label_names'][pred]}")
    print(f"Saved heatmap to: {out_path}")


if __name__ == "__main__":
    main()
