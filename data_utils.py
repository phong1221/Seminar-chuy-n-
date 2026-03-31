import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
import torch

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def tokenize(text: str):
    return text.strip().lower().split()


def build_vocab(df: pd.DataFrame):
    counter = Counter()
    for text in df["text"]:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counter.keys()):
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - length)
    return ids, length


def dataframe_to_tensor_dict(df: pd.DataFrame, vocab: dict[str, int], max_len: int):
    input_ids, lengths, labels, texts = [], [], [], []
    for _, row in df.iterrows():
        ids, length = encode_text(row["text"], vocab, max_len)
        input_ids.append(ids)
        lengths.append(length)
        labels.append(int(row["label"]))
        texts.append(row["text"])
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "texts": texts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data/sentiment_raw.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--max_len", type=int, default=20)
    parser.add_argument("--show_stats", action="store_true")
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_csv)
    required_cols = {"id", "split", "text", "label", "label_name", "num_tokens"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    vocab = build_vocab(train_df)

    train_data = dataframe_to_tensor_dict(train_df, vocab, args.max_len)
    val_data = dataframe_to_tensor_dict(val_df, vocab, args.max_len)
    test_data = dataframe_to_tensor_dict(test_df, vocab, args.max_len)

    torch.save(train_data, output_dir / "train.pt")
    torch.save(val_data, output_dir / "val.pt")
    torch.save(test_data, output_dir / "test.pt")

    with open(output_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    meta = {
        "max_len": args.max_len,
        "vocab_size": len(vocab),
        "pad_id": vocab[PAD_TOKEN],
        "unk_id": vocab[UNK_TOKEN],
        "num_classes": 3,
        "label_names": ["negative", "neutral", "positive"],
    }
    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.show_stats:
        for name, split_df in [("TRAIN", train_df), ("VAL", val_df), ("TEST", test_df)]:
            counts = split_df["label_name"].value_counts().to_dict()
            print(
                f"[{name}] {len(split_df)} mau | "
                f"negative: {counts.get('negative', 0)} "
                f"neutral: {counts.get('neutral', 0)} "
                f"positive: {counts.get('positive', 0)}"
            )
        print(f"Vocab size: {len(vocab)} tu")
        print("Tao ra: data/processed/train.pt, val.pt, test.pt, vocab.json, meta.json")


if __name__ == "__main__":
    main()
