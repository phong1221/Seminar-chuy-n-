# -*- coding: utf-8 -*-
"""
Script tự động tạo heatmap cho phần 5.3 của báo cáo.
Tạo heatmap cho:
  - Câu phân loại ĐÚNG (positive, negative, neutral)
  - Câu phân loại SAI
  - Câu có từ phủ định (not, never, don't)
"""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from model import TransformerClassifier

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"


def tokenize(text: str):
    return text.strip().lower().split()


def encode_text(text: str, vocab: dict, max_len: int):
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab.get(UNK_TOKEN, 1)) for tok in tokens][:max_len]
    length = len(ids)
    if length < max_len:
        ids += [vocab.get(PAD_TOKEN, 0)] * (max_len - length)
    return ids, tokens[:max_len]


def load_model(model_path: Path, meta: dict):
    stem = model_path.stem.replace("model_", "")
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
    return model


def predict_and_get_attention(model, sentence, vocab, meta):
    input_ids, tokens = encode_text(sentence, vocab, meta["max_len"])
    with torch.no_grad():
        logits = model(torch.tensor([input_ids], dtype=torch.long))
        pred = logits.argmax(dim=-1).item()
        weights = model.last_attention_weights[0, :len(tokens), :len(tokens)].cpu().numpy()
    return pred, meta["label_names"][pred], tokens, weights


def save_heatmap(tokens, weights, pred_label, true_label, sentence, save_path, title_extra=""):
    is_correct = pred_label == true_label
    status = "✓ ĐÚNG" if is_correct else "✗ SAI"

    plt.figure(figsize=(7, 6))
    plt.imshow(weights, cmap="YlOrRd")
    plt.colorbar(shrink=0.8)
    plt.xticks(range(len(tokens)), tokens, rotation=45, ha="right", fontsize=10)
    plt.yticks(range(len(tokens)), tokens, fontsize=10)
    plt.xlabel("Key", fontsize=11)
    plt.ylabel("Query", fontsize=11)

    title = f"Attention Heatmap | pred={pred_label} | true={true_label} | {status}"
    if title_extra:
        title += f"\n{title_extra}"
    plt.title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {save_path}")


def main():
    processed_dir = Path("data/processed")
    results_dir = Path("results")

    vocab = json.loads((processed_dir / "vocab.json").read_text(encoding="utf-8"))
    meta = json.loads((processed_dir / "meta.json").read_text(encoding="utf-8"))

    # Use best model (d128_ff256)
    model_path = results_dir / "model_Transformer_d128_ff256.pt"
    if not model_path.exists():
        candidates = sorted(results_dir.glob("model_Transformer*.pt"))
        model_path = candidates[0]
    print(f"Using model: {model_path}")
    model = load_model(model_path, meta)

    # Load test data để tìm câu đúng/sai
    test_data = torch.load(processed_dir / "test.pt")
    test_texts = test_data["texts"]
    test_labels = test_data["labels"].tolist()
    test_input_ids = test_data["input_ids"]

    # === Phân loại tất cả câu test ===
    correct_sentences = {"positive": [], "negative": [], "neutral": []}
    wrong_sentences = []

    with torch.no_grad():
        logits = model(test_input_ids)
        preds = logits.argmax(dim=-1).tolist()

    for i, (text, true_label_id, pred_id) in enumerate(zip(test_texts, test_labels, preds)):
        true_name = meta["label_names"][true_label_id]
        pred_name = meta["label_names"][pred_id]
        if pred_id == true_label_id:
            correct_sentences[true_name].append((text, true_name))
        else:
            wrong_sentences.append((text, true_name, pred_name))

    print(f"\nTest set results:")
    print(f"  Correct: {sum(len(v) for v in correct_sentences.values())} sentences")
    print(f"  Wrong:   {len(wrong_sentences)} sentences")

    # ===================================================
    # CÂU 1: Heatmap cho câu phân loại ĐÚNG
    # ===================================================
    print("\n" + "=" * 60)
    print("SENTENCE 1: Correctly classified sentences")
    print("=" * 60)

    for label_name in ["positive", "negative", "neutral"]:
        if correct_sentences[label_name]:
            sentence, true_label = correct_sentences[label_name][0]
            pred_id, pred_label, tokens, weights = predict_and_get_attention(model, sentence, vocab, meta)
            filename = f"heatmap_correct_{label_name}.png"
            print(f"\n  [{label_name.upper()}] \"{sentence}\" -> pred={pred_label}")
            save_heatmap(
                tokens, weights, pred_label, true_label, sentence,
                results_dir / filename,
                title_extra=f"Câu 1: Phân loại đúng ({label_name})"
            )

    # ===================================================
    # CÂU 2: Heatmap cho câu phân loại SAI
    # ===================================================
    print("\n" + "=" * 60)
    print("SENTENCE 2: Incorrectly classified sentences")
    print("=" * 60)

    if wrong_sentences:
        for i, (sentence, true_label, pred_label) in enumerate(wrong_sentences[:2]):
            _, pred_label_check, tokens, weights = predict_and_get_attention(model, sentence, vocab, meta)
            filename = f"heatmap_wrong_{i + 1}.png"
            print(f"\n  \"{sentence}\" -> true={true_label}, pred={pred_label_check}")
            save_heatmap(
                tokens, weights, pred_label_check, true_label, sentence,
                results_dir / filename,
                title_extra=f"Câu 2: Phân loại sai (true={true_label})"
            )
    else:
        print("  Model classified all test sentences correctly!")
        print("  Creating heatmaps for hard/ambiguous sentences...")
        hard_sentences = [
            ("the acting is not terrible at all", "positive"),
            ("i would not say this film is bad", "positive"),
        ]
        for i, (sentence, expected) in enumerate(hard_sentences):
            _, pred_label, tokens, weights = predict_and_get_attention(model, sentence, vocab, meta)
            filename = f"heatmap_wrong_{i + 1}.png"
            print(f"\n  \"{sentence}\" -> pred={pred_label} (expected={expected})")
            save_heatmap(
                tokens, weights, pred_label, expected, sentence,
                results_dir / filename,
                title_extra=f"Câu 2: Câu khó/ambiguous (expected={expected})"
            )

    # ===================================================
    # CÂU 3: Heatmap cho câu có từ phủ định
    # ===================================================
    print("\n" + "=" * 60)
    print("SENTENCE 3: Sentences with negation words (not, never, don't)")
    print("=" * 60)

    negation_sentences = [
        ("this movie is not bad at all", "positive"),
        ("i will never watch this terrible film", "negative"),
        ("the acting is not great", "negative"),
    ]

    for i, (sentence, expected_label) in enumerate(negation_sentences):
        _, pred_label, tokens, weights = predict_and_get_attention(model, sentence, vocab, meta)
        filename = f"heatmap_negation_{i + 1}.png"
        print(f"\n  \"{sentence}\" -> pred={pred_label} (expected={expected_label})")
        save_heatmap(
            tokens, weights, pred_label, expected_label, sentence,
            results_dir / filename,
            title_extra=f"Câu 3: Có từ phủ định"
        )

    print("\n" + "=" * 60)
    print("DONE! All heatmaps saved to results/ folder")
    print("=" * 60)


if __name__ == "__main__":
    main()
