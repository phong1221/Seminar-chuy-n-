import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TODO 1: Sinh viên tự cài đặt scaled_dot_product_attention
# ============================================================
def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: shape (batch_size, seq_len, d_k)
    Return:
        output  : shape (batch_size, seq_len, d_k)
        weights : shape (batch_size, seq_len, seq_len)
    """
    d_k = Q.size(-1)

    # scores = Q @ K^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ap dung softmax tren chieu cuoi
    weights = F.softmax(scores, dim=-1)

    # tinh output = weights @ V
    output = torch.matmul(weights, V)

    return output, weights


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class SelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # TODO 2: Sinh vien tu cai dat FFN = Linear(d_model, d_ff) -> ReLU -> Linear(d_ff, d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO 3: Viet forward pass cua FFN
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.self_attention = SelfAttention(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.self_attention(x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x, attn_weights


class ClassifierHead(nn.Module):
    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.fc(pooled)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_ff: int, max_len: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
        self.encoder = TransformerEncoderBlock(d_model=d_model, d_ff=d_ff)
        self.classifier = ClassifierHead(d_model=d_model, num_classes=num_classes)
        self.last_attention_weights = None

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        x, attn_weights = self.encoder(x)
        self.last_attention_weights = attn_weights
        logits = self.classifier(x)
        return logits


# ============================================================
# Unit tests de giang vien / sinh vien tu kiem tra model.py
# ============================================================
def _test_scaled_dot_product_attention():
    Q = torch.randn(2, 10, 32)
    K = torch.randn(2, 10, 32)
    V = torch.randn(2, 10, 32)
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (2, 10, 32)
    assert weights.shape == (2, 10, 10)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 10), atol=1e-5)


def _test_self_attention():
    x = torch.randn(2, 10, 64)
    layer = SelfAttention(d_model=64)
    out, weights = layer(x)
    assert out.shape == (2, 10, 64)
    assert weights.shape == (2, 10, 10)


def _test_ffn():
    x = torch.randn(2, 10, 64)
    ffn = FeedForwardNetwork(d_model=64, d_ff=128)
    out = ffn(x)
    assert out.shape == (2, 10, 64)


def _test_encoder_block():
    x = torch.randn(2, 10, 64)
    block = TransformerEncoderBlock(d_model=64, d_ff=128)
    out, weights = block(x)
    assert out.shape == (2, 10, 64)
    assert weights.shape == (2, 10, 10)


def run_tests():
    print("TEST: scaled_dot_product_attention ...", end=" ")
    _test_scaled_dot_product_attention()
    print("PASSED")

    print("TEST: SelfAttention ................", end=" ")
    _test_self_attention()
    print("PASSED")

    print("TEST: FeedForwardNetwork ...........", end=" ")
    _test_ffn()
    print("PASSED")

    print("TEST: TransformerEncoderBlock ......", end=" ")
    _test_encoder_block()
    print("PASSED")

    print("TAT CA TESTS PASSED -- model.py san sang de huan luyen!")


if __name__ == "__main__":
    run_tests()
