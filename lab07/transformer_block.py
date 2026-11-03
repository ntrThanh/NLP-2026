"""
Module: transformer_block.py
Mục đích:
- Cài đặt kiến trúc Transformer Encoder Block tối giản từ các thành phần cơ bản (primitive layers)
  theo đúng yêu cầu của LAB 07 (Vaswani et al., 2017).
- Hỗ trợ trích xuất attention weights phục vụ kiểm tra và trực quan hóa các Attention Heads.
- Định nghĩa mô hình TransformerClassifier phục vụ bài toán Text Classification trên tập IMDb.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding (Vaswani et al., 2017).
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor kích thước (B, T, d_model)
        Output: (B, T, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention tự ghép từ Linear layers và Scaled Dot-Product Attention.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) phải chia hết cho n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Ma trận chiếu cho Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

        # Ma trận chiếu đầu ra W^O
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ):
        """
        q, k, v: Tensor kích thước (B, T, d_model)
        mask: Tensor kích thước (B, 1, 1, T) hoặc (B, 1, T, T) (tùy chọn)
        Trả về:
            output: (B, T, d_model)
            attn_weights: (B, n_heads, T, T)
        """
        B, T, _ = q.size()

        # 1. Chiếu tuyến tính và phân tách thành nhiều heads:
        # (B, T, d_model) -> (B, T, n_heads, d_k) -> (B, n_heads, T, d_k)
        Q = self.w_q(q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(k).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(v).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Scaled Dot-Product Attention:
        # scores: (B, n_heads, T, d_k) @ (B, n_heads, d_k, T) -> (B, n_heads, T, T)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_drop = self.dropout(attn_weights)

        # 3. Nhân với Value V: (B, n_heads, T, T) @ (B, n_heads, T, d_k) -> (B, n_heads, T, d_k)
        context = torch.matmul(attn_weights_drop, V)

        # 4. Ghép nối (Concat) các heads:
        # (B, n_heads, T, d_k) -> (B, T, n_heads, d_k) -> (B, T, d_model)
        concat = context.transpose(1, 2).contiguous().view(B, -1, self.d_model)

        # 5. Chiếu qua ma trận đầu ra W^O:
        output = self.w_o(concat)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network:
    FFN(x) = max(0, x W1 + b1) W2 + b2
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor kích thước (B, T, d_model)
        Output: (B, T, d_model)
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Một khối Transformer Encoder hoàn chỉnh (Mục 17):
    x -> Multi-Head Attention -> Add & Norm -> Feed-Forward -> Add & Norm -> output
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        x: Tensor kích thước (B, T, d_model)
        Trả về:
            out: (B, T, d_model)
            attn_weights: (B, n_heads, T, T)
        """
        # 1. Multi-head Self-Attention + Residual + LayerNorm (Post-LN)
        attn_out, attn_weights = self.mha(x, x, x, mask=mask)
        x = self.norm1(x + self.dropout1(attn_out))

        # 2. Feed-Forward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        out = self.norm2(x + self.dropout2(ffn_out))

        return out, attn_weights


class TransformerClassifier(nn.Module):
    """
    Mô hình Transformer Classifier cho Text Classification (Mục 23):
    Embedding -> Positional Encoding -> Transformer Block -> Pooling -> Linear Classifier
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        num_classes: int = 1,
        pooling: str = "mean",
        dropout: float = 0.1,
        max_len: int = 500,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.pooling = pooling
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoder = SinusoidalPositionalEncoding(
            d_model, max_len=max_len, dropout=dropout
        )
        self.transformer_block = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
        )
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, input_ids: torch.Tensor, return_attn: bool = False):
        """
        input_ids: Tensor (B, T)
        """
        # Padding mask: 1 cho token thật, 0 cho token <PAD>
        # mask shape: (B, 1, 1, T)
        mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # 1. Embedding có co giãn căn bậc hai của d_model (chuẩn Vaswani et al.)
        x = self.embedding(input_ids) * math.sqrt(self.d_model)

        # 2. Cộng Positional Encoding
        x = self.pos_encoder(x)

        # 3. Transformer Encoder Block
        x, attn_weights = self.transformer_block(x, mask=mask)

        # 4. Pooling strategy (Mục 24 & 25)
        if self.pooling == "mean":
            # Masked Mean Pooling: chỉ tính trung bình trên các token thật
            mask_float = (input_ids != self.pad_idx).unsqueeze(-1).float()  # (B, T, 1)
            sum_embeddings = torch.sum(x * mask_float, dim=1)
            sum_mask = torch.clamp(mask_float.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        elif self.pooling in ("cls", "first"):
            # First token pooling
            pooled = x[:, 0, :]
        else:
            raise ValueError(f"Pooling không hỗ trợ: {self.pooling}")

        # 5. Phân lớp đầu ra
        logits = self.classifier(pooled).squeeze(-1)

        if return_attn:
            return logits, attn_weights
        return logits


if __name__ == "__main__":
    # Test shape theo Mục 18 W7.pdf:
    # B = 8, T = 100, d_model = 128, n_heads = 4 -> d_k = 32
    B, T, d_model, n_heads, d_ff = 8, 100, 128, 4, 256
    dummy_x = torch.randn(B, T, d_model)
    block = TransformerBlock(d_model, n_heads, d_ff)
    out, attn = block(dummy_x)

    print("Kiểm tra shape (Mục 18):")
    print(f"Input shape:        {dummy_x.shape}")
    print(f"Output shape:       {out.shape} (Kỳ vọng: [8, 100, 128])")
    print(f"Attn weights shape: {attn.shape} (Kỳ vọng: [8, 4, 100, 100])")
    assert out.shape == (8, 100, 128)
    assert attn.shape == (8, 4, 100, 100)
    print("-> Tất cả kiểm tra shape ĐẠT chuẩn 100%!")
