"""Pretrained Transformer Classification Architecture and Pipeline.

Lab 08 — Pretrained Transformers and BERT Fine-Tuning
Môn học: Xử lý ngôn ngữ tự nhiên và ứng dụng (NLP)
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


class PretrainedClassifier(nn.Module):
    """Mô hình phân loại văn bản sử dụng Pretrained Transformer Encoder (BERT).

    Kiến trúc Pipeline theo Mục 13:
    Input text -> Tokenizer -> Input IDs -> Pretrained Transformer ->
    Contextual representations -> [CLS] representation -> Linear layer -> Class logits
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        from_scratch: bool = False,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.model_name = model_name
        self.freeze_encoder = freeze_encoder

        if from_scratch:
            # Khởi tạo cùng kiến trúc Transformer nhưng với trọng số ngẫu nhiên
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        else:
            # Khởi tạo từ trọng số pre-trained (BERT checkpoint)
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)

        # Chế độ Feature Extraction: đóng băng toàn bộ tham số của encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        # Task-specific classification head
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi (Forward pass).

        Args:
            input_ids: Tensor chỉ số token [batch_size, seq_len]
            attention_mask: Tensor mặt nạ chú ý [batch_size, seq_len]

        Returns:
            logits: Tensor điểm phân loại chưa chuẩn hóa [batch_size, num_classes]
        """
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Trích xuất biểu diễn ẩn tại vị trí token đầu tiên [CLS] (index 0)
        cls_representation = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(cls_representation))
        return logits

    def predict(self, text: str, tokenizer, device: str = "cpu"):
        """Dự đoán cảm xúc cho một câu văn bản."""
        self.eval()
        inputs = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            logits = self.forward(inputs["input_ids"], inputs["attention_mask"])
            probs = F.softmax(logits, dim=-1)
            confidence, pred_class = torch.max(probs, dim=-1)

        return pred_class.item(), confidence.item(), probs.squeeze().tolist()


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Đếm tổng số tham số hoặc số tham số có thể huấn luyện của mô hình."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Test PretrainedClassifier pipeline")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()

    print(f"Loading tokenizer & model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = PretrainedClassifier(
        model_name=args.model_name,
        num_classes=2,
        from_scratch=args.from_scratch,
        freeze_encoder=args.freeze_encoder,
    )
    model.eval()

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    test_sentences = [
        "This movie is an absolute masterpiece of cinema!",
        "Terrible acting, boring plot, complete waste of time.",
    ]

    print("\n--- Testing forward pass on sample sentences ---")
    inputs = tokenizer(test_sentences, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = F.softmax(logits, dim=-1)

    labels = ["Negative", "Positive"]
    for text, prob, logit in zip(test_sentences, probs, logits):
        pred_idx = torch.argmax(prob).item()
        print(f"\nText: '{text}'")
        print(f"Logits: {logit.tolist()}")
        print(f"Probabilities: Neg={prob[0]:.4f}, Pos={prob[1]:.4f}")
        print(f"Prediction: {labels[pred_idx]} (confidence: {prob[pred_idx]:.4f})")


if __name__ == "__main__":
    main()
