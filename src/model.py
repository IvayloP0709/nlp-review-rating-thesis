"""A transformer classifier with an optional small-feature fusion head.

Replaces the original per-checkpoint `CustomSequenceClassification(BertForSequenceClassification)`
/ `...RobertaForSequenceClassification)` pair (two near-identical classes, one per checkpoint) with
one model class parametrized by checkpoint name via `AutoModel`/`AutoConfig`.

This also fixes the original feature-fusion bug: the old code built `extra_features` as one tensor
over the *entire* dataset in its original row order, then every batch did
`extra_features[:len(input_ids)]` -- always the first `batch_size` rows, regardless of which
(shuffled) examples were actually in that batch. Here, `extra_features` is just another column of
the batch dict, so it is always the right rows for the text it's fused with.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput


class FusionClassificationHead(nn.Module):
    """[CLS] embedding, optionally concatenated with extra numeric features, -> logits."""

    def __init__(self, hidden_size: int, num_extra_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        in_features = hidden_size + num_extra_features
        self.dense = nn.Linear(in_features, in_features)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(in_features, num_classes)

    def forward(self, cls_embedding: torch.Tensor, extra_features: Optional[torch.Tensor]) -> torch.Tensor:
        x = cls_embedding if extra_features is None else torch.cat([cls_embedding, extra_features], dim=-1)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class FusionForSequenceClassification(PreTrainedModel):
    """Works with any encoder-only checkpoint (e.g. bert-base-uncased, roberta-base)."""

    def __init__(self, config, num_extra_features: int = 0, num_classes: int = 5, dropout: float = 0.1):
        super().__init__(config)
        self.encoder = AutoModel.from_config(config)
        self.classifier = FusionClassificationHead(config.hidden_size, num_extra_features, num_classes, dropout)
        self.post_init()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        num_extra_features: int = 0,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> "FusionForSequenceClassification":
        config = AutoConfig.from_pretrained(checkpoint)
        model = cls(config, num_extra_features=num_extra_features, num_classes=num_classes, dropout=dropout)
        model.encoder = AutoModel.from_pretrained(checkpoint, config=config)
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        extra_features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        # `labels` is accepted but unused here -- WeightedLossTrainer.compute_loss pops it from
        # the batch itself. It still needs to be a declared parameter: Trainer inspects this
        # signature to decide which batch keys to keep, and silently drops anything it doesn't
        # recognize before compute_loss ever sees the batch.
        del labels
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding, extra_features)
        return SequenceClassifierOutput(logits=logits)


class WeightedLossTrainer(Trainer):
    """HF Trainer with an optional per-class weighted cross-entropy loss.

    Replaces the hand-rolled train/eval loops, gradient accumulation, and early-stopping
    bookkeeping duplicated across every original run script -- those are exactly what
    TrainingArguments + Trainer (+ EarlyStoppingCallback) already provide.
    """

    def __init__(self, *args, class_weights: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss
