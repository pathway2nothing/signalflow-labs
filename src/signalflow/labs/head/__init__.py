"""
Classification and regression heads for temporal models.

Supports:
- Hard classification (2-5 classes)
- Soft labeling (regression/distribution prediction)
- Various architectures (MLP, Attention, Residual, Ordinal)

All heads follow the interface:
    __init__(input_size: int, num_classes: int, **kwargs)
    forward(x: Tensor[batch, input_size]) -> Tensor[batch, num_classes]
"""

from signalflow.labs.head.attention_head import AttentionClassifierHead
from signalflow.labs.head.confidence_head import ClassificationWithConfidenceHead
from signalflow.labs.head.distribution_head import DistributionHead
from signalflow.labs.head.linear_head import LinearClassifierHead
from signalflow.labs.head.mlp_head import MLPClassifierHead
from signalflow.labs.head.ordinal_head import OrdinalRegressionHead
from signalflow.labs.head.residual_head import ResidualClassifierHead

__all__ = [
    "AttentionClassifierHead",
    "ClassificationWithConfidenceHead",
    "DistributionHead",
    "LinearClassifierHead",
    "MLPClassifierHead",
    "OrdinalRegressionHead",
    "ResidualClassifierHead",
]
