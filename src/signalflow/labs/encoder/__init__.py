from signalflow.labs.encoder.conv1d import Conv1dEncoder
from signalflow.labs.encoder.convtran import ConvTranEncoder
from signalflow.labs.encoder.gmlp import gMLPEncoder
from signalflow.labs.encoder.gru import GRUEncoder
from signalflow.labs.encoder.inception import InceptionTimeEncoder
from signalflow.labs.encoder.itransformer import iTransformerEncoder
from signalflow.labs.encoder.lstm import LSTMEncoder
from signalflow.labs.encoder.mamba import MambaEncoder
from signalflow.labs.encoder.omniscale import OmniScaleCNNEncoder
from signalflow.labs.encoder.patchtst import PatchTSTEncoder
from signalflow.labs.encoder.resnet1d import ResNet1dEncoder
from signalflow.labs.encoder.tcn import TCNEncoder
from signalflow.labs.encoder.transformer import TransformerEncoder
from signalflow.labs.encoder.tsmixer import TSMixerEncoder
from signalflow.labs.encoder.xception import XceptionTimeEncoder
from signalflow.labs.encoder.xcm import XCMEncoder

__all__ = [
    # CNN-based
    "Conv1dEncoder",
    "ConvTranEncoder",
    "GRUEncoder",
    "InceptionTimeEncoder",
    # RNN-based
    "LSTMEncoder",
    # State Space Models
    "MambaEncoder",  # Mamba SSM (O(T) complexity)
    "OmniScaleCNNEncoder",
    "PatchTSTEncoder",
    "ResNet1dEncoder",
    "TCNEncoder",
    # Mixer-based
    "TSMixerEncoder",
    # Transformer-based
    "TransformerEncoder",
    "XCMEncoder",
    "XceptionTimeEncoder",
    "gMLPEncoder",
    "iTransformerEncoder",  # Inverted Transformer (ICLR 2024)
]
