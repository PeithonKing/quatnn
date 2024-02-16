from torch import nn
import torch

# nn.MultiheadAttention
# nn.TransformerEncoder
# nn.TransformerDecoder
# nn.TransformerEncoderLayer
# nn.TransformerDecoderLayer
# nn.Transformer

# from .library import quatnn

# quatnn.QMultiheadAttention
# quatnn.QTransformerEncoder
# quatnn.QTransformerDecoder
# quatnn.QTransformerEncoderLayer
# quatnn.QTransformerDecoderLayer
# quatnn.QTransformer

# import transformer

# print(transformer.__all__)


# nn.Linear


model = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)

x = torch.randn(10, 4, 32, 32)

print(model(x).shape)