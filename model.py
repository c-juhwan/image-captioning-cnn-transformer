import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class Encoder(nn.Module):
    def __init__(self, embed_size):
        """Load pretrained EfficientNet as feature extractor."""
        super(Encoder, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        feature_size = self.efficientnet._fc.in_features
        self.linear = nn.Linear(feature_size, embed_size)

    def forward(self, images):
        """Extract feature vectors from input images."""
        with torch.no_grad():
            features = self.efficientnet.extract_features(images)
        features = self.linear(features)

        return features


class Decoder(nn.Module):
    def __init__(self, embed_size, vocab_size):
        """Decode image feature vectors and generates captions."""
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)
    
    def forward(self, features, captions):
        embedding = self.embed(captions)
        out = self.transformer_decoder(tgt=embedding, memory=features)

        print(out)
        print(out.size())

        return out
        