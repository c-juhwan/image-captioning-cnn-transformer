import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class CaptioningModel(nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size):
        super(CaptioningModel, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.batch_size = batch_size
        self.feature_size = self.efficientnet._fc.in_features
        self.linear_feature = nn.Linear(self.feature_size, embed_size)
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.transformer = nn.Transformer(d_model=embed_size, batch_first=True)
        self.linear_out = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.efficientnet.extract_features(images)
        features = features.view(self.batch_size, self.feature_size, -1)
        features = features.transpose(1, 2)
        features = self.linear_feature(features)

        embedding = self.embed(captions)

        out = self.transformer(src=features, tgt=embedding)
        out = self.linear_out(out)

        return out


