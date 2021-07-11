import torch
import torch.nn as nn
import numpy as np
import math
from efficientnet_pytorch import EfficientNet


class CaptioningModel(nn.Module):
    def __init__(self, batch_size, embed_size, vocab_size):
        super(CaptioningModel, self).__init__()
        self.batch_size = batch_size
        self.feature_size = self.efficientnet._fc.in_features

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.pixel_embed = PixelEmbedding()
        self.linear_feature = nn.Linear(self.feature_size, embed_size)
        self.text_embed = TextEmbedding(vocab_size, embed_size, padding_idx=0)
        self.transformer = nn.Transformer(d_model=embed_size, batch_first=True)
        self.linear_out = nn.Linear(embed_size, vocab_size)

    def forward(self, images, captions):
        with torch.no_grad():
            features = self.efficientnet.extract_features(images)
        features = features.view(self.batch_size, self.feature_size, -1)
        features = features.transpose(1, 2) # batch_size, 49, 2560

        features = self.pixel_embed(features)

        features = self.linear_feature(features)

        text_embedding = self.text_embed(captions)

        out = self.transformer(src=features, tgt=text_embedding)
        out = self.linear_out(out)

        return out


class PixelEmbedding(nn.Module):
    """
    Gives spatial information to each pixel extracted by cnn backbone
    """
    def __init__(self):
        super(PixelEmbedding, self).__init__()
    
    def forward(self, features):
        batch_size = features.size(0)
        pixel_size = features.size(1)
        feature_size = features.size(2)

        cls_token = nn.Parameter(torch.randn(1, feature_size))
        positions = nn.Parameter(torch.randn(pixel_size + 1, feature_size))

        cls_tokens = cls_token.repeat(batch_size, 1, 1)

        out = torch.cat([cls_tokens, features], dim=1)
        out += positions

        return out

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, padding_idx):
        super(TextEmbedding, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.pos_encoding = PositionalEncoding(embed_size)


    def forward(self, sentence):
        embedding = self.embed(sentence)
        pos = self.pos_encoding(sentence)

        return embedding + pos

class PositionalEncoding(nn.Module):
    """
    Reference: http://incredible.ai/nlp/2020/02/29/Transformer/#232-github에-구현한-코드-pytorch-문서-참고
    """

    def __init__(self, embed_dim, dropout=0.1, max_seq_len=400):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)].detach()