import torch
import torch.nn as nn
import math
from efficientnet_pytorch import EfficientNet


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CaptioningModel(nn.Module):
    def __init__(self, batch_size, vocab_size, embed_size=256, d_model=512, max_len=300):
        """
        Load pretrained EfficientNet-b7 and initialize other layers
        """
        super(CaptioningModel, self).__init__()
        self.batch_size = batch_size

        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7')
        self.efficientnet.requires_grad_(False)
        self.feature_size = self.efficientnet._fc.in_features

        self.pixel_embed = PixelEmbedding()
        self.linear_feature = nn.Linear(self.feature_size, d_model)

        self.text_embed = TextEmbedding(vocab_size, embed_size, d_model, max_len, padding_idx=0)

        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        #decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        #self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)

        self.linear_out = nn.Linear(d_model, embed_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout()
        self.norm = nn.LayerNorm(embed_size)
        self.linear_out2 = nn.Linear(embed_size, vocab_size)
    
    def forward(self, images, captions, tgt_mask, non_pad_pos):
        """ 
        Args:

        Return:
            torch.Tensor with size (batch_size, max_len, vocab_size)
            word estimation
        """
        # images: (batch_size, 3, H, W) -> default HxW = 224x224 
        # captions: (batch_size, max_len) -> default max_len is 300, caption[1] is vocabulary index

        tgt_key_padding_mask = (captions == 0) # <pad> = 0

        with torch.no_grad():
            features = self.efficientnet.extract_features(images) # Extract feature from EfficientNet
            # features: (batch_size, 2560, H', W') -> H'xW' = 7x7 in case of 224x224

        features = features.view(features.size(0), features.size(1), -1) # features: (batch_size, 2560, 49)
        features = features.transpose(1, 2) # features: (batch_size, 49, 2560)
        features = self.pixel_embed(features) # features: (batch_size, 50, 2560)
        features = self.linear_feature(features) # features: (batch_size, 50, d_model)

        embedding = self.text_embed(captions) # embedding: (batch_size, max_len, d_model)

        out = self.transformer(src=features, tgt=embedding, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        # <nn.Trasformer>
        # src=features: (batch_size, 50, 512) -> N=batch_size, S=50=Source Seq Length, E=d_model=feature number
        # tgt=embedding: (batch_size, 300, 512) -> N=batch_size, T=max_len=Target Seq Length, E=d_model=feature number
        # out: (batch_size, 300, 512) -> N, T, E

        # nn.TransformerDecoder Requires (sequence_length, batch_size, d_model)
        # out = self.transformer_decoder(tgt=embedding.transpose(0, 1), memory=features.transpose(0, 1))

        out = out[non_pad_pos] # delete padding part

        out = self.norm(self.dropout(self.activation(self.linear_out(out)))) # out: (length of all batch, embed_size)
        out = self.linear_out2(out) # out: (length of all batch, vocab_size)

        return out

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask

        
class PixelEmbedding(nn.Module):
    """
    Gives spatial information to each pixel extracted by cnn backbone
    """
    def __init__(self):
        super(PixelEmbedding, self).__init__()
    
    def forward(self, features):
        batch_size = features.size(0)
        pixel_num = features.size(1) # 7x7 -> 49
        feature_num = features.size(2) # 2560

        cls_token = nn.Parameter(torch.randn(1, feature_num)).to(device)
        positions = nn.Parameter(torch.randn(pixel_num + 1, feature_num)).to(device) # +1 for cls_token
        
        cls_tokens = cls_token.repeat(batch_size, 1, 1) # cls_token: (batch_size, 1, feature_num)

        out = torch.cat([cls_tokens, features], dim=1) # out: (batch_size, 1+49, feature_num)
        out += positions # out: (batch_size, 50, feature_num) -> Broadcasting

        return out

class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, d_model, max_len, padding_idx=0):
        super(TextEmbedding, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx)
        self.linear = nn.Linear(embed_size, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)

        self.pos_encoding = PositionalEncoding(d_model, max_seq_len=max_len)

    def forward(self, sentence):
        # sentence: (batch_size, max_len)
        embedding = self.embed(sentence) # embedding: (batch_size, max_len, embed_size)
        embedding = self.dropout(self.activation(self.linear(embedding))) # embedding: (batch_size, max_len, d_model)

        pos = self.pos_encoding(sentence) # pos: (max_len, d_model)

        out = self.norm(embedding + pos)
        return out

class PositionalEncoding(nn.Module):
    """
    Reference: http://incredible.ai/nlp/2020/02/29/Transformer/#232-github에-구현한-코드-pytorch-문서-참고
    """

    def __init__(self, embed_dim, dropout=0.1, max_seq_len=300):
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
