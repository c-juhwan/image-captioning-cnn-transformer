import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
import nltk
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import Encoder, Decoder
from torchvision import transforms
from tqdm.auto import tqdm
import nltk.translate.bleu_score as bleu


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    # bleu.corpus_bleu()

    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = Encoder(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = Decoder(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    data_loader_test = get_loader(args.image_dir_test, args.caption_path_test, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.ckpt', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.ckpt', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_test', type=str, default='dataset/resized_val2017', help='directory for resized test images')
    parser.add_argument('--caption_path_test', type=str, default='dataset/annotations/captions_train2017.json', help='path for test annotation json file')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=2)
    args = parser.parse_args()
    print(args)
    main(args)
