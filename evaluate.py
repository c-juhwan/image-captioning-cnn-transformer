import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from model import CaptioningModel
from torchvision import transforms
from tqdm.auto import tqdm
from torchtext.vocab import Vocab


def main(args):
    transform = transforms.Compose([ 
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for loading trained models')
    parser.add_argument('--resize_size', type=int, default=224 , help='size for resizing images')
    parser.add_argument('--max_len', type=int, default=300 , help='maximum length for each caption')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_test', type=str, default='dataset/test2017', help='directory for resized validation images')

    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors, using Glove dim=300')
    
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
