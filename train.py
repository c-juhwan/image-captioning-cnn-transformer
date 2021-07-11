import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader
from torchvision import transforms
from tqdm.auto import tqdm
from torchtext.vocab import Vocab
from model import CaptioningModel
import time
from datetime import timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(args, epoch_idx, model, data_loader, optimizer, loss_fn, device=device):
    print("Start training epoch[{}/{}]".format(epoch_idx+1, args.num_epochs))
    start_time = time.time()
    model = model.train()

    epoch_loss = 0
    epoch_acc = 0

    for batch_idx, batch in enumerate(data_loader):
        images = batch[0].to(device)
        captions = batch[1].to(device)

        labels = captions[:, 1:]
        non_pad = labels != 0 # <pad> = 0

        optimizer.zero_grad()

        # Exclude <end> from input -> we don't need to feed <end>
        outputs = model(images, captions[:, :-1], non_pad) # outputs: (batch_size, len, vocab_size)
        outputs = outputs.transpose(1, 2) # outputs: (batch_size, vocab_size, len)

        predictions = outputs.contoguous().view(-1, outputs.size(-1))
        labels = labels[non_pad].contiguous().view(-1) # model didn't generate <start>

        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        acc = (predictions.argmax(dim=1) == labels).sum() / len(labels)

        epoch_loss += loss.item()
        epoch_acc += (acc.item() * 100)

    elapsed_time = time.time() - start_time
    elapsed_time = str(timedelta(seconds=elapsed_time))
    average_loss = epoch_loss / len(data_loader)

    print("End training epoch[{}/{}], Elapsed time = {}, Average loss = {}, Train acc = {}"
          .format(epoch_idx+1, args.num_epochs, elapsed_time, average_loss, epoch_acc))


def validation_model(args, epoch_idx, model, data_loader, loss_fn, device=device):
    print("Start validation for epoch[{}/{}]".format(epoch_idx+1, args.num_epochs))
    start_time = time.time()
    model = model.eval()

    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch[0].to(device)
            captions = batch[1].to(device)

            labels = captions[:, 1:]
            non_pad = labels != 0 # <pad> = 0

            # Exclude <end> from input -> we don't need to feed <end>
            outputs = model(images, captions[:, :-1], non_pad) # outputs: (batch_size, len, vocab_size)
            outputs = outputs.transpose(1, 2) # outputs: (batch_size, vocab_size, len)

            predictions = outputs.contoguous().view(-1, outputs.size(-1))
            labels = labels[non_pad].contiguous().view(-1) # model didn't generate <start>

            loss = loss_fn(predictions, labels)
            acc = (predictions.argmax(dim=1) == labels).sum() / len(labels)

            epoch_loss += loss.item()
            epoch_acc += (acc.item() * 100)

        elapsed_time = time.time() - start_time
        elapsed_time = str(timedelta(seconds=elapsed_time))
        average_loss = epoch_loss / len(data_loader)

    print("End validation for epoch[{}/{}], Elapsed time = {}, Average loss = {}, Accuracy = {}"
          .format(epoch_idx+1, args.num_epochs, elapsed_time, average_loss, epoch_acc))



def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained cnn backbone
    transform = transforms.Compose([ 
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary from build_vocab.py
    with open(args.vocab_path, 'rb') as f:
        # use saved vocab file from main() of build_vocab.py
        vocabulary = pickle.load(f)
    
    # Build data loader
    data_loader_train = get_loader(args.image_dir_train, args.caption_path_train, vocabulary, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, max_len=args.max_len) 
    data_loader_val = get_loader(args.image_dir_val, args.caption_path_val, vocabulary, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers, max_len=args.max_len) 
    
    # Build Model
    model = CaptioningModel(args.batch_size, len(vocabulary), args.embed_size, args.d_model, args.max_len)

    # Define loss function and optimizer

    loss_fn = nn.CrossEntropyloss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    for epoch_idx in tqdm(range(args.num_epochs)):
        train_model()

        validation_model()
        


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='CaptioningModel', help='name of model')
    parser.add_argument('--model_path', type=str, default='models/', help='path for saving trained models')
    parser.add_argument('--resize_size', type=int, default=224, help='size for resizing images')
    parser.add_argument('--max_len', type=int, default=300, help='maximum length for each caption')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_train', type=str, default='dataset/train2017', help='directory for resized train images')
    parser.add_argument('--image_dir_val', type=str, default='dataset/val2017', help='directory for resized validation images')
    parser.add_argument('--caption_path_train', type=str, default='dataset/annotations/captions_train2017.json', help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str, default='dataset/annotations/captions_val2017.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of transformer model')
    
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
