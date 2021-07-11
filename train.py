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


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_model(args, epoch, model, data_loader, optimizer, loss_fn, device):
    model = model.train()

    total_step = len(data_loader)
    epoch_loss = 0
    for batch_idx, batch in enumerate(data_loader):
            
        # Set mini-batch dataset
        images = batch[0].to(device)
        captions = batch[1].to(device)

        optimizer.zero_grad()
            
        # Forward, backward and optimize
        outputs = model(images, captions[:, :-1]) 
        outputs = outputs.transpose(1, 2) # batch_size, 11532, args.max_len

        loss = loss_fn(outputs, captions[:, 1:])
        loss.backward()
        optimizer.step()

        epoch_loss = epoch_loss + loss.item()
            
        if batch_idx % args.log_step == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                    .format(epoch, args.num_epochs, batch_idx, total_step, loss.item(), np.exp(loss.item()))) 

        # Save the model checkpoints
        if (batch_idx+1) % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(args.model_path, 'model-{}-{}.pt'.format(epoch+1, batch_idx+1)))

def validation_model(args, epoch, model, data_loader, optimizer, loss_fn, device):
    model.eval()

    total_step = len(data_loader)
    epoch_loss = 0
    validation_accuracy = 0
    for batch_idx, batch in enumerate(data_loader):
        images = batch[0].to(device)
        captions = batch[1].to(device)

        with torch.no_grad():
            outputs = model(images, captions[:, :-1])
                
        loss = loss_fn(outputs, captions[:, 1:])
        predictions = torch.argmax(outputs.detach().cpu(), dim=2)
        accuracy = (predictions == captions).sum() / len(captions) * 100

        validation_accuracy += accuracy

    print('Validation for Epoch [{}/{}] Finished, Validation Accuracy : {:.4f}'
        .format(epoch, args.num_epochs, validation_accuracy))


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.Resize((args.resize_size, args.resize_size)),
        transforms.RandomCrop(args.crop_size), # crop 224x224 from 256x256 image
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
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

    # Build the models
    model = CaptioningModel(args.batch_size, args.embed_size, len(vocabulary)).to(device)
    
    # Loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    # You don't need to train efficientnet
    params = list(list(model.linear_feature.parameters()) + list(model.embed.parameters()) + list(model.transformer.parameters()) + list(model.linear_out.parameters()))
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Epoch the models
    for epoch in tqdm(range(args.num_epochs)):
        # Train
        train_model(args, epoch, model, data_loader_train, optimizer, loss_fn, device)

        # Validation after each epoch
        validation_model(args, epoch, model, data_loader_val, optimizer, loss_fn, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--resize_size', type=int, default=256 , help='size for resizing images')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--max_len', type=int, default=400 , help='maximum length for each caption')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_train', type=str, default='dataset/train2017', help='directory for resized train images')
    parser.add_argument('--image_dir_val', type=str, default='dataset/val2017', help='directory for resized validation images')
    parser.add_argument('--caption_path_train', type=str, default='dataset/annotations/captions_train2017.json', help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str, default='dataset/annotations/captions_val2017.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=1000, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors, using Glove dim=300')
    
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
