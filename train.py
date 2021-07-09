import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import Encoder, Decoder
from torchvision import transforms
from tqdm.auto import tqdm


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    # Image preprocessing, normalization for the pretrained resnet
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size), # crop 224x224 from 256x256 image
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        # use saved vocab file from main() of build_vocab.py
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader_train = get_loader(args.image_dir_train, args.caption_path_train, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 
    data_loader_val = get_loader(args.image_dir_val, args.caption_path_val, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models
    encoder = Encoder(args.embed_size).to(device)
    decoder = Decoder(args.embed_size, len(vocab)).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # Encoder의 CNN 부분은 Pretrained 되어있기 때문에, linear와 batch norm만 학습하면 됨
    params = list(decoder.parameters()) + list(encoder.linear.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    # Show model structure
    print(encoder)
    print(decoder)
    
    # Train the models
    for epoch in tqdm(range(args.num_epochs)):
        encoder.train()
        decoder.train()

        total_step = len(data_loader_train)
        epoch_loss = 0
        for batch_idx, batch in enumerate(tqdm(data_loader_train, total=total_step)):
            
            # Set mini-batch dataset
            images = batch[0].to(device)
            captions = batch[1].to(device)

            decoder.zero_grad()
            encoder.zero_grad()
            
            # Forward, backward and optimize
            features = encoder(images)
            outputs = decoder(features, captions)
            loss = criterion(outputs, captions)
            
            loss.backward()
            optimizer.step()

            epoch_loss = epoch_loss + loss.item()
                
            # Save the model checkpoints
            if (batch_idx+1) % args.save_step == 0:
                torch.save(decoder.state_dict(), os.path.join(
                    args.model_path, 'decoder-{}-{}.ckpt'.format(epoch+1, batch_idx+1)))
                torch.save(encoder.state_dict(), os.path.join(
                    args.model_path, 'encoder-{}-{}.ckpt'.format(epoch+1, batch_idx+1)))

            # Validation after each epoch
            encoder.eval()
            decoder.eval()

            total_step = len(data_loader_train)
            epoch_loss = 0
            for batch_idx, batch in enumerate(tqdm(data_loader_train, total=total_step)):
                images = batch[0].to(device)
                captions = batch[1].to(device)

                with torch.no_grad():
                    features = encoder(images)
                    predictions = decoder(features, captions)
                
                loss = criterion(predictions, captions)

        print('Epoch [{}/{}] Finished, Average Loss: {:.4f}'
                      .format(epoch, args.num_epochs, epoch_loss/total_step))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/' , help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 , help='size for randomly cropping images')
    parser.add_argument('--vocab_path', type=str, default='dataset/vocab_train2017.pkl', help='path for vocabulary wrapper')
    parser.add_argument('--image_dir_train', type=str, default='dataset/resized_train2017', help='directory for resized train images')
    parser.add_argument('--image_dir_val', type=str, default='dataset/resized_val2017', help='directory for resized validation images')
    parser.add_argument('--caption_path_train', type=str, default='dataset/annotations/captions_train2017.json', help='path for train annotation json file')
    parser.add_argument('--caption_path_val', type=str, default='dataset/annotations/captions_val2017.json', help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=1000, help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=1000, help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=512, help='dimension of word embedding vectors')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)
