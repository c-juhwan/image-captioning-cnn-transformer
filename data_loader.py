import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from pycocotools.coco import COCO
from torchtext.data import get_tokenizer


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocabulary, transform=None, max_len=100):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocabulary: torchtext.vocab.Vocab
            transform: image transformer.
        """
        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocabulary = vocabulary
        self.transform = transform
        self.max_len = max_len - 2 # for <start> and <end>

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        vocabulary = self.vocabulary
        ann_id = self.ids[index]
        caption = coco.anns[ann_id]['caption']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        tokenizer = get_tokenizer("basic_english")

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        if len(caption) > self.max_len:
            caption = caption[:self.max_len]

        # Convert caption (string) to word ids.
        tokens = tokenizer(str(caption).lower())
        caption = []
        caption.append(vocabulary['<start>'])
        caption.extend([vocabulary[token] for token in tokens])
        caption.append(vocabulary['<end>'])
        tokens.append('<start>')

        target = torch.Tensor(caption)
        return image, target
    
    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data) # make iterable tuple

    # torch.stack(): Concatenates a sequence of tensors along a new dimension.
    # https://pytorch.org/docs/stable/generated/torch.stack.html
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    max_len = 100

    lengths = [len(cap) for cap in captions]
    
    targets = torch.zeros(len(captions), max_len).long() # (number of captions, maximum size of caption)
    for i, cap in enumerate(captions):
        end = lengths[i] # length of each caption
        targets[i, :end] = cap[:end] # Padding

    return images, targets

def get_loader(root, json, vocabulary, transform, batch_size, shuffle, num_workers, max_len):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root, json=json, vocabulary=vocabulary, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, batch_size=batch_size, 
                                              shuffle=shuffle, num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader
