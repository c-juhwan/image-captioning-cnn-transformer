import torch
import os
from PIL import Image
from pycocotools.coco import COCO
from torchtext.data import get_tokenizer

class COCODataset(torch.utils.data.Dataset):
    """
    Custom dataset class with COCO

    Inherit:
        torch.utils.data.Dataset
    """
    def __init__(self, root:str, json_path:str, vocabulary, transform=None, max_len:int=300):
        """
        Set image and caption file path, load vocabulary built from build_vocab.py

        Args:
            root (str): path for image files
            json_path (str): path for COCO annotations json file
            vocabulary (torchtext.vocab.Vocab): vocabulary built from build_vocab.py
            max_len (int): maximum allowed length of caption
        """
        
        self.root = root
        self.coco = COCO(json_path)
        self.ids = list(self.coco.anns.keys())
        self.vocabulary = vocabulary
        self.max_len = max_len - 2 # For <start> and <end>
        self.tokenizer = get_tokenizer("basic_english")
        self.transform = transform
        self.max_len = max_len - 2 # for <start> and <end>

        global maximum_length
        maximum_length = max_len

    def __getitem__(self, index):
        """
        Returns one data pair tuple (image, caption)

        Return:
            image: PIL Image
            caption (torch.Tensor) 
        """
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']
        path = self.coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if len(caption) > self.max_len:
            caption = caption[:self.max_len]
        
        tokens = self.tokenizer(str(caption).lower()) # list of tokenized words

        caption = [] # int list -> words index
        caption.append(self.vocabulary['<start>'])
        caption.extend([self.vocabulary[token] for token in tokens])
        caption.append(self.vocabulary['<end>'])

        caption = torch.Tensor(caption)
        return img, caption

    def __len__(self):
        """
        Return:
            int: amount of total caption
        """
        return len(self.ids)


def collate_fn(data):
    """
    Make mini-batch of images and captions
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, H, W).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, H, W).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    global maximum_length # Get max_len of COCODataset

    # Descending order sort
    data.sort(key=lambda x: len(x[1]), reverse=True)
    # Get each tuple of images and tuple of captions from list of tuple
    images, captions = zip(*data)
    images = torch.stack(images, 0)

    caption_lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), maximum_length).long()
    for i, cap in enumerate(captions):
        length = caption_lengths[i] # Length of each captions
        targets[i, :length] = cap[:length] # Apply Padding

    return images, targets

def get_loader(root:str, json_path:str, vocabulary, transform, batch_size:int, shuffle:bool, num_workers:int, max_len:int):
    """
    Args:
        root (str): path for image files
        json_path (str): path for COCO annotations json file
        vocabulary (torchtext.vocab.Vocab): vocabulary built from build_vocab.py
        batch_size (int)
        shuffle (bool)
        num_workers (int)
        max_len (int): maximum allowed length of caption

    Return:
        data_loader: torch.utils.data.DataLoader 
                     with custom dataset and custom collate_fn
    """

    dataset = COCODataset(root=root, json_path=json_path, vocabulary=vocabulary, transform=transform, max_len=max_len)

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                              shuffle=shuffle, num_workers=num_workers,
                                              collate_fn=collate_fn)

    return data_loader


