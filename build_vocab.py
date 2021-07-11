import pickle
import argparse
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict
from pycocotools.coco import COCO
from tqdm.auto import tqdm

def load_caption(json_path:str):
    """
    Load caption from json file

    Args:
        json_path (str): path to json file
    Returns:
        OrderedDict
    """
    coco = COCO(json_path)
    counter = Counter()
    ids = coco.anns.keys()
    tokenizer = get_tokenizer("basic_english")

    for i, id in enumerate(tqdm(ids, total=len(ids))):
        caption = str(coco.anns[id]['caption'])
        tokens = tokenizer(caption.lower())
        counter.update(tokens)

    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    return ordered_dict

def build_vocab(ordered_dict:OrderedDict, threshold:int):
    """
    Build torchtext torchtext.vocab.Vocab by torchtext.vocab.vocab()
    Insert special tokens - <pad>, <start>, <end>, <unk>
    
    Args:
        ordered_dict (OrderedDict): result of load_caption() function
        threshold (int): minumum frequency of word
    Returns:
        torchtext.vocab.Vocab
    """
    vocabulary = vocab(ordered_dict, min_freq=threshold)

    vocabulary.insert_token('<pad>', 0)
    vocabulary.insert_token('<start>', 1)
    vocabulary.insert_token('<end>', 2)
    vocabulary.insert_token('<unk>', 3)

    vocabulary.set_default_index(vocabulary['<unk>'])

    return vocabulary

def main(args):
    """
    Build torchtext Vocab object and save it to pickle file
    
    Args:
        args.caption_path (str): path for annotation file
        args.vocab_path (str): path for saving vocabulary data
        args.threshold (int): minimum word count threshold
    """
    json_path = args.caption_path
    vocab_path = args.vocab_path
    threshold = args.threshold

    ordered_dict = load_caption(json_path)
    vocabulary = build_vocab(ordered_dict, threshold)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocabulary, f)

    print("Total vocabulary size: {}".format(len(vocabulary)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='dataset/annotations/captions_train2017.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./dataset/vocab.pkl', 
                        help='path for saving vocabulary data')
    parser.add_argument('--threshold', type=int, default=3, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)