import pickle
import argparse
from torchtext.vocab import vocab
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict
from pycocotools.coco import COCO
from tqdm.auto import tqdm

def load_caption(json_path):
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

def build_vocab(ordered_dict, threshold):
    vocabulary = vocab(ordered_dict, min_freq=threshold)

    vocabulary.insert_token('<pad>', 0)
    vocabulary.insert_token('<start>', 1)
    vocabulary.insert_token('<end>', 2)
    vocabulary.insert_token('<unk>', 3)

    vocabulary.set_default_index(vocabulary['<unk>'])
    print(type(vocabulary))

    return vocabulary

def main(args):
    json_path = args.caption_path
    vocab_path = args.vocab_path
    threshold = args.threshold

    ordered_dict = load_caption(json_path)
    vocabulary = build_vocab(ordered_dict, threshold)

    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_path', type=str, 
                        default='dataset/annotations/captions_train2017.json', 
                        help='path for train annotation file')
    parser.add_argument('--vocab_path', type=str, default='./dataset/vocab.pkl', 
                        help='path for saving vocabulary wrapper')
    parser.add_argument('--threshold', type=int, default=4, 
                        help='minimum word count threshold')
    args = parser.parse_args()
    main(args)