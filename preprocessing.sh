python build_vocab.py --caption_path="dataset/annotations/captions_train2017.json" --vocab_path="./dataset/vocab_train2017.pkl"
python build_vocab.py --caption_path="dataset/annotations/captions_val2017.json" --vocab_path="./dataset/vocab_val2017.pkl"

python resize_image.py --image_dir="./dataset/train2017/" --output_dir="./dataset/resized_train2017/"
python resize_image.py --image_dir="./dataset/test2017/" --output_dir="./dataset/resized_test2017/"
python resize_image.py --image_dir="./dataset/val2017/" --output_dir="./dataset/resized_val2017/"

rm ./dataset/train2017/
rm ./dataset/test017/
rm ./dataset/val2017/