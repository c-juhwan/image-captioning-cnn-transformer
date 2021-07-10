python build_vocab.py

python resize_image.py --image_dir="./dataset/train2017/" --output_dir="./dataset/resized_train2017/"
python resize_image.py --image_dir="./dataset/test2017/" --output_dir="./dataset/resized_test2017/"
python resize_image.py --image_dir="./dataset/val2017/" --output_dir="./dataset/resized_val2017/"

rm -rd ./dataset/train2017/
rm -rd ./dataset/test2017/
rm -rd ./dataset/val2017/