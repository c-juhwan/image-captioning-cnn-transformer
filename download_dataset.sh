mkdir dataset
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P ./dataset/
wget http://images.cocodataset.org/zips/train2017.zip -P ./dataset/
wget http://images.cocodataset.org/zips/val2017.zip -P ./dataset/
wget http://images.cocodataset.org/zips/test2017.zip -P ./dataset/

unzip ./dataset/train2017.zip -d ./dataset/
rm ./dataset/train2017.zip
unzip ./dataset/val2017.zip -d ./dataset/
rm ./dataset/val2017.zip
unzip ./dataset/test2017.zip -d ./dataset/
rm ./dataset/test2017.zip
unzip ./dataset/annotations_trainval2017.zip -d ./dataset/
rm ./dataset/annotations_trainval2017.zip

clear
echo "Downloaded dataset to ./dataset"