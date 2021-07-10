# Install pip packages
pip install -r requirements.txt

# Install COCO API
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI/
make
python setup.py build
python setup.py install

cd ../../

clear
echo "Downloaded required packages"