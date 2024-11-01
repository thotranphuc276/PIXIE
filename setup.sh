#!/bin/bash
sudo apt-get install python3.7
sudo apt-get install python3.7-distutils
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo update-alternatives --config python3
curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
python3.7 get-pip.py
apt-get update
apt-get install -y gcc-7 g++-7
apt-get install -y python3.7-dev

apt-get update
apt-get install -y libgl1-mesa-glx libglib2.0-0

sudo apt-get update
sudo apt-get install unzip

# Set gcc-7 as the default compiler
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
update-alternatives --config gcc

sed -i 's/yaml.load(f)/yaml.load(f, Loader=yaml.SafeLoader)/' ./pixielib/models/lbs.py

pip install -r requirements.txt

wget -P ./data https://huggingface.co/camenduru/TalkingHead/resolve/main/FLAME_albedo_from_BFM.npz

gdown 1fT_JAJ5-Jg6hNgSx281MD-HAEs9i_QRu

gdown 1zcR-rLhNQ7QsW6-QZn9LZeLpjl8CrKHz

unzip -o -q image_cache.zip

# Install pytorch cuda toolkit
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html


