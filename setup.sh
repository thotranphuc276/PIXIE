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

# Set gcc-7 as the default compiler
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 \
                         --slave /usr/bin/g++ g++ /usr/bin/g++-7
update-alternatives --config gcc

# Install pytorch cuda toolkit
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html


