sudo apt-get install python3.7
sudo apt-get install python3.7-distutils
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
sudo update-alternatives --config python3
curl https://bootstrap.pypa.io/pip/3.7/get-pip.py -o get-pip.py
python3.7 get-pip.py
