echo "Running installation..."
sudo apt-get update  
sudo apt-get -y install python
sudo apt-get -y install python-tk
sudo apt-get -y --upgrade install python-pip
pip install --upgrade pip 
sudo pip install --no-cache-dir tensorflow-gpu
sudo pip install --user Cython
sudo pip install --user contextlib2
sudo pip install --user matplotlib
sudo pip install --no-cache-dir spaCy
sudo python -m spacy download en_core_web_sm
sudo pip install python-resize-image
sudo pip install --upgrade virtualenv