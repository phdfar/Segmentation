import os
from IPython.display import clear_output

os.system('pip install -U --no-cache-dir gdown --pre')
os.system('gdown --no-cookies 1j7ua5QKVNc3QSBSG6XSXiRAyBlRWz-2b')
os.system('unzip -qq train.zip')
os.system('rm train.zip')
os.system('wget https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/youtube_vis_train.json')

clear_output()


