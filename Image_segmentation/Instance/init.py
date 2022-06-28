import os
from IPython.display import clear_output
from argparse import ArgumentParser

def init(args):
  if args.mode=='google':
    os.system('pip install -U --no-cache-dir gdown --pre')
    os.system('gdown --no-cookies 1j7ua5QKVNc3QSBSG6XSXiRAyBlRWz-2b')
    os.system('unzip -qq train.zip')
    os.system('rm train.zip')
  if args.mode=='dropbox':
    os.system('pip install -U --no-cache-dir gdown --pre')
    os.system('wget https://www.dropbox.com/s/7uv3w6j5whxw7zs/sda1.backup.tar.gz.aa')
    os.system('wget https://www.dropbox.com/s/94ng0s3k3l35udb/sda1.backup.tar.gz.ab')
    os.system('wget https://www.dropbox.com/s/l0m89q4kvz7usqi/sda1.backup.tar.gz.ac')
    os.system('wget https://www.dropbox.com/s/yhbxvueb9pv12ga/sda1.backup.tar.gz.ad')
    os.system('wget https://www.dropbox.com/s/x2awgr0klyuc029/sda1.backup.tar.gz.ae')
    os.system('cat sda1.backup.tar.gz.* | tar xzf -')
    os.system('rm sda1.backup.tar.gz.aa')
    os.system('rm sda1.backup.tar.gz.ab')
    os.system('rm sda1.backup.tar.gz.ac')
    os.system('rm sda1.backup.tar.gz.ad')
    os.system('rm sda1.backup.tar.gz.ae')
    os.system('tar xfz train.tar.gz')
    os.system('rm train.tar.gz')

    
  os.system('wget https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/youtube_vis_train.json')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='google', required=False)
    args = parser.parse_args()
    init(args)


clear_output()


