import os
from IPython.display import clear_output
from argparse import ArgumentParser

def init(args):
  if 'valid'==args.type:
    if args.mode=='google':
      os.system('pip install -U --no-cache-dir gdown --pre')
      os.system('gdown --no-cookies 1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t')
      os.system('unzip -qq valid.zip')
      os.system('rm valid.zip')
    if args.mode=='dropbox':
      os.system('pip install -U --no-cache-dir gdown --pre')
      os.system('wget https://www.dropbox.com/s/3chm2ns5e03t0rp/valid.zip')
      os.system('unzip -qq valid.zip')
      os.system('rm valid.zip')
    os.system('wget https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/youtube_vis_val.json')

  if 'train'==args.type:
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

  if 'eig_train'==args.type:
    os.system('gdown 1JGJdomjXiTmgGc5dJiv1wZHSYA80xPZ-')
    os.system('gdown 1N_7PV-xzEZYOAD-IxHYCaJk4Q5So2-cA')
    os.system('gdown 1sUYg2bdJ-XcZkjuUOhekqfpYDMGd6YWM')
    os.system('gdown 1-0c0Q7aOZdiG1byYXycAz8iSWK07kvnz')
    os.system('gdown 1yF3z14EAgCFzZRSVUj4ddHJ53tF78MCo')
    os.system('tar xzf eig1.tar.gz')
    os.system('tar xzf eig2.tar.gz')
    os.system('tar xzf eig3.tar.gz')
    os.system('tar xzf eig4.tar.gz')
    os.system('tar xzf eig5.tar.gz')
    os.system('rm eig1.tar.gz')
    os.system('rm eig2.tar.gz')
    os.system('rm eig3.tar.gz')
    os.system('rm eig4.tar.gz')
    os.system('rm eig5.tar.gz')
  if 'eig_valid'==args.type:
      os.system('gdown 11RY6g2dQd1xl3UNnL3f-u008Hc7DlPjP')
      os.system('tar xzf eig_valid.tar.gz')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='google', required=False)
    parser.add_argument('--type', type=str ,default='train', required=False)
    args = parser.parse_args()
    init(args)


clear_output()


