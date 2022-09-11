import os
from IPython.display import clear_output
from argparse import ArgumentParser

def init(args):
  if args.mode=='google':
    os.system('pip install -U --no-cache-dir gdown --pre')
    os.system('gdown --no-cookies 1o586Wjya-f2ohxYf9C1RlRH-gkrzGS8t')
    os.system('unzip -qq valid.zip')
    os.system('rm valid.zip')
  if args.mode=='dropbox':
    os.system('pip install -U --no-cache-dir gdown --pre')
    os.system('wget https://www.dropbox.com/s/y7ebpe7nkrdss8o/valid.zip')
    os.system('unzip -qq valid.zip')
    os.system('rm valid.zip')

    
  #os.system('wget https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/youtube_vis_train.json')
  os.system('wget https://omnomnom.vision.rwth-aachen.de/data/STEm-Seg/dataset_jsons/youtube_vis_val.json')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='google', required=False)
    args = parser.parse_args()
    init(args)


clear_output()


