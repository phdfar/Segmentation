import path
from tensorflow import keras
import visualize
from keras.models import load_model
from keras.callbacks import CSVLogger
import tensorflow as tf
global argss
import keras.backend as K
import score
import scoreinfer
import run_train
def start(args):
    
  keras.backend.clear_session()

  if args.mode=='only_vis':
    seqs = path.getinfo(args)
    if 'h5' in args.model_dir:
      mymodel = load_model(args.model_dir)
    else:
      mymodel=''
      visualize.start(mymodel,seqs,args.model_dir,args)
  elif args.mode=='check_score':
    seqs = path.getinfo(args)
    score.run(args,seqs)
  elif args.mode=='infer_score':
    seqs = path.getinfo(args)
    scoreinfer.run(args,seqs)
  elif args.mode=='train' or args.mode=='test':
      run_train.start(args)