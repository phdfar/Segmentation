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
def start(args):
  seqs = path.getinfo(args)
  keras.backend.clear_session()

  if args.mode=='only_vis':
    if 'h5' in args.model_dir:
      mymodel = load_model(args.model_dir)
    else:
      mymodel=''
      visualize.start(mymodel,seqs,args.model_dir,args)
  if args.mode=='check_score':
    score.run(args,seqs)
  if args.mode=='infer_score':
    scoreinfer.run(args,seqs)
