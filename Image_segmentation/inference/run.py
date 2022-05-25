import path
from tensorflow import keras
import visualize
from keras.models import load_model
from keras.callbacks import CSVLogger
import tensorflow as tf
global argss
import keras.backend as K

def start(args):
  seqs = path.getinfo(args)
  keras.backend.clear_session()

  if args.mode=='only_vis':
    mymodel = load_model(args.model_dir)
    visualize.start(mymodel,seqs,args.model_dir,args)
