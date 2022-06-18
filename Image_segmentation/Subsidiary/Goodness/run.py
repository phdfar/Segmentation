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
import data
import visualize_original
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
  elif args.mode=='valid_original':
      mymodel = load_model(args.model_dir)
      dataset,meta_info,seqs =  data.parse_generic_video_dataset(args.basepath+'valid/', args.basepath +'youtube_vis_val.json')
      print('Number clip is '+ str(len(seqs)))
      visualize_original.start(mymodel,seqs,args.model_dir,args)
      