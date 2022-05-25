import data
import pickle
import random
import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def getinfo(args):
  base_dir=args.basepath+'valid/'
  dataset_json = args.basepath +'youtube_vis_val.json'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  print('Number clip is '+ str(len(seqs)))
  return seqs
