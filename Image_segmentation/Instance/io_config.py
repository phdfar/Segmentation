from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2

def run(myself,path):
  dim = (myself.img_size[1],myself.img_size[0])
  frameindex= list(path.keys())[0]
  imagepath = path[frameindex][0]
  seq = path[frameindex][1]
  flagmulti = path[frameindex][2]


  img = load_img(myself.baseinput+'train/'+imagepath, target_size=myself.img_size)
  x = np.asarray(img)
    
  masks,mask_total = seq.load_masks_for_instance([frameindex]);
  y=[]
  for mask in masks:
      temp = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
      y.append(np.expand_dims(temp, 2))
      
  if myself.num_instance!=len(masks):
      y.append(np.expand_dims(np.zeros(myself.img_size, np.uint8), 2))
  mask_total = cv2.resize(mask_total, dim, interpolation = cv2.INTER_NEAREST)
  y.append(np.expand_dims(mask_total, 2))
      
  return x,y
