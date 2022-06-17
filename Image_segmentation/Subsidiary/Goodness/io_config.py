from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
#from models.Spectral import load

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def run(myself,path):
  frameindex= list(path.keys())[0]
  imagepath = path[frameindex][0]
  seq = path[frameindex][1]
  flagmulti = path[frameindex][2]
  
  
  img = np.asarray(load_img(myself.baseinput+'train/'+imagepath, target_size=myself.img_size,grayscale=False))
  sp = imagepath.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
  eig = np.load(myself.baseinput2+eigpath) #data/VOC2012/eigs/laplacian/
    
  dim = (myself.img_size[1],myself.img_size[0])
  eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
  eig1 = NormalizeData(eig1)
  
  eig1[eig1<=0.15]=0;eig1[eig1>0.15]=1;
  eig1 = np.expand_dims(eig1,2)

  x = [img,eig1]
  
  namey = eigpath.replace('.pth.npy','.png')

  y = myself.goodness_score[namey]
  y = np.expand_dims(y,0)

  return x,y
