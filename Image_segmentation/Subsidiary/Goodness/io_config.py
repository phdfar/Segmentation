from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from models.Spectral import load
def run(myself,path):
  frameindex= list(path.keys())[0]
  imagepath = path[frameindex][0]
  seq = path[frameindex][1]
  flagmulti = path[frameindex][2]

  if myself.network=='spectral':
      x = load.loadeig(myself,imagepath)
      
  else:
      img = load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size)
      x = np.asarray(img)

  """
  if self.colorspace=='rgb':
    img = load_img(self.basepath+'train/'+imagepath, target_size=self.img_size)
  if self.colorspace=='lab':
    img = load_img(self.basepath+'train/'+imagepath, target_size=self.img_size)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2LAB)
  if self.colorspace=='hsv':
    img = load_img(self.basepath+'train/'+imagepath, target_size=self.img_size)
    img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2HSV)
  """

  """
  if myself.channel_input==4:
    opt = load_img(myself.basepath+'train_rgo/train/'+imagepath, target_size=myself.img_size)
    opt = np.asarray(opt);opt = opt[:,:,2];opt = np.expand_dims(opt, 2)
    x = np.concatenate((np.asarray(img),opt),axis=-1)

  else:
  """
  #x = np.asarray(img)
  #x = img.copy()
  if flagmulti==0:
    if myself.task == 'semantic_seg':
      mask = seq.load_one_masks_semantic([frameindex],myself.dicid)
    else:
      mask = seq.load_one_masks([frameindex])
  else:
    if myself.task == 'semantic_seg':
      mask = seq.load_multi_masks_semantic([frameindex],myself.dicid)
    elif myself.task == 'binary_seg':
      mask = seq.load_multi_masks([frameindex]);
    elif myself.task == 'instance_seg':
      mask = seq.load_multi_masks_instance([frameindex]);

  # resize image
  dim = (myself.img_size[1],myself.img_size[0])
  temp = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
  y= np.expand_dims(temp, 2)

  """
    y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
    for j, path in enumerate(batch_target_img_paths):
        img = load_img(path, target_size=self.img_size, color_mode="grayscale")
        y[j] = np.expand_dims(img, 2)
        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        y[j] -= 1
  """

  return x,y
