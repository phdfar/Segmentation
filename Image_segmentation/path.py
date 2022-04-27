import data
import pickle
import random

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

  
def getinfo(args):
  base_dir='/content/train/'
  dataset_json = '/content/youtube_vis_train.json'
  meta_plus_path = '/content/stemseg/meta_plus_youtube_vis.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  with open(meta_plus_path, 'rb') as handle:
    meta_plus = pickle.load(handle)
  valid=[];
  lenf=0;
  for i in meta_plus:
    if i['number_instances']==args.num_instance and i['number_unique_class']==args.unq_class:
      for c in i['unique_class']:
        if c in list(args.classid):
          valid.append(i['id'])
          #lenf+=i['clip_length']
          #print(i)
          
  allframe_train=[];allframe_val=[];allframe_test=[]
  for seq in seqs:
    if seq.id in valid:
      a = int(np.floor(seq.length*0.67))
      b = a + int(np.ceil(seq.length*0.1))
      p=0; allpath = seq.image_paths; random.shuffle(allpath)
      for frame in allpath:
        if p<=a:
          allframe_train.append({frame:seq})
        elif p>a and p<=b:
          allframe_val.append({frame:seq})
        else:
          allframe_test.append({frame:seq})
        p+=1;

      

  print('Number Train frame : ',len(allframe_train))
  print('Number Val frame : ',len(allframe_val))
  print('Number Test frame : ',len(allframe_test))

  return allframe_train,allframe_val,allframe_test

class dataloader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, args,input_img_paths):
        self.batch_size = args.batchsize
        self.img_size = args.imagesize
        self.input_img_paths = input_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            imagepath=list(path.keys())[0]
            img = load_img(imagepath, target_size=self.img_size)
            x[j] = np.asarray(img)
            #path[imagepath].
            
        """
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        """
        return x, y
