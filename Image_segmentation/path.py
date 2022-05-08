import data
import pickle
import random
import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

  
def getinfo(args):
  base_dir=args.basepath+'train/'
  dataset_json = args.basepath +'youtube_vis_train.json'
  meta_plus_path = args.basepath+ 'Segmentation/meta_plus_youtube_vis.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  with open(meta_plus_path, 'rb') as handle:
    meta_plus = pickle.load(handle)
  valid=[];
  lenf=0;
  flag_multi=0;
  if  args.num_instance>1:
    flag_multi=1;
    
  for i in meta_plus:
    flag=0;
    if  args.num_instance==1000 and args.unq_class==1000:
      for c in i['unique_class']:
        if c not in list(args.classid):
          flag=1;
      if flag==0:
        valid.append(i['id'])
        #print(i)
          
    elif args.num_instance==1000 and args.unq_class!=1000:
      if i['number_unique_class']==args.unq_class:
        for c in i['unique_class']:
          if c not in list(args.classid):
            flag=1;  
        if flag==0:
          valid.append(i['id'])
          #print(i)
          
    elif i['number_instances']==args.num_instance and i['number_unique_class']==args.unq_class:
      for c in i['unique_class']:
        if c not in list(args.classid):
          flag=1;  
        if flag==0:
          valid.append(i['id'])
          #print(i)

          #lenf+=i['clip_length']
          #print(i)
          
  allframe_train=[];allframe_val=[];allframe_test=[]
  for seq in seqs:
    if seq.id in valid:
      a = int(np.floor(seq.length*0.67))
      b = a + int(np.ceil(seq.length*0.1))
      p=0; 
      allindex = list(range(len(seq.image_paths)))
      allpath = seq.image_paths;
      random.Random(1337).shuffle(allindex)
      for frame in allindex:
        if p<=a:
          allframe_train.append({frame:[allpath[frame],seq,flag_multi]})
        elif p>a and p<=b:
          allframe_val.append({frame:[allpath[frame],seq,flag_multi]})
        else:
          allframe_test.append({frame:[allpath[frame],seq,flag_multi]})
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
        self.basepath = args.basepath

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            frameindex= list(path.keys())[0]
            imagepath = path[frameindex][0]
            img = load_img(self.basepath+'train/'+imagepath, target_size=self.img_size)
            x[j] = np.asarray(img)
            seq = path[frameindex][1]
            flagmulti = path[frameindex][2]
            if flagmulti==0:
              mask = seq.load_one_masks([frameindex])
            else:
              mask = seq.load_multi_masks([frameindex]);
            # resize image
            dim = (self.img_size[1],self.img_size[0])
            temp = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            y[j] = np.expand_dims(temp, 2)
            
        """
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        """
        return x, y
