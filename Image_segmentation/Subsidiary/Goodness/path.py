import data
import pickle
import random
import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import io_config

jsonf = {'train':'youtube_vis_train.json','valid':'youtube_vis_val.json'}
def getinfo(args):
  base_dir=args.jsonpath+args.data+'/'
  dataset_json = args.jsonpath +jsonf[args.data]
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  print('Number clip is '+ str(len(seqs)))
  return seqs


def getinfo_train(args):
  base_dir=args.basepath+'train/'
  dataset_json = args.basepath +'youtube_vis_train.json'
  meta_plus_path = args.basepath+ 'Segmentation/meta_plus_youtube_vis.pickle'
  goodness_path = args.basepath+ 'Segmentation/Image_segmentation/Subsidiary/Goodness/allpath_goodness.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  
 
    
  with open(meta_plus_path, 'rb') as handle:
    meta_plus = pickle.load(handle)
  with open(goodness_path, 'rb') as handle:
    goodness_file = pickle.load(handle)
  good_train = goodness_file[0];good_val = goodness_file[1];good_test = goodness_file[2];
        
  #print(good_train)
  valid=[];

    
  for i in meta_plus:
      valid.append(i['id'])
  
          
  allframe_train=[];allframe_val=[];allframe_test=[]
  for seq in seqs:
    if seq.id in valid:
      #a = int(np.floor(seq.length*0.67))
      #b = a + int(np.ceil(seq.length*0.1))
      #p=0; 
      #allindex = list(range(len(seq.image_paths)))
      allpath = seq.image_paths;
      for frame,p in enumerate(allpath):
          q = p.replace('jpg','png')
          sp = q.split('/'); name = sp[-2]+'_'+sp[-1]
          if name in good_train:
              allframe_train.append({frame:[p,seq,1]})
          elif name in good_val:
              allframe_val.append({frame:[p,seq,1]})
          elif name in good_test:
              allframe_test.append({frame:[p,seq,1]})

      #random.Random(1337).shuffle(allindex)
      # for frame in allindex:
      #   if p<=a:
      #     allframe_train.append({frame:[allpath[frame],seq,flag_multi]})
      #   elif p>a and p<=b:
      #     allframe_val.append({frame:[allpath[frame],seq,flag_multi]})
      #   else:
      #     allframe_test.append({frame:[allpath[frame],seq,flag_multi]})
      #   p+=1;

      

  print('Number Train frame : ',len(allframe_train))
  print('Number Val frame : ',len(allframe_val))
  print('Number Test frame : ',len(allframe_test))

  return allframe_train,allframe_val,allframe_test

class dataloader_2i(keras.utils.Sequence):
        
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, args,input_img_paths):
        self.batch_size = args.batchsize
        self.img_size = args.imagesize
        self.input_imagesize = args.input_imagesize
        self.input_img_paths = input_img_paths
        self.basepath = args.basepath
        self.task = args.task
        self.channel_input  = args.channel_input
        self.colorspace = args.colorspace
        self.network = args.network
        self.baseinput=args.baseinput
        self.config=args.config
        self.branch_input = args.branch_input
        self.baseinput2=args.baseinput2

        goodness_score = args.basepath+ 'Segmentation/Image_segmentation/Subsidiary/Goodness/goodness_score.pickle'
        with open(goodness_score, 'rb') as handle:
          goodness_score = pickle.load(handle)
        self.goodness_score = goodness_score

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.input_imagesize + (self.channel_input,), dtype="uint8")
        z = np.zeros((self.batch_size,) + self.input_imagesize + (1,), dtype="float32")
        y = np.zeros((self.batch_size,) + (1,), dtype="float32")
        
        for j, path in enumerate(batch_input_img_paths):
            a,y[j] = io_config.run(self,path)
            x[j]=a[0];
            z[j]=a[1];
                   
        return [x,z],y


