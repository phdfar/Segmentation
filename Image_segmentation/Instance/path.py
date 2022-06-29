import data
import pickle
import random
import cv2
import io_config
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

def getinfo(args):
  base_dir=args.basepath+'train/'
  dataset_json = args.basepath +'youtube_vis_train.json'
  meta_plus_path = args.basepath+ 'Segmentation/meta_plus_youtube_vis.pickle'
  dataset,meta_info,seqs =  data.parse_generic_video_dataset(base_dir, dataset_json)
  
  if args.classid=='all':
    args.classid = list(np.linspace(1,40,40).astype('int32'))
  
  dicid={};
  if args.task=='semantic_seg':
    i=1;
    for x in args.classid:
      dicid.update({x:i});i+=1
  
  flag_multi=0;
  if  args.num_instance>1:
    flag_multi=1;
    
  with open(meta_plus_path, 'rb') as handle:
    meta_plus = pickle.load(handle)
  valid=[];

    
  for i in meta_plus:
    if  i['number_instances']==args.num_instance:
        valid.append(i['id'])
           
    # flag=0;
    # if  args.num_instance==1000 and args.unq_class==1000:
    #   for c in i['unique_class']:
    #     if c not in list(args.classid):
    #       flag=1;
    #   if flag==0:
    #     valid.append(i['id'])
    #     #print(i)
          
    # elif args.num_instance==1000 and args.unq_class!=1000:
    #   if i['number_unique_class']==args.unq_class:
    #     for c in i['unique_class']:
    #       if c not in list(args.classid):
    #         flag=1;  
    #     if flag==0:
    #       valid.append(i['id'])
    #       #print(i)
          
    # elif i['number_instances']==args.num_instance and i['number_unique_class']==args.unq_class:
    #   for c in i['unique_class']:
    #     if c not in list(args.classid):
    #       flag=1;  
    #     if flag==0:
    #       valid.append(i['id'])
    #       #print(i)

    #       #lenf+=i['clip_length']
    #       #print(i)
          
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

    def __init__(self, args,input_img_paths,dicid):
        self.batch_size = args.batchsize
        self.img_size = args.imagesize
        self.input_imagesize = args.input_imagesize
        self.input_img_paths = input_img_paths
        self.basepath = args.basepath
        self.task = args.task
        self.dicid = dicid
        self.channel_input  = args.channel_input
        self.colorspace = args.colorspace
        self.network = args.network
        self.baseinput=args.baseinput
        self.config=args.config
        self.num_instance=args.num_instance

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.input_imagesize + (self.channel_input,), dtype="float32")
        z = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        w = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        t = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            x[j],a = io_config.run(self,path)
            z[j]=a[0];
            w[j]=a[1];
            t[j]=a[2];
        return x, [z,w,t]
