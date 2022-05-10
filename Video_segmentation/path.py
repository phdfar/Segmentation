import data
import pickle
import random
import cv2
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
import itertools


def create_frame_index(subseq_length,start,finish):
    niter =  subseq_length*15;
    subseq_span_range = [3];subsequence_idxes=[]
    clip_length = finish-start
    for _ in range(niter):
        subseq_span = min(random.choice(subseq_span_range), clip_length - 1)
        max_start_idx = clip_length - subseq_span - 1
        assert max_start_idx >= 0

        start_idx = 0 if max_start_idx == 0 else random.randint(0, max_start_idx)
        end_idx = start_idx + subseq_span
        sample_idxes =  np.round(np.linspace(start+start_idx, start + end_idx, subseq_length)).astype(np.int32).tolist()

        assert len(set(sample_idxes)) == len(sample_idxes)  # sanity check: ascertain no duplicate indices
        subsequence_idxes.append(sample_idxes)
    
    subsequence_idxes.sort()
    subsequence_idxes=list(k for k,_ in itertools.groupby(subsequence_idxes))
    #print('subsequence_idxes',subsequence_idxes)
    return subsequence_idxes
  
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
      if i['clip_length']>=20:
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
      
      #print('clip length :',seq.length)

      train_a = 0; train_b=seq.length-(args.subseq_length*2);
      val_a = train_b; val_b=val_a+args.subseq_length
      test_a = val_b; test_b = test_a + args.subseq_length
      
      train_index=create_frame_index(args.subseq_length,0,train_b)
      val_index=create_frame_index(args.subseq_length,val_a,val_b)
      test_index=create_frame_index(args.subseq_length,test_a,test_b)

      p=0;
      for t in train_index:
        p+=1
        allframe_train.append({p:[t,seq,flag_multi]})
      p=0;
      for t in val_index:
        p+=1
        allframe_val.append({p:[t,seq,flag_multi]})
      p=0;
      for t in test_index:
        p+=1
        allframe_test.append({p:[t,seq,flag_multi]})

      """
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
      """
      
  #print(asd)
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
        self.subseq_length = args.subseq_length

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        dim = (self.img_size[1],self.img_size[0])
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (self.subseq_length*3,1), dtype="float32")
        #y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y1 = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y2 = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y3 = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        y4 = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")

        for j, path in enumerate(batch_input_img_paths):
            clipindex= list(path.keys())[0]
            frameindex = path[clipindex][0]
            seq = path[clipindex][1]
            flagmulti = path[clipindex][2]
            temp=[];temp_mask=[]
            for f in frameindex:
              img  = load_img(self.basepath+'train/'+seq.image_paths[f], target_size=self.img_size)
              temp.append(np.asarray(img))
              if flagmulti==0:
                mask = seq.load_one_masks([f])
              else:
                mask = seq.load_multi_masks([f]);
              mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST);mask = np.expand_dims(mask, 2)
              temp_mask.append(mask);
            temp = np.concatenate((tuple(temp)),axis=-1);
            x[j] = np.expand_dims(temp,3)
            y1[j] = temp_mask[0]
            y2[j] = temp_mask[1]
            y3[j] = temp_mask[2]
            y4[j] = temp_mask[3]
                

        """
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        """
        return x, [y1,y2,y3,y4]
