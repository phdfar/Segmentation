
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import PIL
from tensorflow import keras
import os
def run(args,seqs):
    if args.score == 'optical_flow':
        optical_flow(args,seqs)

        
        
def vis(rgb,opt,mask,gtn,clipname,FS):
  rgb = np.asarray(rgb)
  result = rgb.copy();
  temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'uint8')
  TP=0;FP=0;FN=0;TN=0;
  try:
      fast_res=mask-gtn
  except:
      dim = (gtn.shape[1],gtn.shape[0])
      mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
      opt = cv2.resize(opt, dim, interpolation = cv2.INTER_NEAREST)
      fast_res=mask-gtn

  tpc = np.where(fast_res==0);temp[tpc]=1;
  tp = np.where(temp+mask==2)
  TP = len(tp[0])
  result[tp]=((0,255,0)+result[tp])//2
  tn = np.where(temp+mask==1)
  TN = len(tn[0])
  temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'uint8')
  fp = np.where(fast_res==1);
  FP = len(fp[0])
  result[fp]=((255,0,0)+ result[fp])//2

  fn  = np.where(fast_res==-1)
  FN =  len(fn[0])
  result[fn]=((255,255,0)+result[fn])//2

  result = np.concatenate((rgb,opt,result),axis=1);
    
  font = cv2.FONT_HERSHEY_SIMPLEX;
  footer1 = np.zeros((40,result.shape[1],3),'uint8');al=2;
  text = clipname + ' ' + str(FS)[:4]+'%'
  cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.4, (255,255,0), 1, cv2.LINE_AA);al+=160;  
  result = np.concatenate((result,footer1),axis=0);
  return result
def metric(gtn,mask):

    #rgb = np.asarray(rgb)
    #result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'uint8')
    TP=0;FP=0;FN=0;TN=0;
    try:
        fast_res=mask-gtn
    except:
        dim = (gtn.shape[1],gtn.shape[0])
        mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
        fast_res=mask-gtn

    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask==2)
    TP = len(tp[0])
    #result[tp]=((0,255,0)+result[tp])//2
    tn = np.where(temp+mask==1)
    TN = len(tn[0])
    temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'uint8')
    fp = np.where(fast_res==1);
    FP = len(fp[0])
    #result[fp]=((255,0,0)+ result[fp])//2
    
    fn  = np.where(fast_res==-1)
    FN =  len(fn[0])
    #result[fn]=((255,255,0)+result[fn])//2
    """
    from time import time
    t1 = time()
    y_true = gtn;y_pred=mask;smooth=1e-7
    y_pred = y_pred.ravel().tolist()
    y_true = y_true.ravel().tolist()
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_pred, y_true)
    IOU=m.result().numpy()
    t2 = time()
    print('ds',t2-t1)
    """
    accuracy = (TP + TN) / (TP + TN + FN + FP)
    try:
      precision = TP / (TP + FP)
    except:
      precision = 0
    try:
      recall = TP / (TP + FN)
    except:
      recall = 0
    try:
      FS = (2*recall*precision)/(precision+recall)
    except:
      FS=0
    return FS,0

def cluster(opt,gtn):
  kmeans = KMeans(n_clusters=4)
  a = opt.shape[0]; b=opt.shape[1]
  feats = opt.reshape(a*b,1)


  kmeans = MiniBatchKMeans(n_clusters=4, batch_size=4096, max_iter=5000, random_state=0)
  clusters = kmeans.fit_predict(feats)
  
  output = clusters.reshape(a,b)
  allversion=[]
  for c in range(0,4):
    temp = output.copy()
    temp[temp!=c]=5; temp[temp==c]=0; temp=temp//5;
    allversion.append(temp)
  score_f=[];

  for mask in allversion:
    fs,iou = metric(gtn,mask)
    score_f.append(fs);
  
  return allversion[np.argmax(score_f)],max(score_f)

def find_good_frame(name,all_name,all_FS):
    fs=[];ns=[]
    for i in range(len(all_name)):
        if name in all_name[i]:
            fs.append(all_FS[i])
            ns.append(all_name[i])
    vs = np.argmax(fs)
    return ns[vs]
  
def optical_flow(args,seqs):
  full_result=[];score_FS_clip={};score_IOU_clip={};
  out_folder=str(args.tr1)+'_'+str(args.tr2)
  try:
    os.mkdir(out_folder)
  except:
    pass
                          
  clip = pd.read_csv(args.clipscore)
  all  = pd.read_csv(args.allscore)
  idx = np.argmax(clip['FS'])
  name = clip['Clip'][idx]
  all_name = list(all['Names'])
  all_FS = list(all['FS'])
  valid=[]
  for cl in range(len(clip)):
    if args.tr1<=clip['FS'][cl]<=args.tr2:
      valid.append(clip['Clip'][cl])
  for i,seq in enumerate(seqs):
    seq_path = seq.image_paths
    inputs=[];imagepath=[];score_temp_i=[];score_temp_f=[]
    sp = seq_path[0].split('/');
    if sp[1] in valid:
        name = find_good_frame(sp[1],all_name,all_FS)
        name=name.replace('_','/');goodframe=name.replace('.png','.jpg')
        for frameindex,frame in enumerate(seq_path):
            if frame=='JPEGImages/'+goodframe:
              rgb = cv2.imread(args.basepath+'train/'+frame)
              frame = frame.replace('.jpg','.png')
              frame = frame.replace('JPEGImages','')
              #print(frame)
              opt = cv2.imread(args.basepath+args.score_path+frame,0)
              gtn = seq.load_multi_masks([frameindex]);
              bestcluster,FS = cluster(opt,gtn)
              opt = cv2.imread(args.basepath+args.score_path+frame)
              res = vis(rgb,opt,bestcluster,gtn,frame,FS)
              sp = frame.split('/'); filename=sp[-2]+'_'+sp[-1]
              res = keras.preprocessing.image.array_to_img(res)
              res.save(out_folder+'/'+filename)

