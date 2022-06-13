
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

def run(args,seqs):
    if args.score == 'optical_flow':
        optical_flow(args,seqs)
    if args.score == 'have_mask':
        have_mask(args,seqs)


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

def cluster(opt):
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
  return allversion

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
def have_mask(args,seqs):
    full_result=[];score_FS_clip={};score_IOU_clip={};
    for i,seq in enumerate(seqs):
      seq_path = seq.image_paths
      inputs=[];imagepath=[];score_temp_i=[];score_temp_f=[]

      for frameindex,frame in enumerate(seq_path):

          frame = frame.replace('.jpg','.png')
          frame = frame.replace('JPEGImages','')

          #print(frame)
          try:
              framex = frame.replace('/','_')
              mask = cv2.imread(args.basepath+args.score_path+framex)
          except:
              framex = frame.replace('/','_')
              eigpath = args.basepath+args.score_path+framex[1:]
              eigpath=eigpath.replace('.png','.pth.npy');#eigpath = sp[-2]+'_'+name;
              eig = np.load(eigpath)
              dim = (args.imagesize[1],args.imagesize[0])
              f = NormalizeData(eig[:,:,1])
              f = cv2.resize(f, dim, interpolation = cv2.INTER_NEAREST)
              tr=0.15
              f[f<=tr]=0;f[f>tr]=1;mask = f.copy()
              gtn = seq.load_multi_masks([frameindex]);
              dim = (gtn.shape[1],gtn.shape[0])
              mask = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
              fs,iou = metric(gtn,mask)
              sp = frame.split('/'); filename=sp[-2]+'_'+sp[-1]
              full_result.append((filename,fs))
              score_temp_f.append(fs)

      score_FS_clip.update({sp[-2]:max(score_temp_f)})
        #print(score_FS_clip)
      if i%100==0:
          print(i)
          
    df = pd.DataFrame(full_result,columns =['Names','FS','IOU'])
    df.to_csv('result_'+args.score_path+'.csv')
    with open('result_FS_clip'+args.score_path+'.csv', 'w') as f:
        f.write("%s,%s\n"%('Clip','FS'))
        for key in score_FS_clip.keys():
          f.write("%s,%s\n"%(key,score_FS_clip[key]))
    
def optical_flow(args,seqs):
  full_result=[];score_FS_clip={};score_IOU_clip={};
  for i,seq in enumerate(seqs):
    seq_path = seq.image_paths
    inputs=[];imagepath=[];score_temp_i=[];score_temp_f=[]

   

    for frameindex,frame in enumerate(seq_path):

        frame = frame.replace('.jpg','.png')
        frame = frame.replace('JPEGImages','')
        #print(frame)
        opt = cv2.imread(args.basepath+args.score_path+frame,0)
        gtn = seq.load_multi_masks([frameindex]);
        allversion = cluster(opt)
        score_f=[];score_i=[]

        for mask in allversion:
          fs,iou = metric(gtn,mask)
          score_f.append(fs);score_i.append(iou);

        sp = frame.split('/'); filename=sp[-2]+'_'+sp[-1]
        full_result.append((filename,max(score_f),max(score_i)))
        score_temp_i.append(max(score_i))
        score_temp_f.append(max(score_f))


        #print(score_i)
        #print(max(score_i))
        #import pickle 
        #with open('/content/aa.pickle','wb') as h:
        #  pickle.dump([allversion,gtn],h)

    score_FS_clip.update({sp[-2]:max(score_temp_f)})
    score_IOU_clip.update({sp[-2]:max(score_temp_i)})
    #print(score_FS_clip)
    if i%100==0:
      print(i)


  df = pd.DataFrame(full_result,columns =['Names','FS','IOU'])
  df.to_csv('result_'+args.score_path+'.csv')
  with open('result_FS_clip'+args.score_path+'.csv', 'w') as f:
      f.write("%s,%s\n"%('Clip','FS'))
      for key in score_FS_clip.keys():
        f.write("%s,%s\n"%(key,score_FS_clip[key]))
  with open('result_IOU_clip'+args.score_path+'.csv', 'w') as f:
      f.write("%s,%s\n"%('Clip','IOU'))
      for key in score_IOU_clip.keys():
        f.write("%s,%s\n"%(key,score_IOU_clip[key]))
          

