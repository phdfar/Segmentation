
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np
import pandas as pd

def run(args,seqs):
    if args.score == 'optical_flow':
        optical_flow(args,seqs)


def metric(gtn,mask):

    #rgb = np.asarray(rgb)
    #result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'uint8')
    TP=0;FP=0;FN=0;TN=0;
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
    return FS

def cluster(opt):
  kmeans = KMeans(n_clusters=4)
  a = opt.shape[0]; b=opt.shape[1]
  feats = opt.reshape(a*b,1)
  clusters = kmeans.fit_predict(feats)
  output = clusters.reshape(a,b)
  cl = list(set(output.ravel().tolist()))
  allversion=[]
  for c in cl:
    temp = output.copy()
    temp[temp!=c]=5; temp[temp==c]=0; temp=temp//5;
    allversion.append(temp)
  return allversion

def optical_flow(args,seqs):
  full_result=[];score_clip={};
  for i,seq in enumerate(seqs):
    seq_path = seq.image_paths
    inputs=[];imagepath=[];score_temp=[]
    for frameindex,frame in enumerate(seq_path):
        frame = frame.replace('.jpg','.png')
        frame = frame.replace('JPEGImages','')
        #print(frame)
        opt = cv2.imread(args.basepath+args.score_path+frame,0)
        gtn = seq.load_multi_masks([frameindex]);
        allversion = cluster(opt)
        score_i=[]
        for mask in allversion:
          score_i.append(metric(gtn,mask))
        sp = frame.split('/'); filename=sp[-2]+'_'+sp[-1]
        full_result.append((filename,max(score_i)))
        score_temp.append(max(score_i))
        #print(score_i)
        #print(max(score_i))
        #import pickle 
        #with open('/content/aa.pickle','wb') as h:
        #  pickle.dump([allversion,gtn],h)
        break;
        
    
    score_clip.update({sp[-2]:max(score_temp)})
    if i==10:
      break
    print(score_clip)
  df = pd.DataFrame(full_result,columns =['Names','FS'])
  df.to_csv(args.basepath+'result_'+args.score_path+'.csv')
  with open(args.basepath+'result_clip'+args.score_path+'.csv', 'w') as f:
      f.write("%s,%s\n"%('Clip','FS'))
      for key in score_clip.keys():
        f.write("%s,%s\n"%(key,score_clip[key]))
          

