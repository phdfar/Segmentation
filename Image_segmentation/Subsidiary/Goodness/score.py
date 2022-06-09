
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np

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
    temp[temp==c]==5; temp[temp!=c]==0; temp=temp//5;
    allversion.append(temp)
  return allversion

def optical_flow(args,seqs):
  for seq in seqs:
    seq_path = seq.image_paths
    inputs=[];imagepath=[]
    for frameindex,frame in enumerate(seq_path):
        frame = frame.replace('.jpg','.png')
        frame = frame.replace('JPEGImages','')
        print(frame)
        opt = cv2.imread(args.basepath+args.score_path+frame)
        gtn = seq.load_multi_masks([frameindex]);
        allversion = cluster(opt)
        score_i=[]
        for mask in allversion:
          score_i.append(metric(gtn,mask))
        max(score_i)

