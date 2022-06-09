
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2
import numpy as np

def start(args,seqs):
    if args.task == 'optical_flow':
        optical_flow(args,seqs)


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
        opt = cv2.imread(args.basepath+args.data+'/'+frame)
        mask = seq.load_multi_masks([frameindex]);

