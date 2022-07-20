

import json
from pycocotools import mask as masktools
import numpy as np
import cv2

def decode_rle(seg):
    rle_mask = {
        "counts": seg['counts'].encode('utf-8'),
        "size": seg['size']
    }
    mask = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
    return mask

def decode_rle_gt(seg,h,w):
    rle_mask = {
        "counts": seg.encode('utf-8'),
        "size": (h,w)
    }
    mask = np.ascontiguousarray(masktools.decode(rle_mask).astype(np.uint8))
    return mask

def extract_gt(gt_mask,h,w):
  Total_mask=0;
  for g in gt_mask.keys():
    Total_mask=Total_mask+decode_rle_gt(gt_mask[g],h,w)
  
  if np.sum(Total_mask)==0:
    print('gt_mask.keys() --> ',gt_mask.keys())
    Total_mask = np.zeros((h,w), np.uint8)

  Total_mask[Total_mask==2]=1;
  return Total_mask

def binary_seg_totoal(pred,gt,allrgb,path,all_FS,all_PR,all_RE,all_AC):
  keys = gt.keys()
  
  TP=0;FP=0;FN=0;TN=0;
  for k in list(keys):
    mask = pred[k].astype('float32'); gtn = gt[k].astype('float32')
    result = allrgb[k]

    #gtn = cv2.resize(gtn, (1280,720), interpolation = cv2.INTER_NEAREST)
    #gtn = cv2.resize(gtn, (1280,720), interpolation = cv2.INTER_NEAREST)

    temp =  np.zeros((gtn.shape[0],gtn.shape[1]),'float32')
    #print(temp.shape)
    #print(mask.shape)
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask==2)
    TP = len(tp[0])
    result[tp]=((0,255,0)+result[tp])//2   
    tn = np.where(temp+mask==1)
    TN = len(tn[0])
    fp = np.where(fast_res==1);
    FP = len(fp[0])
    result[fp]=((255,0,0)+ result[fp])//2
    fn  = np.where(fast_res==-1)
    FN =  len(fn[0])
    result[fn]=((255,255,0)+result[fn])//2
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
    all_FS.append(FS); all_RE.append(recall); all_PR.append(precision); all_AC.append(accuracy);
    sp = path[k].split('/'); filename = sp[-2]+'_'+sp[-1]
    result = cv2.resize(result, (320,192), interpolation = cv2.INTER_NEAREST)

    footer1 = np.zeros((40,320,3),'uint8');al=2;
    footer = np.zeros((40,320,3),'uint8');al=2;

    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(footer1, filename, (al,footer1.shape[0]-20), font, 0.4, (255,255,255), 1, cv2.LINE_AA);al+=60;  
    text = 'FS ' + str(FS)[:4]+'%' + ' RE ' + str(recall)[:4]+'%' + ' PR ' + str(precision)[:4]+'%' + ' AC ' + str(precision)[:4]+'%' 
    cv2.putText(footer, text, (2,footer.shape[0]-20), font, 0.3, (255,255,255), 1, cv2.LINE_AA);al+=60;
    result = np.concatenate((result,footer1,footer),axis=0); 
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB);
    cv2.imwrite('vis/'+filename,result)
  return all_FS,all_RE,all_PR,all_AC

def run(args):
  with open('youtube_vis_train.json', 'r') as fh:
    dataset = json.load(fh)

  meta_info = dataset["meta"]

  # convert instance and category IDs from str to int
  meta_info["category_labels"] = {int(k): v for k, v in meta_info["category_labels"].items()}
  gt_data={}
  if "segmentations" in dataset["sequences"][0]:
      for seq in dataset["sequences"]:
          seq["categories"] = {int(iid): cat_id for iid, cat_id in seq["categories"].items()}
          seq["segmentations"] = [
              {
                  int(iid): seg
                  for iid, seg in seg_t.items()
              }
              for seg_t in seq["segmentations"]
          ]
          gt_data.update({seq['id']:[seq["categories"],seq["segmentations"],seq['image_paths'],seq['height'],seq['width']]})

  try:
    import os
    os.mkdir('vis')
  except:
    pass

  with open(args.jsonfile, 'r') as fh:
      result = json.load(fh)

  all_video=[]
  for i in range(len(result)):
      instance = result[i]
      all_video.append(instance['video_id'])
          
          
  all_video = list(set(all_video))
  video_dic={}
  for video in all_video:
      video_dic.update({video:[]})
      
  for i in range(len(result)):
      instance = result[i]
      for video in all_video:
          if instance['video_id'] == video and instance['score']>=args.tr:
              value = video_dic[video] + [i]
              video_dic.update({video:value})

  all_FS=[];all_PR=[];all_RE=[];all_AC=[];

  for video in video_dic.keys():
      instances_id = list(set(video_dic[video]))
      Total_mask_pred={};Total_mask_gt={}
      ln = len(result[instances_id[0]]['segmentations'])
      for l in range(0,ln):
        Total_mask_pred.update({l:0})
      allrgb=[]
      for instance in instances_id:
          label = result[instance]['category_id']
          segmentations = result[instance]['segmentations']
          for i,seg in enumerate(segmentations):
              mask_pred = decode_rle(seg)
              Total_mask_gt.update({i:extract_gt(gt_data[video][1][i],gt_data[video][3],gt_data[video][4])})
              path = gt_data[video][2][i]
              rgb = cv2.imread ('/content/train/'+path,cv2.IMREAD_COLOR)
              rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB);allrgb.append(rgb)
              new_mask = Total_mask_pred[i]+mask_pred
              new_mask[new_mask==2]=1;
              Total_mask_pred.update({i:new_mask})
      all_FS,all_PR,all_RE,all_AC = binary_seg_totoal(Total_mask_pred,Total_mask_gt,allrgb,gt_data[video][2],all_FS,all_PR,all_RE,all_AC)
      #all_FS.append(FS); all_RE.append(RE); all_PR.append(PR); all_AC.append(AC);
  
  FS = np.mean(all_FS);RE = np.mean(all_RE);PR=np.mean(all_PR);AC=np.mean(all_AC)
  print(FS,RE,PR,AC)

from argparse import ArgumentParser

def main(args):
  run(args)          

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--jsonfile', type=str ,default='results.json', required=False)
    parser.add_argument('--tr', type=float ,default=0.45, required=False)
    args = parser.parse_args()

    main(args)



