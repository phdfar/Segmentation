from pandas._libs.lib import dicts_to_array
from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
import data
import cv2
import numpy as np
import os
from tensorflow import keras
import path
from sklearn import metrics
from keras import backend as K
import tensorflow as tf
import csv

def start(mymodel,allframe_test,name,args,dicid):
    if args.task == 'binary_seg':
        start_binary(mymodel,allframe_test,name,args,dicid)
    if args.task == 'semantic_seg':
        start_semantic(mymodel,allframe_test,name,args,dicid)
        
def start_semantic(mymodel,allframe_test,name,args,dicid):
    x = 300;full_result=[];y_pred=[];y_true=[];IOU=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0,0,0,0]})
    for x in args.classid:
      category_score.update({x:[0,0,0,0,0]})
    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      IOU,category_score,tacx,tprx,trex,tfsx = run_semantic(test_preds_batch,batch_test,name,args,y_pred,y_true,dicid,IOU,category_score)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
        
    print('****')
    print(np.mean(IOU))
    with open('category_score'+name+'.csv', 'w') as f:
        for key in category_score.keys():
            dr = category_score[key][1];
            if dr==0:
              dr=1;
            f.write("%s,%s\n"%(key,category_score[key][0]/dr))
            
    lendata=len(allframe_test)
    tac = tac/lendata
    tpr = tpr/lendata
    tre = tre/lendata
    tfs = tfs/lendata
    print('*********')
    print("accuracy",tac)
    print("precision",tpr)
    print("recall",tre)
    print("FS",tfs)
    print('*********')
    

def start_binary(mymodel,allframe_test,name,args,dicid):
    x = 1200;full_result=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0]})
    for x in args.classid:
      category_score.update({x:[0,0]})
    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      tacx,tprx,trex,tfsx,full_result,category_score = run_binary(test_preds_batch,batch_test,name,args,full_result,category_score,dicid)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
     
    with open('category_score'+name+'.csv', 'w') as f:
        for key in category_score.keys():
            dr = category_score[key][1];
            if dr==0:
              dr=1;
            f.write("%s,%s\n"%(key,category_score[key][0]/dr))

    lendata=len(allframe_test)
    tac = tac/lendata
    tpr = tpr/lendata
    tre = tre/lendata
    tfs = tfs/lendata
    print('*********')
    print("accuracy",tac)
    print("precision",tpr)
    print("recall",tre)
    print("FS",tfs)
    print('*********')
  
    TT = [] ; TT.append(('Total',tac,tpr,tre,tfs))
    full_result = TT + full_result
    df = pd.DataFrame(full_result,columns =['Names','accuracy','precision','recall','FS'])
    df.to_csv('result_'+name+'.csv')
        
def run_binary(test_preds,allpath,name,args,full_result,category_score,dicid):
  flag_multi=0;
  if  args.num_instance>1:
        flag_multi=1;
    
  Taccuracy=0
  Tprecision=0
  Trecall=0
  TFS=0
  for ii in range(len(test_preds)):
    path = allpath[ii]
    frameindex= list(path.keys())[0]
    imagepath = path[frameindex][0]
    #tep.append(imagepath)
    seq = path[frameindex][1]
    flagmulti = path[frameindex][2]
    if flagmulti==0:
      mask = seq.load_one_masks([frameindex])
    else:
      mask = seq.load_multi_masks([frameindex]);
            
    cat = seq.load_class([frameindex],dicid);
    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
  
    # resize image
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            

    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];
    rgb = np.asarray(rgb)
    result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    TP=0;FP=0;FN=0;TN=0;
    
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask==2)
    TP = len(tp[0])
    result[tp]=((0,255,0)+result[tp])//2
    
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(result)
    #break
    tn = np.where(temp+mask==1)
    TN = len(tn[0])
    
    
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')

    fp = np.where(fast_res==1);
    FP = len(fp[0])
    result[fp]=((255,0,0)+ result[fp])//2
    
    fn  = np.where(fast_res==-1)
    FN =  len(fn[0])
    result[fn]=((255,255,0)+result[fn])//2

    #result = ((rgb+result)//2)
    """
    for i in range(mask.shape[0]):
      for j in range(mask.shape[1]):
        if mask[i][j] == gtn[i][j] and mask[i][j]==1:
          result[i][j][0]=0;result[i][j][1]=255;result[i][j][2]=0;
          TP+=1;
        elif mask[i][j] != gtn[i][j] and mask[i][j]==1:
          result[i][j][0]=255;result[i][j][1]=0;result[i][j][2]=0;
          FP+=1;
        elif mask[i][j] != gtn[i][j] and mask[i][j]==0:
          result[i][j][0]=255;result[i][j][1]=255;result[i][j][2]=0;
          FN+=1;
        else:
          TN+=1;
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

    for c in cat:
      newFS = category_score[c][0] + FS
      newC = category_score[c][1] + 1
      category_score.update({c:[newFS,newC]})


    Taccuracy+=accuracy
    Tprecision+=precision
    Trecall+=recall
    TFS+=FS

    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    full_result.append((filename,accuracy,precision,recall,FS))
    try:
      os.mkdir('result')
    except:
      pass
    res.save('result/'+filename)

  lendata=len(test_preds)
  tac = Taccuracy/lendata
  tpr = Tprecision/lendata
  tre = Trecall/lendata
  tfs = TFS/lendata
  
  print("accuracy",tac)
  print("precision",tpr)
  print("recall",tre)
  print("FS",tfs)
  print('---------')
  return Taccuracy,Tprecision,Trecall,TFS,full_result,category_score


  
  
def run_semantic(test_preds,allpath,name,args,y_pred,y_true,dicid,IOU,category_score):
  flag_multi=0;
  if args.num_instance>1:
        flag_multi=1;
    
  Taccuracy=0
  Tprecision=0
  Trecall=0
  TFS=0;
  for ii in range(len(test_preds)):
    path = allpath[ii]
    frameindex= list(path.keys())[0]
    imagepath = path[frameindex][0]
    #tep.append(imagepath)
    seq = path[frameindex][1]
    flagmulti = path[frameindex][2]
    if flagmulti==0:
        if args.task == 'semantic_seg':  
            mask = seq.load_one_masks_semantic([frameindex],dicid)
        else:
            mask = seq.load_one_masks([frameindex])
    else:
        if args.task == 'semantic_seg':  
            mask = seq.load_multi_masks_semantic([frameindex],dicid)
        else:
            mask = seq.load_multi_masks([frameindex]);
            
    # resize image
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
    cat = seq.load_class([frameindex],dicid);

            

    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];

    y_true = gtn;y_pred=mask;smooth=1e-7
    y_pred = y_pred.ravel().tolist()
    y_true = y_true.ravel().tolist()
    m = tf.keras.metrics.MeanIoU(num_classes=41)
    m.update_state(y_pred, y_true)
    iou=m.result().numpy()
    IOU.append(iou)
    for c in cat:
      newiou = category_score[c][0] + iou
      newC = category_score[c][1] + 1
      category_score.update({c:[newFS,newC]})
    
    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
    rgb = np.asarray(rgb)
    result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask>=2)
    TP = len(tp[0])
    result[tp]=((0,255,0)+result[tp])//2
    Pmask = np.where(mask>0);
    Pgtn = np.where(gtn>0);
    
    precision = TP / len(Pmask[0])
    recall = TP / len(Pgtn[0])
    FS = (2*recall*precision)/(precision+recall)
    accuracy = len(tpc[0])/(args.imagesize[0]*args.imagesize[1])
    Taccuracy+=accuracy
    Tprecision+=precision
    Trecall+=recall
    TFS+=FS
    
    
    tn = np.where(temp+mask==1)
    TN = len(tn[0])
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    
    fp = np.where(fast_res==1);
    FP = len(fp[0])
    result[fp]=((255,0,0)+ result[fp])//2
    
    fn  = np.where(fast_res==-1)
    FN =  len(fn[0])
    result[fn]=((255,255,0)+result[fn])//2
    
  lendata=len(test_preds)
  tac = Taccuracy/lendata
  tpr = Tprecision/lendata
  tre = Trecall/lendata
  tfs = TFS/lendata
  print('---------')
  print("IOU",mean(IOU))
  print("accuracy",tac)
  print("precision",tpr)
  print("recall",tre)
  print("FS",tfs)
  print('---------')
  """
    #import pickle
    #with open('loader.pickle', 'wb') as handle:
      #pickle.dump([y_true,y_pred], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    y_true_f = K.flatten(K.one_hot(K.cast(y_true,tf.int32), num_classes=40)[...,1:])
    y_pred_f = K.flatten(K.one_hot(K.cast(y_pred,tf.int32), num_classes=40)[...,1:])
    intersect = K.sum(y_true_f.numpy()* y_pred_f.numpy(), axis=-1)
    denom = K.sum(y_true_f.numpy() + y_pred_f.numpy(), axis=-1)

    IOU.append(K.mean((2. * intersect / (denom + smooth))).numpy());
    
    #y_pred += mask.ravel().tolist()
    #y_true += gtn.ravel().tolist()
    
  #print(metrics.confusion_matrix(y_true, y_pred))
  #print(metrics.classification_report(y_true, y_pred, digits=args.num_class))
  """

  return IOU,category_score,Taccuracy,Tprecision,Trecall,TFS
