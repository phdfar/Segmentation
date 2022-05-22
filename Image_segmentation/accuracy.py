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

def start(mymodel,allframe_test,name,args,dicid):
    if args.task == 'binary_seg':
        start_binary(mymodel,allframe_test,name,args,dicid)
    if args.task == 'semantic_seg':
        start_semantic(mymodel,allframe_test,name,args,dicid)
        
def start_semantic(mymodel,allframe_test,name,args,dicid):
    x = 300;full_result=[];y_pred=[];y_true=[];IOU=[]
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      IOU = run_semantic(test_preds_batch,batch_test,name,args,y_pred,y_true,dicid,IOU)
    print('****')
    print(np.mean(IOU))

def start_binary(mymodel,allframe_test,name,args,dicid):
    x = 1200;full_result=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      tacx,tprx,trex,tfsx,full_result = run_binary(test_preds_batch,batch_test,name,args,full_result)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
      
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
        
def run_binary(test_preds,allpath,name,args,full_result):
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
            

    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
  
    # resize image
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            

    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];

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
  return Taccuracy,Tprecision,Trecall,TFS,full_result


  
  
def run_semantic(test_preds,allpath,name,args,y_pred,y_true,dicid,IOU):
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
            

    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];

    y_true = gtn;y_pred=mask;smooth=1e-7
    y_pred = y_pred.ravel().tolist()
    y_true = y_true.ravel().tolist()
    m = tf.keras.metrics.MeanIoU(num_classes=41)
    m.update_state(y_pred, y_true)
    IOU.append(m.result().numpy())

    #import pickle
    #with open('loader.pickle', 'wb') as handle:
      #pickle.dump([y_true,y_pred], handle, protocol=pickle.HIGHEST_PROTOCOL)
    """
    y_true_f = K.flatten(K.one_hot(K.cast(y_true,tf.int32), num_classes=40)[...,1:])
    y_pred_f = K.flatten(K.one_hot(K.cast(y_pred,tf.int32), num_classes=40)[...,1:])
    intersect = K.sum(y_true_f.numpy()* y_pred_f.numpy(), axis=-1)
    denom = K.sum(y_true_f.numpy() + y_pred_f.numpy(), axis=-1)

    IOU.append(K.mean((2. * intersect / (denom + smooth))).numpy());
    """
    #y_pred += mask.ravel().tolist()
    #y_true += gtn.ravel().tolist()
    
  #print(metrics.confusion_matrix(y_true, y_pred))
  #print(metrics.classification_report(y_true, y_pred, digits=args.num_class))

  return IOU
