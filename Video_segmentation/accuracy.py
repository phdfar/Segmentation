from tensorflow.keras.preprocessing.image import load_img
import pandas as pd
import data
import cv2
import numpy as np
import os
from tensorflow import keras
import path

def start(mymodel,allframe_test,name,args):
    x = 1200;full_result=[];tac=0;tpr=0;tre=0;tfs=0;lendata=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      tacx,tprx,trex,tfsx,full_result,lendata = run_clip(test_preds_batch,batch_test,name,args,full_result,lendata)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
      
    #lendata=len(allframe_test)
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
        
def run_clip(test_preds,allpath,name,args,full_result,lendata):
  flag_multi=0;
  if  args.num_instance>1:
        flag_multi=1;
    
  Taccuracy=0
  Tprecision=0
  Trecall=0
  TFS=0;q=0;
  for ii in range(len(test_preds)):
    path = allpath[ii]
    clipindex= list(path.keys())[0]
    frameindex = path[clipindex][0]
    seq = path[clipindex][1]
    flagmulti = path[clipindex][2]
    """    
    frameindex= list(path.keys())[0]
    imagepath = path[frameindex][0]
    #tep.append(imagepath)
    seq = path[frameindex][1]
    flagmulti = path[frameindex][2]
    """
    s=0;
    for f in frameindex:
        if flagmulti==0:
          mask = seq.load_one_masks([f])
        else:
          mask = seq.load_multi_masks([f]);
            
        # resize image
        dim = (args.imagesize[1],args.imagesize[0])
        gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
        frame=test_preds[ii][s];s+=1;q+=1;lendata+=1
        mask = np.argmax(frame, axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        mask = mask[:,:,0];

        result = np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
        temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
        TP=0;FP=0;FN=0;TN=0;

        fast_res=mask-gtn
        tpc = np.where(fast_res==0);temp[tpc]=1;
        tp = np.where(temp+mask==2)
        TP = len(tp[0])
        result[tp]=(0,255,0)

        tn = np.where(temp+mask==1)
        TN = len(tn[0])

        temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')

        fp = np.where(fast_res==1);
        FP = len(fp[0])
        result[fp]=(255,0,0)

        fn  = np.where(fast_res==-1)
        FN =  len(fn[0])
        result[fn]=(255,255,0)
   
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
        
  lendatax=q;#len(test_preds)
  tac = Taccuracy/lendatax
  tpr = Tprecision/lendatax
  tre = Trecall/lendatax
  tfs = TFS/lendatax
  
  print("accuracy",tac)
  print("precision",tpr)
  print("recall",tre)
  print("FS",tfs)
  print('---------')
  return Taccuracy,Tprecision,Trecall,TFS,full_result,lendata


def run_image(test_preds,allpath,name,args,full_result):
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
            
    # resize image
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            

    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];

    result = np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    TP=0;FP=0;FN=0;TN=0;
    
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask==2)
    TP = len(tp[0])
    result[tp]=(0,255,0)
    
    #import matplotlib.pyplot as plt
    #plt.figure()
    #plt.imshow(result)
    #break
    tn = np.where(temp+mask==1)
    TN = len(tn[0])
    
    
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')

    fp = np.where(fast_res==1);
    FP = len(fp[0])
    result[fp]=(255,0,0)
    
    fn  = np.where(fast_res==-1)
    FN =  len(fn[0])
    result[fn]=(255,255,0)
   
    
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
