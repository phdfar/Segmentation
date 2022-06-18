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
import json
import pickle

def start(mymodel,allframe_test,name,args):
    run(mymodel,allframe_test,name,args)

    
def run(mymodel,allframe_test,name,args):
    dispatcher_loader={1:path.dataloader_2i,2:path.dataloader_2i}

    goodness_score = args.basepath+ 'Segmentation/Image_segmentation/Subsidiary/Goodness/goodness_score.pickle'
    with open(goodness_score, 'rb') as handle:
      goodness_score = pickle.load(handle)
      
    x = 1200;full_result=[];
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0]})
    for x in args.classid:
      category_score.update({x:[0,0]})
      
    MAE=0;
    for batch_test in allframe_test_chunk:
      #test_gen_batch = path.dataloader(args,batch_test,dicid) 
      test_gen_batch = dispatcher_loader[args.branch_input](args,batch_test)   
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      full_result,MAE = run_binary(test_preds_batch,batch_test,name,args,full_result,MAE,goodness_score)
      
     
   

    lendata=len(allframe_test)
    MAE = MAE/lendata

    print('*********')
    print("MAE",MAE)
    print('*********')
  
    df = pd.DataFrame(full_result,columns =['Names','MAE'])
    df.to_csv('result_'+name+'.csv')

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
        
def run_binary(test_preds,allpath,name,args,full_result,MAE,goodness_score):
  p=0;tempMAE=0;
  for ii in range(len(test_preds)):
    path = allpath[ii]
    frameindex= list(path.keys())[0]
    imagepath = path[frameindex][0]
    #tep.append(imagepath)
    seq = path[frameindex][1]
    
    
    mask = seq.load_multi_masks([frameindex]);
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
            
    img = np.asarray(load_img(args.baseinput+'train/'+imagepath, target_size=args.imagesize,grayscale=False))
    sp = imagepath.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
    eig = np.load(args.baseinput2+eigpath) #data/VOC2012/eigs/laplacian/
      
    dim = (args.imagesize[1],args.imagesize[0])
    eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
    eig1 = NormalizeData(eig1)
    
    eig1[eig1<=0.15]=0;eig1[eig1>0.15]=1;
    eig1 = np.expand_dims(eig1,2);eig1 = np.concatenate((eig1,eig1,eig1),axis=-1);
    gtn = np.expand_dims(gtn,2);gtn = np.concatenate((gtn,gtn,gtn),axis=-1);
    #result = np.asarray([img,eig1*255,gtn])
    result = np.concatenate((img,eig1*255,gtn),axis=1);

    namey = eigpath.replace('.pth.npy','.png')

    y_true = float(goodness_score[namey])
    y_pred = float(test_preds[ii][0])
    y_pred = round(y_pred, 2)
    MAE = MAE + abs(y_true-y_pred)
    tempMAE = tempMAE + abs(y_true-y_pred)
 
    font = cv2.FONT_HERSHEY_SIMPLEX;
    footer1 = np.zeros((40,result.shape[1],3),'uint8');al=2;
    text = 'y-true ' + str(y_true) + ' y-pred ' + str(y_pred) + ' diff ' + str(y_true-y_pred)
    cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.6, (255,255,255), 1, cv2.LINE_AA);al+=160;
    result = np.concatenate((result,footer1),axis=0);


    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    full_result.append((filename,y_true,y_pred,abs(y_true-y_pred)))
    p+=1;
    try:
      os.mkdir('result')
    except:
      pass
  
    res.save('result/'+filename)
  print('MAE',tempMAE/p)
  return full_result,MAE


