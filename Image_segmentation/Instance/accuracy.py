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
import torch.nn.functional as F
import denseCRF

def start(mymodel,allframe_test,name,args,dicid):
    if args.task == 'binary_seg':
        start_binary(mymodel,allframe_test,name,args,dicid)
    if args.task == 'semantic_seg':
        start_semantic(mymodel,allframe_test,name,args,dicid)
    if args.task == 'instance_seg':
        start_instance(mymodel,allframe_test,name,args,dicid)
        
def start_semantic(mymodel,allframe_test,name,args,dicid):
    x = 75;full_result=[];y_pred=[];y_true=[];IOU=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0,0,0,0]})
    for x in args.classid:
      category_score.update({x:[0,0,0,0,0]})
    
    with open(args.basepath +'youtube_vis_train.json', 'r') as fh:
        dataset = json.load(fh)
    meta_info = dataset["meta"]
    category_label = {int(k): v for k, v in meta_info["category_labels"].items()}
    category_color={};category_color.update({0:(0,0,0)})
    for x in args.classid:
      r = np.random.randint(0,255,1)[0]
      g = np.random.randint(0,150,1)[0]
      b = np.random.randint(0,255,1)[0]
      category_color.update({x:(r,g,b)})

    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      IOU,category_score,tacx,tprx,trex,tfsx = run_semantic(test_preds_batch,batch_test,name,args,y_pred,y_true,dicid,IOU,category_score,category_label,category_color)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
    print('****')
    print(np.mean(IOU))
    with open('category_score'+name+'.csv', 'w') as f:
        f.write("%s,%s,%s,%s,%s\n"%('Class','IOU','Recall','Precision','FS'))
        for key in category_score.keys():
            dr = category_score[key][4];
            if dr==0:
              dr=1;
            f.write("%s,%s,%s,%s,%s\n"%(key,category_score[key][0]/dr,category_score[key][1]/dr,category_score[key][2]/dr,category_score[key][3]/dr))
            
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
    
def start_instance(mymodel,allframe_test,name,args,dicid):
    x = 75;full_result=[];y_pred=[];y_true=[];IOU=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0,0,0,0]})
    for x in args.classid:
      category_score.update({x:[0,0,0,0,0]})
    
    with open(args.basepath +'youtube_vis_train.json', 'r') as fh:
        dataset = json.load(fh)
    meta_info = dataset["meta"]
    category_label = {int(k): v for k, v in meta_info["category_labels"].items()}
    category_color={};category_color.update({0:(0,0,0)})
    cl=[(255,0,0),(255,255,0),(0,255,255),(255,0,255),(0,0,255),(0,255,0)]
    for x in range(1,7):
      category_color.update({x:cl[x-1]})

    for batch_test in allframe_test_chunk:
      test_gen_batch = path.dataloader(args,batch_test,dicid)    
      test_preds_batch = mymodel.predict(test_gen_batch)
      print('check accuracy')
      IOU,category_score,tacx,tprx,trex,tfsx = run_instance(test_preds_batch,batch_test,name,args,y_pred,y_true,dicid,IOU,category_score,category_label,category_color)
      tac+=tacx; tpr+=tprx; tre+=trex; tfs+=tfsx;
      
    print('****')
    print(np.mean(IOU))
    with open('category_score'+name+'.csv', 'w') as f:
        f.write("%s,%s,%s,%s,%s\n"%('Class','IOU','Recall','Precision','FS'))
        for key in category_score.keys():
            dr = category_score[key][4];
            if dr==0:
              dr=1;
            f.write("%s,%s,%s,%s,%s\n"%(key,category_score[key][0]/dr,category_score[key][1]/dr,category_score[key][2]/dr,category_score[key][3]/dr))
            
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
    dispatcher_loader={1:path.dataloader,2:path.dataloader_2i}

    x = 1200;full_result=[];tac=0;tpr=0;tre=0;tfs=0;
    final_list= lambda test_list, x: [test_list[i:i+x] for i in range(0, len(test_list), x)]
    allframe_test_chunk=final_list(allframe_test, x);
    category_score={};category_score.update({0:[0,0]})
    for x in args.classid:
      category_score.update({x:[0,0]})
    for batch_test in allframe_test_chunk:
      #test_gen_batch = path.dataloader(args,batch_test,dicid) 
      test_gen_batch = dispatcher_loader[args.branch_input](args,batch_test,dicid)   
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
            
    rgb = np.asarray(rgb)
    frame=test_preds[ii]
    if args.corrector=='crf':
        import denseCRF
        w1    = 10.0  # weight of bilateral term
        alpha = 80    # spatial std
        beta  = 13    # rgb  std
        w2    = 3.0   # weight of spatial term
        gamma = 3     # spatial std
        it    = 5.0   # iteration
        param = (w1, alpha, beta, w2, gamma, it)
    
        #unary_potentials = F.one_hot(torch.from_numpy(bestcluster).long(), num_classes=2)
        mask = denseCRF.densecrf(rgb, frame, param)  # (H_pad, W_pad)
    elif args.corrector=='':
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


  
  
def run_semantic(test_preds,allpath,name,args,y_pred,y_true,dicid,IOU,category_score,category_label,category_color):
  flag_multi=0;
  if args.num_instance>1:
        flag_multi=1;
    
  Taccuracy=0
  Tprecision=0
  Trecall=0
  TFS=0;
  li = args.imagesize[0]*args.imagesize[1]
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

    
    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
    rgb = np.asarray(rgb)
    result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask>=2)

    result[tp]=((0,255,0)+result[tp])//2
 
    
    accuracy = len(tpc[0])/(li)
    true_mask =temp*mask; true_label = list(set(true_mask.ravel().tolist()))

    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')    
    fps = np.where(fast_res!=0);temp[fps]=1;
    miss_mask =temp*mask; miss_label = list(set(miss_mask.ravel().tolist()))
    
    #catx = list(set(gtn.ravel().tolist()))
    Sre=0;Spr=0;Sfs=0;
    for cls in cat:
        if cls!=0:
            #fp = np.where(miss_mask==cls);
            tp = np.where(true_mask==cls);
            gtp = np.where(gtn==cls);
            msp = np.where(mask==cls);
            
            recall = len(tp[0]) / len(gtp[0]);Sre+=recall
            if len(msp[0])!=0:
                precision = len(tp[0]) / len(msp[0]);
            else:
                precision=0;
            Spr+=precision
            try:
              FS = (2*recall*precision)/(precision+recall);Sfs+=FS
            except:
              FS = 0

            
            newiou = category_score[cls][0] + iou
            newre = category_score[cls][1] + recall
            newpr = category_score[cls][2] + precision
            newfs = category_score[cls][3] + FS
            newC = category_score[cls][4] + 1
            category_score.update({cls:[newiou,newre,newpr,newfs,newC]})
    
    Taccuracy+=accuracy
    Tprecision+=(Spr/len(cat))
    Trecall+=(Sre/len(cat))
    TFS+=(Sfs/len(cat))
    
 
    font = cv2.FONT_HERSHEY_SIMPLEX;al=2;
    footer2 = np.zeros((40,args.imagesize[1],3),'uint8')+255;
    for miss in miss_label:
        if miss!=0:
            color = category_color[miss]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
            fp = np.where(miss_mask==miss);
            result[fp]=(color+ result[fp])//2
            text = category_label[miss] + ' ' + str((len(fp[0])*100)/li)[:4]+'%'
            cv2.putText(footer2, text, (al,footer2.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=120;
    footer1 = np.zeros((40,args.imagesize[1],3),'uint8')+200;al=2;
    for true in true_label:
        if true!=0:
          color = category_color[true]
          color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
          tp = np.where(true_mask==true); gtp = np.where(gtn==true)
          text = category_label[true] + ' ' + str((len(tp[0])*100)/li)[:4]+'% | ' + str((len(gtp[0])*100)/li)[:4]+'% ' 
          cv2.putText(footer1, text, (al,footer2.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=160;
    
    result = np.concatenate((result,footer1,footer2),axis=0);    
    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
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
  print('---------')
  print("IOU",np.mean(IOU))
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

def run_instance(test_preds,allpath,name,args,y_pred,y_true,dicid,IOU,category_score,category_label,category_color):
  flag_multi=0;
  if args.num_instance>1:
        flag_multi=1;
    
  Taccuracy=0
  Tprecision=0
  Trecall=0
  TFS=0;
  li = args.imagesize[0]*args.imagesize[1]
  for ii in range(len(test_preds)):
    path = allpath[ii]
    frameindex= list(path.keys())[0]
    imagepath = path[frameindex][0]
    #tep.append(imagepath)
    seq = path[frameindex][1]
    flagmulti = path[frameindex][2]
    mask = seq.load_multi_masks_instance([frameindex]);
    
    
    
    # resize image
    dim = (args.imagesize[1],args.imagesize[0])
    gtn = cv2.resize(mask, dim, interpolation = cv2.INTER_NEAREST)
    cat = seq.load_class([frameindex],dicid);


    frame=test_preds[ii]

    mask = np.argmax(frame, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask[:,:,0];

    cat  = list(set(gtn.ravel().tolist()))
    cat_mask = list(set(mask.ravel().tolist()))
    
      
    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
    rgb = np.asarray(rgb)
    result1 = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    result2 = rgb.copy();
    
    footer1 = np.zeros((40,args.imagesize[1],3),'uint8');al=2;
    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(footer1, str(len(cat)-1), (al,footer1.shape[0]-20), font, 0.7, (255,0,0), 2, cv2.LINE_AA);al+=60;

    for cls in cat:
        if cls!=0:
            #temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
            gtp = np.where(gtn==cls);
            color = category_color[cls]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))             
            result1[gtp]=(color+ result1[gtp])//2
            text = str((len(gtp[0])*100)/li)[:4]+'%' 
            cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=60;
            
    footer2 = np.zeros((40,args.imagesize[1],3),'uint8');al=2;
    font = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(footer2, str(len(cat_mask)-1), (al,footer2.shape[0]-20), font, 0.7, (255,0,0), 2, cv2.LINE_AA);al+=60;
    
    for cls in cat_mask:
        if cls!=0:
            #temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
            msp = np.where(mask==cls);
            color = category_color[cls]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))             
            result2[msp]=(color+ result2[msp])//2        
            text = str((len(msp[0])*100)/li)[:4]+'%' 
            cv2.putText(footer2, text, (al,footer2.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=60;
            


    result1 = np.concatenate((result1,footer1),axis=0); 
    result2 = np.concatenate((result2,footer2),axis=0); 

    result = np.concatenate((result1,result2),axis=1);    
    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    try:
      os.mkdir('result')
    except:
      pass
    res.save('result/'+filename)
      
    """
    y_true = gtn;y_pred=mask;smooth=1e-7
    y_pred = y_pred.ravel().tolist()
    y_true = y_true.ravel().tolist()
    m = tf.keras.metrics.MeanIoU(num_classes=41)
    m.update_state(y_pred, y_true)
    iou=m.result().numpy()
    IOU.append(iou)

    
    rgb = load_img(args.basepath+'train/'+imagepath, target_size=args.imagesize)
    rgb = np.asarray(rgb)
    result = rgb.copy();# np.zeros((args.imagesize[0],args.imagesize[1],3),'uint8')
    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')
    
    fast_res=mask-gtn
    tpc = np.where(fast_res==0);temp[tpc]=1;
    tp = np.where(temp+mask>=2)

    result[tp]=((0,255,0)+result[tp])//2
 
    
    accuracy = len(tpc[0])/(li)
    true_mask =temp*mask; true_label = list(set(true_mask.ravel().tolist()))

    temp =  np.zeros((args.imagesize[0],args.imagesize[1]),'uint8')    
    fps = np.where(fast_res!=0);temp[fps]=1;
    miss_mask =temp*mask; miss_label = list(set(miss_mask.ravel().tolist()))
    
    #catx = list(set(gtn.ravel().tolist()))
    Sre=0;Spr=0;Sfs=0;
    for cls in cat:
        if cls!=0:
            #fp = np.where(miss_mask==cls);
            tp = np.where(true_mask==cls);
            gtp = np.where(gtn==cls);
            msp = np.where(mask==cls);
            
            recall = len(tp[0]) / len(gtp[0]);Sre+=recall
            if len(msp[0])!=0:
                precision = len(tp[0]) / len(msp[0]);
            else:
                precision=0;
            Spr+=precision
            try:
              FS = (2*recall*precision)/(precision+recall);Sfs+=FS
            except:
              FS = 0

            
            newiou = category_score[cls][0] + iou
            newre = category_score[cls][1] + recall
            newpr = category_score[cls][2] + precision
            newfs = category_score[cls][3] + FS
            newC = category_score[cls][4] + 1
            category_score.update({cls:[newiou,newre,newpr,newfs,newC]})
    
    Taccuracy+=accuracy
    Tprecision+=(Spr/len(cat))
    Trecall+=(Sre/len(cat))
    TFS+=(Sfs/len(cat))
    
 
    font = cv2.FONT_HERSHEY_SIMPLEX;al=2;
    footer2 = np.zeros((40,args.imagesize[1],3),'uint8')+255;
    for miss in miss_label:
        if miss!=0:
            color = category_color[miss]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
            fp = np.where(miss_mask==miss);
            result[fp]=(color+ result[fp])//2
            text = category_label[miss] + ' ' + str((len(fp[0])*100)/li)[:4]+'%'
            cv2.putText(footer2, text, (al,footer2.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=120;
    footer1 = np.zeros((40,args.imagesize[1],3),'uint8')+200;al=2;
    for true in true_label:
        if true!=0:
          color = category_color[true]
          color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ])) 
          tp = np.where(true_mask==true); gtp = np.where(gtn==true)
          text = category_label[true] + ' ' + str((len(tp[0])*100)/li)[:4]+'% | ' + str((len(gtp[0])*100)/li)[:4]+'% ' 
          cv2.putText(footer1, text, (al,footer2.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=160;
    
   
    
    
  lendata=len(test_preds)
  tac = Taccuracy/lendata
  tpr = Tprecision/lendata
  tre = Trecall/lendata
  tfs = TFS/lendata
  print('---------')
  print("IOU",np.mean(IOU))
  print("accuracy",tac)
  print("precision",tpr)
  print("recall",tre)
  print("FS",tfs)
  print('---------')
  """
  return IOU,category_score,Taccuracy,Tprecision,Trecall,TFS
