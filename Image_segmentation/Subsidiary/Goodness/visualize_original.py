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
import matplotlib.pyplot as plt
import numpy as np
import torch
#import torch.nn.functional as F
#import denseCRF
def start(mymodel,seqs,name,args):
    with open(args.basepath +'youtube_vis_val.json', 'r') as fh:
        dataset = json.load(fh)
    meta_info = dataset["meta"]

    category_label = {int(k): v for k, v in meta_info["category_labels"].items()}
    category_color={};category_color.update({0:(0,0,0)})
    for x in args.classid:
      r = np.random.randint(0,255,1)[0]
      g = np.random.randint(0,150,1)[0]
      b = np.random.randint(0,255,1)[0]
      category_color.update({x:(r,g,b)})
    run(args,mymodel,seqs,category_label,category_color)



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def get_eig(args,imagepath):
    sp = imagepath.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
    eig = np.load(args.baseinput2+eigpath) #data/VOC2012/eigs/laplacian/
      
    dim = (args.imagesize[1],args.imagesize[0])
    eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
    eig1 = NormalizeData(eig1)
    
    eig1[eig1<=0.15]=0;eig1[eig1>0.15]=1;
    eig1 = np.expand_dims(eig1,2);
    return eig1
    
def draw(outputs,inputs,imagepath,args):
    
    img = np.asarray(load_img(imagepath, target_size=args.imagesize,grayscale=False))
    eig1 = get_eig(args,imagepath)
    eig1 = np.concatenate((eig1,eig1,eig1),axis=-1);
    
    
    result = np.concatenate((img,eig1*255),axis=1);


    y_pred = float(outputs[0])
    y_pred = round(y_pred, 2)
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX;
    footer1 = np.zeros((40,result.shape[1],3),'uint8');al=2;
    text = ' y-pred ' + str(y_pred) 
    cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.6, (255,255,255), 1, cv2.LINE_AA);al+=160;
    result = np.concatenate((result,footer1),axis=0);
   
    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    res.save('result/'+filename)
    


def run(args,mymodel,seqs,category_label,category_color):
    full_result=[]
    try:
        os.mkdir('result')
    except:
        pass
    for seq in seqs:
        seq_path = seq.image_paths
        inputs1=[];inputs2=[];imagepath=[]
        for frame in seq_path:
            rgb = load_img(args.basepath+'valid/'+frame, target_size=args.imagesize)
            rgb = np.asarray(rgb)
            inputs1.append(rgb)
            inputs2.append(get_eig(args,args.basepath+'valid/'+frame))
            imagepath.append(args.basepath+'valid/'+frame)
        outputs = mymodel.predict([np.asarray(inputs1),np.asarray(inputs2)])
        for p in range(len(outputs)):
            draw(outputs[p],inputs1[p],imagepath[p],args)
            y_pred = float(outputs[p][0]);y_pred = round(y_pred, 2)
            filename = imagepath[p].split('/'); filename=filename[-2]+'_'+filename[-1]
            full_result.append((filename,y_pred))
    df = pd.DataFrame(full_result,columns =['Names','y-pred'])
    df.to_csv('result_valid_'+args.model_dir+'.csv')
    
      
