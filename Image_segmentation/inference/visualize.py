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
import torch.nn.functional as F
import denseCRF
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


def draw_semantic(outputs,mask,img,cat_mask,size,category_color,category_label,args,imagepath):
    font = cv2.FONT_HERSHEY_SIMPLEX;
    footer1 = np.zeros((40,args.imagesize[1],3),'uint8')+200;al=2;
    rgb = img.copy()
    for lb in cat_mask:
        if lb!=0:
            color = category_color[lb]
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
            tp = np.where(mask==lb);
            text = category_label[lb] + ' ' + str((len(tp[0])*100)/size)[:4]+'%'
            cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=160;
            rgb[tp]=(color+ rgb[tp])//2
            
    result = np.concatenate((rgb,footer1),axis=0);
    res = keras.preprocessing.image.array_to_img(result)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    res.save('result/'+filename)
        
def draw_binary(outputs,mask,img,cat_mask,size,category_color,category_label,args,imagepath):
    #font = cv2.FONT_HERSHEY_SIMPLEX;
    #footer1 = np.zeros((40,args.imagesize[1],3),'uint8')+200;al=2;
    w1    = 10.0  # weight of bilateral term
    alpha = 80    # spatial std
    beta  = 13    # rgb  std
    w2    = 3.0   # weight of spatial term
    gamma = 3     # spatial std
    it    = 5.0   # iteration
    param = (w1, alpha, beta, w2, gamma, it)
    rgb = img.copy()
    for lb in cat_mask:
        if lb!=0:
            #color = category_color[lb]
            color = (0,255,0)
            tp = np.where(mask==lb);
            #text = category_label[lb] + ' ' + str((len(tp[0])*100)/size)[:4]+'%'
            #cv2.putText(footer1, text, (al,footer1.shape[0]-20), font, 0.4, color, 1, cv2.LINE_AA);al+=160;
            rgb[tp]=(color+ rgb[tp])//2
    if args.corrector=='crf':
        rgb_c = img.copy()
        #unary_potentials = F.one_hot(torch.from_numpy(bestcluster).long(), num_classes=2)
        segmap_crf = denseCRF.densecrf(rgb, outputs, param)  # (H_pad, W_pad)
        for lb in [0,1]:
            if lb!=0:
                color = (0,255,0)
                tp = np.where(segmap_crf==lb);                
                rgb_c[tp]=(color+ rgb_c[tp])//2
                rgb = np.concatenate((rgb,rgb_c),axis=1);
            
    #result = np.concatenate((rgb,footer1),axis=0);
    res = keras.preprocessing.image.array_to_img(rgb)
    filename = imagepath.split('/'); filename=filename[-2]+'_'+filename[-1]
    res.save('result/'+filename)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def offline(path,args):
    mask=[]
    for i in path:
        sp = i.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
        eig = np.load(args.model_dir+eigpath)
        dim = (args.imagesize[1],args.imagesize[0])
        f = NormalizeData(eig[:,:,1])
        f = cv2.resize(f, dim, interpolation = cv2.INTER_NEAREST)
        tr=0.15
        f[f<=tr]=0;f[f>tr]=1;
        mask.append(f)
    return np.asarray(mask)

def run(args,mymodel,seqs,category_label,category_color):
    dispatcher_draw={'semantic':draw_semantic,'binary':draw_binary}
    try:
        os.mkdir('result')
    except:
        pass
    size = args.imagesize[1]*args.imagesize[0]
    for seq in seqs:
        seq_path = seq.image_paths
        inputs=[];imagepath=[]
        for frame in seq_path:
            rgb = load_img(args.basepath+'valid/'+frame, target_size=args.imagesize)
            rgb = np.asarray(rgb)
            inputs.append(rgb)
            imagepath.append(args.basepath+'valid/'+frame)
        if 'h5' in args.model_dir:
            outputs = mymodel.predict(np.asarray(inputs))
        else:
            outputs = offline(imagepath,args)
            
        for p in range(len(outputs)):
            mask = outputs[p]
            if 'h5' in args.model_dir:
                mask = np.argmax(mask, axis=-1)
                mask = np.expand_dims(mask, axis=-1)
                mask = mask[:,:,0];
            cat_mask = list(set(mask.ravel().tolist()))
            dispatcher_draw[args.task](outputs[p],mask,inputs[p],cat_mask,size,category_color,category_label,args,imagepath[p])
    
      
