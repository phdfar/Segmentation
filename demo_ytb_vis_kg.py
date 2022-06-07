# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 11:40:24 2022

@author: Farnoosh
"""
import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image,ImageOps
import PIL
from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

alist=[
    "something1",
    "something12",
    "something17",
    "something2",
    "something25",
    "something10"]

alist.sort(key=natural_keys)

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def load_image2(imfile):
    newsize = (1280, 720)
    imfile  = Image.open(imfile)
    imfile  = imfile.resize(newsize)
    img = np.array(imfile).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz(img, flo,number,imfile1,folder,args):
    #img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()

    #img = PIL.ImageOps.autocontrast(flo[:,:,0])
    #img.show()
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo) ;

    flox = ImageOps.grayscale(Image.fromarray(flo.astype('uint8'), 'RGB'))
    try:
        outputfile = args.out_dir+folder
        #print(outputfile)
        os.mkdir(outputfile)
        #os.system('mkdir -p'+outputfile)
    except:
        pass
    imfile1 = imfile1.replace('jpg','png')
    flox.save(outputfile+'/'+imfile1)
    #img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()
    #data = img_flo[:, :, [2,1,0]]/255.0
    #data = data*255
    #data = data.astype(np.uint8)
    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)

    #cv2.imwrite('output/img'+str(number)+'.png',flox)
    #cv2.imwrite('output/imh'+str(number)+'.png',data)

    #res = np.hstack((img,equ)) #stacking images side-by-side
    #cv2.waitKey()

def group(args):
    groupdict={}
    allfile = glob.glob(args.path+'/*/*.jpg')
    tempnames=[];last_groupe='';p=0;list_gropue=[]
    print('all file finded : ',len(allfile))
    #print('salam')
    for a in allfile:
        sp=a.split('/'); groupe = sp[-2] ; name=sp[-1]; list_gropue.append(groupe)
        if last_groupe == groupe:
            tempnames.append(name)
        else:
            tempnames=[]
            tempnames.append(name)
        groupdict.update({groupe : tempnames})
        last_groupe = groupe;p+=1;
    #print(p)
    return groupdict,list(set(list_gropue))

def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        groupdict,list_gropue = group(args);
        for g in range(len(groupdict)):
            
            if int(args.start) <= g <= int(args.finish):
                
                images=groupdict[list_gropue[g]]
                images.sort(key=natural_keys)
                number = 0;
                
                for imfile1, imfile2 in zip(images[:-1], images[1:]):
                    
                    #print(imfile1);print(imfile2);print(number);print('---------')
                    #print(args.path+'/'+list_gropue[g]+'/'+imfile1)
                    
                    try:
                        image1 = load_image(args.path+'/'+list_gropue[g]+'/'+imfile1)
                        image2 = load_image(args.path+'/'+list_gropue[g]+'/'+imfile2)                
                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        viz(image1, flow_up,number,imfile1,list_gropue[g],args)
                    except:
                        print('ffxxxxxx')
                        image1 = load_image2(args.path+'/'+list_gropue[g]+'/'+imfile1)
                        image2 = load_image2(args.path+'/'+list_gropue[g]+'/'+imfile2)
                        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                        viz(image1, flow_up,number,imfile1,list_gropue[g],args)


                    number+=1
                    if number == len(images)-1:
                        try:
                            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
                        except:
                            print('exxxxxx')
                            image1 = load_image2(args.path+'/'+list_gropue[g]+'/'+imfile1)
                            image2 = load_image2(args.path+'/'++list_gropue[g]+'/'+imfile2)
                            flow_low, flow_up = model(image2, image1, iters=20, test_mode=True)
                        viz(image2, flow_up,number,imfile2,list_gropue[g],args)
                            
                if g%50==0:
                    print(g,'/',len(groupdict))

        name = str(args.finish)+'.tar.gz'
        os.system('tar czf '+name+ ' '+args.out_dir )            



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--out_dir', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--start', help="dataset for evaluation")
    parser.add_argument('--finish', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
