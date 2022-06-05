from tensorflow.keras import layers
from tensorflow import keras
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def loadeig(myself,imagepath):
    
    sp = imagepath.split('/'); name=sp[-1].replace('.jpg','.pth.npy');eigpath = sp[-2]+'_'+name;
    eig = np.load(myself.baseinput+'data/VOC2012/eigs/laplacian/'+eigpath)
    if myself.config==0:
        dim = (myself.img_size[1],myself.img_size[0])
        e1 = eig[:,:,1];
        eig = cv2.resize(e1, dim, interpolation = cv2.INTER_NEAREST)
        eig = NormalizeData(eig)
        eig = np.expand_dims(eig,2);
        return eig
    elif myself.config==1:
        img = load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=True)
        #img = NormalizeData(np.asarray(img));
        img = np.expand_dims(img,2);
        dim = (myself.img_size[1],myself.img_size[0])
        e1 = eig[:,:,1];
        eig = cv2.resize(e1, dim, interpolation = cv2.INTER_NEAREST)
        eig = NormalizeData(eig)
        eig = np.expand_dims(eig,2);
        return np.concatenate((img,eig),axis=-1)
    elif myself.config==2:
        img = np.asarray(load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=False))
        img = NormalizeData(np.asarray(img));
        #img = np.expand_dims(img,2);
        dim = (myself.img_size[1],myself.img_size[0])
        e1 = eig[:,:,1];
        eig = cv2.resize(e1, dim, interpolation = cv2.INTER_NEAREST)
        eig = NormalizeData(eig)
        eig = np.expand_dims(eig,2);
        return np.concatenate((img,eig),axis=-1)
    elif myself.config==3:
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        eig2 = NormalizeData(eig2)
        eig1 = np.expand_dims(eig1,2);eig2 = np.expand_dims(eig2,2);
        return np.concatenate((eig1,eig2),axis=-1)
    elif myself.config==4:
        img = load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=True)
        #img = NormalizeData(np.asarray(img));
        img = np.expand_dims(img,2);
        
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        eig2 = NormalizeData(eig2)
        eig1 = np.expand_dims(eig1,2);eig2 = np.expand_dims(eig2,2);
        return np.concatenate((img,eig1,eig2),axis=-1)
    elif myself.config==5:
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        eig2 = NormalizeData(eig2)
        eig3 = cv2.resize(eig[:,:,3], dim, interpolation = cv2.INTER_NEAREST)
        eig3 = NormalizeData(eig3)
        eig1 = np.expand_dims(eig1,2);eig2 = np.expand_dims(eig2,2);eig3 = np.expand_dims(eig3,2);
        return np.concatenate((eig1,eig2,eig3),axis=-1)
    elif myself.config==6:
        img = load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=False)
        #img = NormalizeData(np.asarray(img));
        #img = np.expand_dims(img,2);
        
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        eig2 = NormalizeData(eig2)
        eig1 = np.expand_dims(eig1,2);eig2 = np.expand_dims(eig2,2);
        return np.concatenate((img,eig1,eig2),axis=-1)
    elif myself.config==7:
        img = np.asarray(load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=False))
        #img = NormalizeData(np.asarray(img));
        #img = np.expand_dims(img,2);
        
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        #eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        #eig2 = NormalizeData(eig2)
        eig1 = np.expand_dims(eig1,2);#eig2 = np.expand_dims(eig2,2);
        #z = np.concatenate((eig1,eig2),axis=-1)
        return [img,eig1]
    elif myself.config==8:
        img = np.asarray(load_img(myself.basepath+'train/'+imagepath, target_size=myself.img_size,grayscale=False))
        #img = NormalizeData(np.asarray(img));
        #img = np.expand_dims(img,2);
        
        dim = (myself.img_size[1],myself.img_size[0])
        eig1 = cv2.resize(eig[:,:,1], dim, interpolation = cv2.INTER_NEAREST)
        eig1 = NormalizeData(eig1)
        eig2 = cv2.resize(eig[:,:,2], dim, interpolation = cv2.INTER_NEAREST)
        eig2 = NormalizeData(eig2)
        eig1 = np.expand_dims(eig1,2);eig2 = np.expand_dims(eig2,2);
        eig1[eig1<=0.15]=0;
        z = np.concatenate((eig1,eig2),axis=-1)
        
        return [img,z]


