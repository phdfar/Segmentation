from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import cv2
import fire
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm
import extract_utils as utils
from cv2 import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img



def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

model_name = 'dino_vits16'

which_block=-1
patch_size = 16
output_dict = {}
accelerator = Accelerator(fp16=True, cpu=True)
model, val_transform, patch_size, num_heads = utils.get_model(model_name)

if 'dino' in model_name or 'mocov3' in model_name:
    feat_out = {}
    def hook_fn_forward_qkv(module, input, output):
        feat_out["qkv"] = output
    model._modules["blocks"][which_block]._modules["attn"]._modules["qkv"].register_forward_hook(hook_fn_forward_qkv)
else:
    raise ValueError(model_name)

name = 'C:/Users/Tarasheh/Desktop/deep-spectral-segmentation/extract/image.jpg'
#images = cv2.imread(name)
images = load_img(name)
images = np.asarray(images)

images = tf.transpose(images, perm=[2,0,1]);
images = images.numpy(); images = np.expand_dims(images,0)
images = torch.Tensor(images)


# Reshape image
P = patch_size
B, C, H, W = images.shape
H_patch, W_patch = H // P, W // P
H_pad, W_pad = H_patch * P, W_patch * P
T = H_patch * W_patch + 1  # number of tokens, add 1 for [CLS]
# images = F.interpolate(images, size=(H_pad, W_pad), mode='bilinear')  # resize image
images = images[:, :, :H_pad, :W_pad]
images = images.to(accelerator.device)

# Forward and collect features into output dict
if 'dino' in model_name or 'mocov3' in model_name:
    # accelerator.unwrap_model(model).get_intermediate_layers(images)[0].squeeze(0)
    model.get_intermediate_layers(images)[0].squeeze(0)
    # output_dict['out'] = out
    output_qkv = feat_out["qkv"].reshape(B, T, 3, num_heads, -1 // num_heads).permute(2, 0, 3, 1, 4)
    # output_dict['q'] = output_qkv[0].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
    output_dict['k'] = output_qkv[1].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
    # output_dict['v'] = output_qkv[2].transpose(1, 2).reshape(B, T, -1)[:, 1:, :]
else:
    raise ValueError(model_name)

# # Metadata
# output_dict['indices'] = indices[0]
# output_dict['file'] = files[0]
# output_dict['id'] = id
# output_dict['model_name'] = model_name
# output_dict['patch_size'] = patch_size
# output_dict['shape'] = (B, C, H, W)
output_dict = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in output_dict.items()}


# Load affinity matrix
feats = output_dict['k'].squeeze().cpu()
featsnorm = F.normalize(feats, p=2, dim=-1)

W_feat = (featsnorm @ featsnorm.T)
W_feat = (W_feat * (W_feat > 0))
W_feat = W_feat / W_feat.max()  # NOTE: If features are normalized, this naturally does nothing
W_feat = W_feat.cpu().numpy()
      
 # Combine
W_comb = W_feat #+ W_color * image_color_lambda  # combination
D_comb = np.array(utils.get_diagonal(W_comb).todense())  # is dense or sparse faster? not sure, should check
K=5;    
eigenvalues, eigenvectors = eigsh(D_comb - W_comb, k=K, sigma=0, which='LM', M=D_comb)
eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors.T).float()
  
eigenvectors = eigenvectors.numpy()
eig = eigenvectors[1,:];
plt.imshow(eig.reshape(H_patch,W_patch))
plt.figure()
plt.imshow(W_feat)
plt.figure()
plt.imshow(D_comb - W_comb)

mask = np.asarray(load_img('mask.png',target_size=(45,80)))[:,:,0]
mask2 = mask.copy()
mask[mask!=250]=0;mask[mask==250]=1;
mask2[mask2!=236]=0;mask2[mask2==236]=1;


eigenvectorsx = eigenvectors.copy()
#eigenvectorsx[1,:] = mask.reshape(45*80)
#eigenvectorsx[2,:] = mask.reshape(45*80)

#eigenvectorsx[0,:] = eigenvectorsx[0,:] *  mask.reshape(45*80)

eigenvectorsx[1,:] = 5*eigenvectorsx[1,:] *  mask2.reshape(45*80)
eigenvectorsx[2,:] = 5*eigenvectorsx[2,:] *  mask2.reshape(45*80)

eigenvectorsx[3,:] = 5*eigenvectorsx[3,:] *  mask.reshape(45*80)
eigenvectorsx[4,:] = 5*eigenvectorsx[4,:] *  mask.reshape(45*80)

X = D_comb - W_comb
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered)

#ist = np.dot(X_centered, pca.components_)
XZ = np.dot(X_centered, eigenvectorsx.T)
    
Xhat = np.dot(XZ, eigenvectorsx)
Xhat += np.mean(X, axis=0) 


eigenvaluesx, eigenvectorsx = eigsh(D_comb-Xhat,k=K, sigma=0, which='LM', M=D_comb)
#eigenvaluesx = eigenvaluesx.numpy()
eig = eigenvectorsx.T[1,:];
plt.figure()
plt.imshow(eig.reshape(H_patch,W_patch))

eig = eigenvectorsx.T[2,:];
plt.figure()
plt.imshow(eig.reshape(H_patch,W_patch))

asd

import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification


X, y = make_classification(n_samples=1000)
n_samples = X.shape[0]

pca = PCA()
X_transformed = pca.fit_transform(X)

# We center the data and compute the sample covariance matrix.
X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / n_samples
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)
    print('--------')
    

import numpy as np
import sklearn.datasets, sklearn.decomposition

X = sklearn.datasets.load_iris().data
mu = np.mean(X, axis=0)

pca = sklearn.decomposition.PCA()
pca.fit(X)

nComp = 2
Xhat = np.dot(pca.transform(X)[:,:nComp], pca.components_[:nComp,:])
Xhat += mu

print(Xhat[0,])

# # Save
# accelerator.save(output_dict, str(output_file))
# accelerator.wait_for_everyone()

X_transformed = pca.transform(X)

X_centered = X - np.mean(X, axis=0)
cov_matrix = np.dot(X_centered.T, X_centered) / 150

#ist = np.dot(X_centered, pca.components_)
ist = np.dot(X_centered, pca.components_.T[:,:1])
    
Xhat = np.dot(ist[:,:1], pca.components_[:1,:])
Xhat += mu
