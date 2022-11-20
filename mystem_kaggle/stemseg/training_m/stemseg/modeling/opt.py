import numpy as np
import cv2
from matplotlib import pyplot as plt

def gethomography(img1,img2):
  MIN_MATCH_COUNT = 8
  # Initiate SIFT detector
  sift = cv2.SIFT_create()

  # find the keypoints and descriptors with SIFT
  kp1, des1 = sift.detectAndCompute(img1,None)
  kp2, des2 = sift.detectAndCompute(img2,None)

  if len(kp1)>2000:
    g = len(kp1)//2000
    kp1 = kp1[0::g]
    des1= des1[0::g,:]

  if len(kp2)>2000:
    g = len(kp2)//2000
    kp2 = kp2[0::g]
    des2= des2[0::g,:]

  FLANN_INDEX_KDTREE = 0
  index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
  search_params = dict(checks = 50)

  flann = cv2.FlannBasedMatcher(index_params, search_params)


  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch(des1,des2, k=2)

  #matches = flann.knnMatch(des1c,des2c,k=2)

  # store all the good matches as per Lowe's ratio test.
  good = []
  for m,n in matches:
      if m.distance < 0.7*n.distance:
          good.append(m)

  if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    """
    h,w = img1.shape
    temp1 = np.zeros((h,w))+1
    temp2 = cv2.warpPerspective(temp1, M, (w,h))
    """
    """
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                      singlePointColor = None,
                      matchesMask = matchesMask, # draw only inliers
                      flags = 2)
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    plt.imshow(img3, 'gray')
    plt.figure()
    """
    im_out = cv2.warpPerspective(img1, M, (img1.shape[1],img1.shape[0]))
    #plt.imshow(im_out)
    return im_out,1

  else:
      #print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
      matchesMask = None
      return 0,0

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)



def getopticalflow(a,b):
  
  mask = np.zeros((a.shape[0],a.shape[1],3)).astype('uint8')
    
  # Sets image saturation to maximum
  mask[..., 1] = 255

  def norm(a):
    dfmax, dfmin = a.max(), a.min()
    df = (a - dfmin)/(dfmax - dfmin)
    df = (df*255).astype('uint8')
    return df
  # Calculates dense optical flow by Farneback method
  flow = cv2.calcOpticalFlowFarneback(norm(a), norm(b), 
                                      None,
                                      0.5, 5, 5, 3, 5, 1.2, 0)
    
  # Computes the magnitude and angle of the 2D vectors
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    

  angle[magnitude<10]=0;
  magnitude[magnitude<10]=0;
  #magnitude = magnitude * 5


  # Sets image hue according to the optical flow 
  # direction
  mask[..., 0] = angle * 180 / np.pi / 2

  
  # Sets image value according to the optical flow
  # magnitude (normalized)
  mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
  # Converts HSV to RGB (BGR) color representation
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  #A = rgb[:,:,2]
  a = adjust_gamma(rgb[:,:,2], gamma=5.0)
  b = adjust_gamma(rgb[:,:,0], gamma=5.0)
  if np.max(a)>0:
    a = a / np.max(a)
  if np.max(b)>0:
    b = b / np.max(b)

  #plt.imshow(a)
  #plt.figure()
  #plt.imshow(b)

  return a,b


def getopticalflow_gray(a,b):
  
  mask = np.zeros((a.shape[0],a.shape[1],3)).astype('uint8')
    
  # Sets image saturation to maximum
  mask[..., 1] = 255

  def norm(a):
    dfmax, dfmin = a.max(), a.min()
    df = (a - dfmin)/(dfmax - dfmin)
    df = (df*255).astype('uint8')
    return df
  # Calculates dense optical flow by Farneback method
  flow = cv2.calcOpticalFlowFarneback(norm(a), norm(b), 
                                      None,
                                      0.5, 5, 5, 3, 5, 1.2, 0)
    
  # Computes the magnitude and angle of the 2D vectors
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
  
  angle[magnitude<5]=0;
  magnitude[magnitude<5]=0;
  #magnitude = magnitude * 5
  

  # Sets image hue according to the optical flow 
  # direction
  mask[..., 0] = angle * 180 / np.pi / 2

  
  # Sets image value according to the optical flow
  # magnitude (normalized)
  mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
  # Converts HSV to RGB (BGR) color representation
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  """
  #A = rgb[:,:,2]
  a = adjust_gamma(rgb[:,:,2], gamma=15.0)
  b = adjust_gamma(rgb[:,:,0], gamma=15.0)
  if np.max(a)>0:
    a = a / np.max(a)
  if np.max(b)>0:
    b = b / np.max(b)	
  rgb[:,:,2] = a;
  rgb[:,:,0] = b;

  """
  gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
  gray = adjust_gamma(gray, gamma=2.0)
  rgb[:,:,0] = gray;rgb[:,:,1] = gray;rgb[:,:,2] = gray;

  #plt.imshow(a)
  #plt.figure()
  #plt.imshow(rgb)

  return rgb


def getopticalflow_rgb(a,b):
  
  mask = np.zeros((a.shape[0],a.shape[1],3)).astype('uint8')
    
  # Sets image saturation to maximum
  mask[..., 1] = 255

  def norm(a):
    dfmax, dfmin = a.max(), a.min()
    df = (a - dfmin)/(dfmax - dfmin)
    df = (df*255).astype('uint8')
    return df
  # Calculates dense optical flow by Farneback method
  flow = cv2.calcOpticalFlowFarneback(norm(a), norm(b), 
                                      None,
                                      0.5, 5, 5, 3, 5, 1.2, 0)
    
  # Computes the magnitude and angle of the 2D vectors
  magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
  """
  angle[magnitude<5]=0;
  magnitude[magnitude<5]=0;
  #magnitude = magnitude * 5
  """

  # Sets image hue according to the optical flow 
  # direction
  mask[..., 0] = angle * 180 / np.pi / 2

  
  # Sets image value according to the optical flow
  # magnitude (normalized)
  mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    
  # Converts HSV to RGB (BGR) color representation
  rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)

  """
  #A = rgb[:,:,2]
  a = adjust_gamma(rgb[:,:,2], gamma=15.0)
  b = adjust_gamma(rgb[:,:,0], gamma=15.0)
  if np.max(a)>0:
    a = a / np.max(a)
  if np.max(b)>0:
    b = b / np.max(b)	
  rgb[:,:,2] = a;
  rgb[:,:,0] = b;

  """
  #gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
  rgb = adjust_gamma(rgb, gamma=2.0)
  #rgb[:,:,0] = gray;rgb[:,:,1] = gray;rgb[:,:,2] = gray;

  #plt.imshow(a)
  #plt.figure()
  #plt.imshow(rgb)

  return rgb
