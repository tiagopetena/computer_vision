from itertools import combinations_with_replacement
from math import degrees
from matplotlib import pyplot as plt
from scipy import ndimage as ndi
from skimage._shared.utils import convert_to_float
from skimage.transform import resize
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

def in_threshold(p, i, t):
    if i+t<p or i-t>p: 
        return True
    return False 

circle_mask = np.array([
    [0,0,1,1,1,0,0],
    [0,1,1,1,1,1,0],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1],
    [0,1,1,1,1,1,0],
    [0,0,1,1,1,0,0]
])
def build_circle(r):
    A = np.arange(-r,r+1)**2
    dists = np.sqrt(A[:,None] + A)
    return (np.abs(dists-r)<0.5).astype(int)

idx_m = np.array([
    [-3,-3,-3,-3,-3,-3,-3],
    [-2,-2,-2,-2,-2,-2,-2],
    [-1,-1,-1,-1,-1,-1,-1],
    [0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1],
    [2,2,2,2,2,2,2],
    [3,3,3,3,3,3,3],
])

def _prepare_grayscale_input_nD(image):
    image = np.squeeze(image)

    return image.astype(float)

def _compute_derivatives(image, mode='constant', cval=0):
    derivatives = [ndi.sobel(image, axis=i, mode=mode, cval=cval)
                   for i in range(image.ndim)]

    return derivatives

def structure_tensor(image, sigma=1, mode='constant', cval=0, order=None):

    if order is None:
        if image.ndim == 2:
            order = 'xy'

    image = _prepare_grayscale_input_nD(image)

    derivatives = _compute_derivatives(image, mode=mode, cval=cval)

    if order == 'xy':
        derivatives = reversed(derivatives)

    # structure tensor
    A_elems = [ndi.gaussian_filter(der0 * der1, sigma, mode=mode, cval=cval)
               for der0, der1 in combinations_with_replacement(derivatives, 2)]

    return A_elems

def corner_harris(image, method='k', k=0.05, eps=1e-6, sigma=1):
    """Compute Harris corner measure response image."""

    Arr, Arc, Acc = structure_tensor(image, sigma, order='xy')

    # determinant
    detA = Arr * Acc - Arc ** 2
    # trace
    traceA = Arr + Acc

    if method == 'k':
        response = detA - k * traceA ** 2
    else:
        response = 2 * detA / (traceA + eps)

    return response

def brief_pairs(patch_size=31, n=256, seed=10):
    patch_mask = np.zeros((patch_size,patch_size))
    np.random.seed(seed)
    scale = np.sqrt(1/25*patch_size**2)
    Xx = np.random.normal(patch_size//2, scale, patch_size).astype(int)
    np.random.seed(seed+1) 
    Xy = np.random.normal(patch_size//2, scale, patch_size).astype(int)
    X = [(x,y) for x,y in zip(Xx, Xy)]

    np.random.seed(seed+2) 
    Yx = np.random.normal(patch_size//2, scale, patch_size).astype(int)
    np.random.seed(seed+3) 
    Yy = np.random.normal(patch_size//2, scale, patch_size).astype(int)
    Y = [(x,y) for x,y in zip(Yx, Yy)]

    return X, Y

def m_pq(patch, p, q):
    xp = idx_m.T**p
    yq = idx_m**q 
    return (xp*yq*patch).sum()

class Fast():
    

    def __init__(self, threshold, img):
        self.threshold = threshold
        self.img = img
        self.height = self.img.shape[0]
        self.width = self.img.shape[1]
        self.keypoints = []
        self.corner_keypoints = []
        self.brief_pairs = brief_pairs()
        self.theta = []

    def detect(self):
        img = self.img
        height, width = img.shape 

        thresholdArr = self.threshold*img
        y = width
        x = height
        ci01 = ((img[:,3:]   < img[:, :y-3] -  thresholdArr[:,3:] )   | (img[:,3:]   >  thresholdArr[:,3:]   + img[:, :y-3]))
        ci02 = ((img[:x-3,:] < img[3:, :]   -  thresholdArr[:x-3,:] ) | (img[:x-3,:] >  thresholdArr[:x-3,:] + img[3:, :]))
        ci03 = ((img[:,:y-3] < img[:, 3:]   -  thresholdArr[:,:y-3] ) | (img[:,:y-3] >  thresholdArr[:,:y-3] + img[:, 3:]))
        ci04 = ((img[3:,:]   < img[:x-3,:]  -  thresholdArr[3:,:])    | (img[3:,:]   >  thresholdArr[3:,:]   + img[:x-3,:]))

        #x+1 y-3
        ci05 = ((img[:x-1,3:]   < img[1:, :y-3] -  thresholdArr[:x-1,3:] )   | (img[:x-1,3:]   >  thresholdArr[:x-1,3:]   + img[1:, :y-3]))
        #x+2 y-2
        ci06 = ((img[:x-2,2:]   < img[2:, :y-2] -  thresholdArr[:x-2,2:] )   | (img[:x-2,2:]   >  thresholdArr[:x-2,2:]   + img[2:, :y-2]))
        #x+3 y-1
        ci07 = ((img[:x-3,1:]   < img[3:, :y-1] -  thresholdArr[:x-3,1:] )   | (img[:x-3,1:]   >  thresholdArr[:x-3,1:]   + img[3:, :y-1]))
        #x+2 y+2
        ci08 = ((img[:x-2,:y-2]   < img[2:, 2:] -  thresholdArr[:x-2,:y-2] )   | (img[:x-2,:y-2]   >  thresholdArr[:x-2,:y-2]   + img[2:, 2:]))
        #x+1 y+3
        ci09 = ((img[:x-1,:y-3]   < img[1:, 3:] -  thresholdArr[:x-1,:y-3] )   | (img[:x-1,:y-3]   >  thresholdArr[:x-1,:y-3]   + img[1:, 3:]))
        #x-1 y+3
        ci10 = ((img[1:,:y-3]   < img[:x-1,3:]  -  thresholdArr[1:,:y-3])    | (img[1:,:y-3]   >  thresholdArr[1:,:y-3]   + img[:x-1,3:]))
        #x-2 y+2
        ci11 = ((img[2:,:y-2]   < img[:x-2,2:]  -  thresholdArr[2:,:y-2])    | (img[2:,:y-2]   >  thresholdArr[2:,:y-2]   + img[:x-2,2:]))
        #x-3 y-1
        ci12 = ((img[3:,1:]   < img[:x-3,:y-1]  -  thresholdArr[3:,1:])    | (img[3:,1:]   >  thresholdArr[3:,1:]   + img[:x-3,:y-1]))
        #x-2 y-2
        ci13 = ((img[2:,2:]   < img[:x-2,:y-2]  -  thresholdArr[2:,2:])    | (img[2:,2:]   >  thresholdArr[2:,2:]   + img[:x-2,:y-2]))
        #x-1 y-3
        ci14 = ((img[1:,3:]   < img[:x-1,:y-3]  -  thresholdArr[1:,3:])    | (img[1:,3:]   >  thresholdArr[1:,3:]   + img[:x-1,:y-3]))
        #x+2 y-1
        ci15 = ((img[:x-2,1:]   < img[2:, :y-1] -  thresholdArr[:x-2,1:] )   | (img[:x-2,1:]   >  thresholdArr[:x-2,1:]   + img[2:, :y-1]))
        #x-3 y+1
        ci16 = ((img[3:,:y-1]   < img[:x-3,1:]  -  thresholdArr[3:,:y-1])    | (img[3:,:y-1]   >  thresholdArr[3:,:y-1]   + img[:x-3,1:]))

        keyPointsCount = (1*ci01[3:x-3,:y-3-3] + 1*ci02[3:,3:y-3] + 1*ci03[3:x-3,3:] + 1* ci04[:x-3-3,3:y-3]+1*ci05[3:x-1-2,:y-3-3]+1*ci06[3:x-2-1,1:y-2-3]
            +1*ci07[3:,2:y-1-3]+1*ci08[3:x-2-1,3:y-2-1]+1*ci09[3:x-1-2,3:]+1*ci10[2:x-1-3,3:]+1*ci11[1:x-2-3,3:y-2-1]+1*ci12[:x-3-3,2:y-1-3]
            +1*ci13[1:x-2-3,1:y-2-3]+1*ci14[2:x-1-3,:y-3-3]+1*ci15[2:x-1-3,:y-3-3]+1*ci16[:x-3-3,3:y-1-2])
        circles = np.where(keyPointsCount>=12)
        for i,x in  enumerate(circles[0]):
                self.keypoints.append((circles[1][i],circles[0][i]))  

        self.oriented_fast()

    def oriented_fast(self, N=100):
        corner = corner_harris(self.img)

        corner_kp = []
        for kp in self.keypoints:
            if kp[1]<=3 or kp[0]<=3 or kp[1]>=self.height-4 or kp[0]>=self.width-4:
                continue
            corner_kp.append((kp, corner[kp[1], kp[0]]))

        print('n keypoints =', len(self.keypoints))
        self.corner_keypoints = sorted(corner_kp, key=lambda x:x[1], reverse=True)[:N+1]
        # self.keypointsMap = np.zeros((self.img.shape[0], self.img.shape[1]),dtype=np.float)
        # self.thetaMap = np.zeros((self.img.shape[0], self.img.shape[1]),dtype=np.float)
        oriented_keypoints = []
        thetas = []
        for kp in self.corner_keypoints:
            patch = self.img[kp[0][1]-3:kp[0][1]+4, kp[0][0]-3:kp[0][0]+4] 
            patched_circle = patch*circle_mask 
            m10 = m_pq(patched_circle, 1, 0)
            m01 = m_pq(patched_circle, 0, 1)
            m00 = m_pq(patched_circle, 0, 0)
            theta = np.arctan2(m01,m10)
            C = (m10/m00, m01/m00) 
            oriented_keypoints.append(kp[0])
            thetas.append(theta)
            # self.keypointsMap[kp[0][1],kp[0][0]]=1
            # self.thetaMap[kp[0][1],kp[0][0]]=theta
        self.keypoints = oriented_keypoints
        self.theta = thetas

        # print(oriented_keypoints)