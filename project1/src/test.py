
import numpy as np 
import cv2, glob
import pickle

from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from train_batches import SGDC_on_img
from filters import get_filter_bank, oriented_energy, filter_img
from convolution import convolve
from segRead import readSegEdge, readSeg
from features import ft1, ft2, ft3, ft4
 
    
TEST_PATH = 'input/BSDS300/images/test/*.jpg' 
SEG_PATH = 'input/BSDS300/human/*/*/*.seg'

print('Generating filter bank...')
filter_bank = get_filter_bank(scales=[0.5,0.75,0.9], kernel_size=11, show=False)
print('done')

# load segmentations
print('Loading segmentation files...')
segs_fnames= glob.glob(SEG_PATH)
print('Finished')


# load images
print('Loading images files...')
filenames = glob.glob(TEST_PATH)
images = [{'img':cv2.imread(img_path, 0), 'id':int(img_path.split('.')[0].split('/')[-1]), 'labels':[]} for img_path in filenames] 
print('Finished')


# join each img with all its diff labels 
# TODO: do this in a less unefficient wat
for img in images:
    for seg in segs_fnames:
        if int(seg.split('.')[0].split('/')[-1]) == img['id']: 
            img['labels'].append(seg) 

# Split train/validation
# images = images[0:10] # reduced for easier debbuging 
test_size = int(len(images)*1)
imgs_test = images[:test_size][0:20]
imgs_val = images[test_size:]
print(f'Train:{len(imgs_test)} Validation:{len(imgs_val)}')

# FEATURES: 
# #1
# Create the filters based on the oriented energy (1) per scale:, plus the center-surrounded one. 
# filtered = oriented_energy(filter_bank, images)
clf1_preTrained = pickle.load(open('input/model1.pkl', 'rb'))
# #2
# Create the filters based on the filter bank.
clf2_preTrained = pickle.load(open('input/model2.pkl', 'rb')) 
# #3
# Create the filters based on the even filters, plus the center-surrounded one.
clf3_preTrained = pickle.load(open('input/model3.pkl', 'rb')) 
# #4
# Create the filters based on the odd filters, plus the center-surrounded one. 
clf4_preTrained = pickle.load(open('input/model4.pkl', 'rb')) 

score1_avg = 0
score2_avg = 0
score3_avg = 0
score4_avg = 0
score_avg_avg = 0


# Train each classifier in a (1-image*nlabels) batch size
for i, img in enumerate(imgs_test):
    score1, score2, score3, score4, score_avg = SGDC_on_img(img, clf1_preTrained, ft1, clf2_preTrained, ft2, clf3_preTrained, ft3, clf4_preTrained, ft4, filter_bank, train=0, save_pred=False)
    print(f'{i+1}/{len(imgs_test)} images')


    score1_avg =  score1_avg + (score1/len(imgs_test))
    score2_avg =  score2_avg + (score2/len(imgs_test))
    score3_avg =  score3_avg + (score3/len(imgs_test))
    score4_avg =  score4_avg + (score4/len(imgs_test))
    score_avg_avg = score_avg_avg + (score_avg/len(imgs_test))

print(f'feature 1 score: {score1_avg}')
print(f'feature 2 score: {score2_avg}')
print(f'feature 3 score: {score3_avg}')
print(f'feature 4 score: {score4_avg}')
print(f'avg score: {score_avg_avg}')