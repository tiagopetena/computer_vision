
import numpy as np 
import cv2, glob
import pickle

from matplotlib import pyplot as plt
from multiprocessing import Pool
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


from filters import get_filter_bank, oriented_energy, filter_img
from convolution import convolve
from segRead import readSegEdge
from features import ft1, ft2, ft3, ft4
 

def SGDC_on_img(img, clf1, ft1, clf2, ft2, clf3, ft3, clf4, ft4, filter_bank, train=1, save_pred=False):
    """[summary]

    Args:
        img ([type]): [description]

    Returns:
        [type]: [description]
    """

    score1 = 0
    score2 = 0
    score3 = 0
    score4 = 0
    score_avg = 0
 
    odds, evens, oe, center_surr = filter_img(img['img'], filter_bank)
 
    rows = [] 
    for label in img['labels']:  
        rows.append({
            'odd_s1_0':odds[0][0],
            'odd_s1_30':odds[0][1],
            'odd_s1_60':odds[0][2],
            'odd_s1_90':odds[0][3],
            'odd_s1_120':odds[0][4],
            'odd_s1_150':odds[0][5],
            'odd_s2_0':odds[1][0],
            'odd_s2_30':odds[1][1],
            'odd_s2_60':odds[1][2],
            'odd_s2_90':odds[1][3],
            'odd_s2_120':odds[1][4],
            'odd_s2_150':odds[1][5],
            'odd_s3_0':odds[2][0],
            'odd_s3_30':odds[2][1],
            'odd_s3_60':odds[2][2],
            'odd_s3_90':odds[2][3],
            'odd_s3_120':odds[2][4],
            'odd_s3_150':odds[2][5],
            'even_s1_0':evens[0][0],
            'even_s1_30':evens[0][1],
            'even_s1_60':evens[0][2],
            'even_s1_90':evens[0][3],
            'even_s1_120':evens[0][4],
            'even_s1_150':evens[0][5],
            'even_s2_0':evens[1][0],
            'even_s2_30':evens[1][1],
            'even_s2_60':evens[1][2],
            'even_s2_90':evens[1][3],
            'even_s2_120':evens[1][4],
            'even_s2_150':evens[1][5],
            'even_s3_0':evens[2][0],
            'even_s3_30':evens[2][1],
            'even_s3_60':evens[2][2],
            'even_s3_90':evens[2][3],
            'even_s3_120':evens[2][4],
            'even_s3_150':evens[2][5],
            'oe_s1_0':oe[0][0],
            'oe_s1_30':oe[0][1],
            'oe_s1_60':oe[0][2],
            'oe_s1_90':oe[0][3],
            'oe_s1_120':oe[0][4],
            'oe_s1_150':oe[0][5],
            'oe_s2_0':oe[1][0],
            'oe_s2_30':oe[1][1],
            'oe_s2_60':oe[1][2],
            'oe_s2_90':oe[1][3],
            'oe_s2_120':oe[1][4],
            'oe_s2_150':oe[1][5],
            'oe_s3_0':oe[2][0],
            'oe_s3_30':oe[2][1],
            'oe_s3_60':oe[2][2],
            'oe_s3_90':oe[2][3],
            'oe_s3_120':oe[2][4],
            'oe_s3_150':oe[2][5],
            'csur_s1_0':center_surr[0],
            'csur_s2_0':center_surr[1], 
            'csur_s3_0':center_surr[2], 
            'Y':readSegEdge(label)['label'].flatten()
        }) 
    # #1
    # Create the filters based on the oriented energy (1) per scale:, plus the center-surrounded one.
    # F = (OE on each orientation + CS) at each scale => |F| = 3*(6+1)  
    # filtered = oriented_energy(filter_bank, images)

    for i, label in enumerate(rows):  
        X1 = np.array([label[key] for key in ft1]).T
        X2 = np.array([label[key] for key in ft2]).T
        X3 = np.array([label[key] for key in ft3]).T
        X4 = np.array([label[key] for key in ft4]).T

        Y = label['Y'] 

        scaler = StandardScaler()


        scaler.fit(X1)
        X1 = scaler.transform(X1)
        scaler.fit(X2)
        X2 = scaler.transform(X2)
        scaler.fit(X3)
        X3 = scaler.transform(X3)
        scaler.fit(X4)
        X4 = scaler.transform(X4)


        if train==1:
            clf1.partial_fit(X1,Y, classes=np.unique(Y))
            clf2.partial_fit(X2,Y, classes=np.unique(Y)) 
            clf3.partial_fit(X3,Y, classes=np.unique(Y))
            clf4.partial_fit(X4,Y, classes=np.unique(Y)) 
        if train == 0: 

            y1 = clf1.predict(X1) 
            y2 = clf2.predict(X2) 
            y3 = clf3.predict(X3) 
            y4 = clf4.predict(X4)  
            
            score1 = score1 + (clf1.score(X1, Y)/len(rows))
            score2 = score2 + (clf2.score(X2, Y)/len(rows))
            score3 = score3 + (clf3.score(X3, Y)/len(rows))
            score4 = score4 + (clf4.score(X4, Y)/len(rows))
            score_avg = (score1+score2+score3+score4)/4


    if train == 0:
        if save_pred == True:
            plt.imshow(y1.reshape(321, 481))
            plt.show()
            plt.imshow(y2.reshape(321, 481))
            plt.show()
            plt.imshow(y3.reshape(321, 481))
            plt.show()
            plt.imshow(y4.reshape(321, 481))
            plt.show()
        print(score1, score2, score3, score4, score_avg)

    return score1, score2, score3, score4, score_avg

def main():
    TRAINING_PATH = 'input/BSDS300/images/train/*.jpg' 
    SEG_PATH = 'input/BSDS300/human/*/*/*.seg'

    print('Generating filter bank...')
    filter_bank = get_filter_bank(scales=[0.5,1,1.5], kernel_size=11, show=False)
    print('done')

    # load segmentations
    print('Loading segmentation files...')
    segs_fnames= glob.glob(SEG_PATH)
    print('Finished')
  
    # load images
    print('Loading images files...')
    filenames = glob.glob(TRAINING_PATH)
    images = [{'img':cv2.imread(img_path, 0), 'id':int(img_path.split('.')[0].split('/')[-1]), 'labels':[]} for img_path in filenames] 
    print('Finished')


    # join each img with all its diff labels 
    # TODO: do this in a less unefficient wat
    for img in images:
        for seg in segs_fnames: 
            if int(seg.split('.')[0].split('/')[-1]) == img['id']:
                img['labels'].append(seg) 
    # Split train/validation
    images = images[0:200] # reduced for easier debbuging 
    train_size = int(len(images)*1)
    imgs_train = images[:train_size]
    imgs_val = images[train_size:]
    print(f'Train:{len(imgs_train)} Validation:{len(imgs_val)}')

    # FEATURES: 
    # #1
    # Create the filters based on the oriented energy (1) per scale:, plus the center-surrounded one.
    # F = (OE on each orientation + CS) at each scale => |F| = 3*(6+1)  
    # filtered = oriented_energy(filter_bank, images)

    clf1 = SGDClassifier(loss='log', n_jobs=-1, shuffle=False, max_iter=100, verbose=0, class_weight= {1: 25})
    # #2
    # Create the filters based on the filter bank.

    clf2 = SGDClassifier(loss='log', n_jobs=-1, shuffle=False, max_iter=100, verbose=0, class_weight= {1: 25})
    # #3
    # Create the filters based on the even filters, plus the center-surrounded one.

    clf3 = SGDClassifier(loss='log', n_jobs=-1, shuffle=False, max_iter=100, verbose=0, class_weight= {1: 25})
    # #4
    # Create the filters based on the odd filters, plus the center-surrounded one. 
    clf4 = SGDClassifier(loss='log', n_jobs=-1, max_iter=100, verbose=0, class_weight= {1: 25})

    # Train each classifier in a (1-image*nlabels) batch size
    for i, img in enumerate(imgs_train):
        SGDC_on_img(img, clf1, ft1, clf2, ft2, clf3, ft3, clf4, ft4, filter_bank)
        print(f'{i+1}/{len(imgs_train)} images')
    
    with open('input/model1.pkl','wb') as f:
        pickle.dump(clf1,f)
    with open('input/model2.pkl','wb') as f:
        pickle.dump(clf2,f)
    with open('input/model3.pkl','wb') as f:
        pickle.dump(clf3,f)
    with open('input/model4.pkl','wb') as f:
        pickle.dump(clf4,f)

    # validate
    score1_avg = 0
    score2_avg = 0
    score3_avg = 0
    score4_avg = 0
    score_avg_avg = 0

    for i, img in enumerate(imgs_val):
        score1, score2, score3, score4, score_avg = SGDC_on_img(img, clf1, ft1, clf2, ft2, clf3, ft3, clf4, ft4, filter_bank, train=0)
        print(f'{i+1}/{len(imgs_val)} images')

        score1_avg =  score1_avg + score1/len(imgs_val)
        score2_avg =    score2_avg + score2/len(imgs_val)
        score3_avg =    score3_avg + score3/len(imgs_val)
        score4_avg =    score4_avg + score4/len(imgs_val)
        score_avg_avg =    score_avg_avg + score_avg/len(imgs_val)

    print(f'feature 1 score: {score1_avg}')
    print(f'feature 2 score: {score2_avg}')
    print(f'feature 3 score: {score3_avg}')
    print(f'feature 4 score: {score4_avg}')
    print(f'avg score: {score_avg_avg}')
    
if __name__ == "__main__":
    main()