#!/usr/bin/env python3
import os
import cv2
import random
import numpy as np
from skimage.transform import rescale
from scipy.signal import convolve2d
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from embed_panebbianco import embeddedFinalMethod
from detection_panebbianco import extractWatermark, similarity

methodName = "panebbianco_threshold_absTH"
numTestImages = 101 # The number of images to be tested from 0 to 101

# @brief Add awgn to the image
#
# @param img    the original image
# @param std    the standard deviation of the gaussian noise
# @param seed   the seed for the generator
#
# @return the image with the noise
def awgn(img, std, seed):
  mean = 0.0
  np.random.seed(seed)
  outImg = img + np.random.normal(mean, std, img.shape)
  outImg = np.clip(outImg, 0, 255)
  return outImg


# @brief Add blur to the image
#
# @param img    the original image
# @param sigma  the sigma parameter for the gaussian filter
#
# @return the blur image
def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  outImg = gaussian_filter(img, sigma)
  return outImg


# @brief sahrp an image
#
# @param img    the original image
# @param alpha  the alpha parameter for the gaussian filter
# @param sigma  the sigma parameter for the gaussian filter
#
# @return the sharpened image
def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  outImg = img + alpha * (img - filter_blurred_f)
  return outImg

# @brief apply a median filter to the image
#
# @param img    the original image
# @param kernel_size  the size of the kernel
#
# @return the modified image
def median(img, kernel_size):
  from scipy.signal import medfilt
  outImg = medfilt(img, kernel_size)
  return outImg

# @brief resize an image
#
# @param img    the original image
# @param scale  the scaling factor for the image
#
# @return the resized image
def resize(img, scale):
  x, y = img.shape
  outImg = rescale(img, scale)
  outImg = rescale(outImg, 1/scale)
  outImg = outImg[:x, :y]
  return outImg

# @brief compress an image using JPEG
#
# @param img    the original image
# @param scale  the quality factor
#
# @return the compressed image
def jpeg_compression(img, QF):
  cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
  attacked = cv2.imread('tmp.jpg', 0)
  os.remove('tmp.jpg')

  return attacked
random.seed(3)
def random_attack(img):
  i = random.randint(1,6)
  if i==1:
    attacked = awgn(img, 5.0, 123)
  elif i==2:
    attacked = blur(img, [3, 2])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 5])
  elif i==5:
    attacked = resize(img, 0.5)
  elif i==6:
    attacked = jpeg_compression(img, 75)
  return attacked

def main():
    try:
        scores = []
        labels = []
        watermark=np.load('./Utilities/panebbianco.npy')
        for i in range(numTestImages):
            print(f'\r{"{:.2f}".format(100*i/numTestImages)} %', end='')
            img = cv2.imread('./Images/101_Images/'+ str(i).zfill(4) +'.bmp', 0)
            fakemark = np.random.uniform(0.0, 1.0, 1024)
            fakemark = np.uint8(np.rint(fakemark))
            watermarkedImg = embeddedFinalMethod(img, watermark)
            attackedImg = random_attack(watermarkedImg)
            watExtracted = extractWatermark(img, attackedImg)
            scores.append(similarity(watermark, watExtracted))
            labels.append(1)
            scores.append(similarity(fakemark, watExtracted))
            labels.append(0)

        print('\r100.00 %', end='')
        print('\n', end='')

        print('---')
        print('-v- RESULTS -v-')
        fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(10,10))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC_' + methodName + '_' + str(numTestImages))
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig('./ROCs/' + 'ROC_' + methodName + '_' + str(numTestImages) + '.png')
        idx_tpr = np.where((fpr-0.1)==min(i for i in (fpr-0.1) if i > 0))
        print('For a FPR approximately equals to 0.1 corresponds a TPR equals to %0.2f' % tpr[idx_tpr[0][0]])
        print('For a FPR approximately equals to 0.1 corresponds a threshold equals to %0.2f' % tau[idx_tpr[0][0]])
        print('Check FPR %0.2f' % fpr[idx_tpr[0][0]])
    except ValueError as e:
        print("\x1b[6;31mError: " + str(e) + "\x1b[0m")


if __name__ == "__main__":
    main()
