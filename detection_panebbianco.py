import cv2
from wpsnr import wpsnr
from scipy.fft import dct, idct
import numpy as np
import pywt

#                LL    LH   HL   HH
alphaPaper    = [8.3, 10,  10,  12] # 30 20 20 20
extractWeight = [0.45, 0.2, 0.2, 0.15]
THRESHOLD = 10.91 # 9.91
blockSizeDwtDctSvd = 8

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    # s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    s = abs(np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def dctCoeffMatrix(fromDwt):
    if(fromDwt.shape[0] != fromDwt.shape[1]):
        raise ValueError('fromDwt input is not a square matrix')

    numBlockSide = int(fromDwt.shape[0]/blockSizeDwtDctSvd)
    dctMatrix = np.empty((numBlockSide, numBlockSide))
    dctMatrixSortWatermarked = np.empty((numBlockSide, numBlockSide))
    for i in range(numBlockSide):
        for j in range(numBlockSide):
            minX = int(i * blockSizeDwtDctSvd)
            maxX = int((i+1) * blockSizeDwtDctSvd)
            minY = int(j * blockSizeDwtDctSvd)
            maxY = int((j+1) * blockSizeDwtDctSvd)
            dctMatrix = dct(dct(fromDwt[minX:maxX, minY:maxY],axis=0, norm='ortho'),axis=1, norm='ortho')
            dctMatrixSortWatermarked[i,j] = dctMatrix[0, 0]

    return dctMatrixSortWatermarked

def extractDWT(dwt_original,dwt_watermarked, alphaPaperLocal):
    dctMatrixWatermarked = dctCoeffMatrix(dwt_watermarked)
    dctMatrixOriginal = dctCoeffMatrix(dwt_original)

    u, s, v = np.linalg.svd(dctMatrixOriginal)
    s_mod_star = np.matmul(np.matmul(u.transpose(),dctMatrixWatermarked),v.transpose())

    watermarkToReturn = (s_mod_star - np.diag(s)) / (alphaPaperLocal)

    return watermarkToReturn

def extractWatermark(original, watermarkedImage):
    if (original.shape != (512,512)) or (watermarkedImage.shape != (512,512)) :
        raise ValueError("Size of images aren't standard.")

    # Do dwt transform to original image
    originalCoeff = pywt.dwt2(original, 'haar')
    LL_original, (LH_original, HL_original, HH_original) = originalCoeff

    # Do dwt transformof the watermarked image
    watermarkedCoeff = pywt.dwt2(watermarkedImage, 'haar')
    LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = watermarkedCoeff

    extractedWatermarkLL = extractDWT(LL_original,LL_watermarked, alphaPaper[0])
    extractedWatermarkLH = extractDWT(LH_original,LH_watermarked, alphaPaper[1])
    extractedWatermarkHL = extractDWT(HL_original,HL_watermarked, alphaPaper[2])
    extractedWatermarkHH = extractDWT(HH_original,HH_watermarked, alphaPaper[3])

    returnWatermark = extractedWatermarkLL*extractWeight[0]+extractedWatermarkLH*extractWeight[1]+extractedWatermarkHL*extractWeight[2]+extractedWatermarkHH*extractWeight[3]
    # returnWatermark = returnWatermark.round()
    returnWatermark = returnWatermark.flatten()
    return returnWatermark


def detection(input1, input2, input3):
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    watermark = extractWatermark(original, watermarked)
    attackedWat = extractWatermark(original, attacked)
    output1 = 0
    if(similarity(watermark, attackedWat) > THRESHOLD):
      output1 = 1

    output2 = wpsnr(watermarked, attacked)
    return output1, output2
