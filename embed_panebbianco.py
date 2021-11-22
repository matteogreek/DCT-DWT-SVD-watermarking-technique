#!/usr/bin/env python3

import cv2
import pywt
from scipy.fft import dct, idct
import numpy as np
import functions as f

#                LL    LH   HL   HH
alphaPaper    = [13.7,   8,  8,  8] # 30 20 20 20
extractWeight = [0.45, 0.2, 0.2, 0.15]
blockSizeDwtDctSvd = 8

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

# @return inverse of dctCoeffMatrixes, return HL/LL/.. of watermarked image
def idctCoeffMatrix(originalFromDwt, watermarkedMatrix):
    if(originalFromDwt.shape[0] != originalFromDwt.shape[1]):
        raise ValueError('fromDwt input is not a square matrix')

    numBlockSide = int(originalFromDwt.shape[0]/blockSizeDwtDctSvd)
    toInvDwt = np.empty((originalFromDwt.shape[0], originalFromDwt.shape[1]))

    for i in range(numBlockSide):
        for j in range(numBlockSide):
            minX = int(i * blockSizeDwtDctSvd)
            maxX = int((i+1) * blockSizeDwtDctSvd)
            minY = int(j * blockSizeDwtDctSvd)
            maxY = int((j+1) * blockSizeDwtDctSvd)
            dctMatrix = dct(dct(originalFromDwt[minX:maxX, minY:maxY],axis=0, norm='ortho'),axis=1, norm='ortho')
            dctMatrix[0, 0] = watermarkedMatrix[i,j]
            toInvDwt[minX:maxX, minY:maxY] = idct(idct(dctMatrix,axis=0, norm='ortho'),axis=1, norm='ortho')

    return toInvDwt

def svdInsert(dctMatrix, watermark, alphaPaperLocal):
    # apply SVD to the result matrix. The only matrix that we need is "s" (according to dctMatrix = USV^(T))
    u,s,v = np.linalg.svd(dctMatrix)
    s_mod = np.diag(s) + watermark * alphaPaperLocal

    dcDwtWat = np.matmul(u,np.matmul(s_mod,v))

    return dcDwtWat

# @param originalImg        Original image with shape of (512,512)
# @param watermark          Watermark with shape of (32,32)
# @return watermarkImage    Waterarked Image
def embedded(originalImg, watermark):
    watermark = watermark.reshape(32,32)
    if (originalImg.shape != (512,512)) or (watermark.shape != (32,32)) :
        raise ValueError("Size of images aren't standard.")

    # Compute dwt of original image
    coeff = pywt.dwt2(originalImg, 'haar')   # Old: originalImg
    LL, (LH, HL, HH) = coeff

    ## LL
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(LL)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsert(dctMatrix, watermark, alphaPaper[0])

    # Find the modified LL component
    LL_star = idctCoeffMatrix(LL, Mp)

    ## LH
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(LH)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsert(dctMatrix, watermark, alphaPaper[1])

    # Find the modified LH component
    LH = idctCoeffMatrix(LH, Mp)
    ##

    ## HL
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(HL)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsert(dctMatrix, watermark, alphaPaper[2])

    # Find the modified HL component
    HL = idctCoeffMatrix(HL, Mp)
    ##

    ## HH
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(HH)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsert(dctMatrix, watermark, alphaPaper[3])

    # Find the modified HH component
    HH = idctCoeffMatrix(HH, Mp)
    ##

    # compute inverse of idw2 with watermarked components
    watermarkedImage=pywt.idwt2((LL_star,(LH,HL,HH)),'haar')


    return watermarkedImage


# @param originalImg        Original image with shape of (512,512)
# @param watermark          Watermark with shape of (32,32)
# @param ssNum              Number of SS watermark to insert into image
# @param alphaSS            Alpha of SS
# @return watermarkImage    Waterarked Image
def embeddedFinalMethod(originalImg, watermark):
    watermarkedImgA = embedded(originalImg, watermark)
    cv2.imwrite('../Images/temp.bmp', watermarkedImgA)
    watermarkedImg = cv2.imread('../Images/temp.bmp', 0)
    return watermarkedImg


def main():
    try:
        imagesToWatermark = f.imageOnFolder('../originalImages/')
        watermark=np.load('../Utilities/panebbianco.npy')
        for imagePath in imagesToWatermark:
            imageName = imagePath.split('.')[0]
            imagePath = "../originalImages/" + imagePath
            originalImg = cv2.imread(imagePath, 0)
            watermarkedImg = embedded(originalImg, watermark)
            cv2.imwrite(f'../watermarkedImages/{imageName}_panebbianco.bmp', watermarkedImg)
            cv2.imwrite(f'../imagesToAttack/panebbianco_{imageName}.bmp', watermarkedImg)
            print(f'WPSNR {imageName}: {f.wpsnr(originalImg, watermarkedImg)} dB')
    except ValueError as e:
        print("\x1b[6;31mError: " + str(e) + "\x1b[0m")

if __name__ == "__main__":
    main()
