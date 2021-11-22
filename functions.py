#!/usr/bin/env python3

import os
import math
import cv2
import random
import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import rescale
from scipy.signal import convolve2d
from scipy.fft import dct, idct
import mpmath as mp
from os import listdir
from os.path import isfile, join

# --- PROFESSOR FUNCTIONS ---

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

# @brief compute the WPSNR
#
# @param img1   the original image
# @param img2   the modified image
#
# @return WPSNR in decibels
def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0

  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 9999999
  csf = np.genfromtxt('../Utilities/csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(csf,2), mode='valid')
  decibels = 20.0*np.log10(1.0/math.sqrt(np.mean(np.mean(ew**2))))
  return decibels

 # @brief Compute the similitude of two watermarks
 #
 # @param extracted          the extracted watermark
 # @param watermark          the original watermark
 #
 # @return the similitude of the watermark extracted with the one provided
def similarity(extracted,watermark):
    if(len(extracted) != len(watermark)):
        raise ValueError('The two watermarks need to be of the same size.')

    sim = abs(np.sum(np.multiply(extracted, watermark)) / np.sqrt(np.sum(np.multiply(watermark, watermark))))
    # sim = np.inner(extracted,watermark) / sqrt(np.inner(watermark,watermark))
    return sim

 # @brief Attack an image with a random attack
 #
 # @param img          the original image
 #
 # @return The attacked image
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


# --- OUR FUNCTIONS ---

# @brief Obtaint the indexes of the highest values in a monodimensional array
#
# @param a      the vector to analyze
# @param k      the number of largest values required
#
# @return the indexes of the k largest values
def k_largest_index_argsort(a, k):
    idx = np.argsort(a.ravel())[:-k-1:-1]
    # return np.column_stack(np.unravel_index(idx, a.shape))
    return idx

# @brief matrix APDCBT
#
# @param dim            the dimension of the Bmatrix
#
# @return the Bmatrix
def apdcbtMatrix(dim):
  M = N = dim
  Bmatrix = np.zeros((M, N))

  for m in range(0, M):
    for n in range(0, N):
      if n == 0:
        Bmatrix[m, n] = (N-m)/(N*N)
      else:
        Bmatrix[m, n] = ((N-m)*math.cos((m*n*math.pi)/N) - mp.csc((n*math.pi)/N)*math.sin((m*n*math.pi)/N))/(N*N)
  return Bmatrix

# @brief APDCBT
#
# @param img            the original image
#
# @return the matrix of the coefficients
def apdcbt(img, Bmatrix):
  imgShape = img.shape
  if(imgShape[0] != imgShape[1]):
      raise ValueError('A sqare image is necessary')

  img = np.double(img)
  coeff = np.zeros((imgShape))
  # temp = np.zeros((imgShape))

  coeff = np.matmul(np.matmul(Bmatrix, img), np.atleast_2d(Bmatrix).T.conj())

  return coeff

# @brief inverse APDCBT
#
# @param coeff            the matrix of the coefficients
#
# @return the imgage
def iapdcbt(coeff, Bmatrix):
  imgShape = coeff.shape
  if(imgShape[0] != imgShape[1]):
      raise ValueError('A sqare image is necessary')

  coeff = np.double(coeff)

  imgInv = np.linalg.inv(Bmatrix)
  outImg = np.matmul(np.matmul(imgInv, coeff), np.atleast_2d(imgInv).T.conj())

  return outImg

# @brief Spread Spectrum watermarking
#
# @param img            the original image
# @param watermark      the watermark normal distributed with mean equal to zero
# @param alpha          the constant for the multiplicative watermarking
#
# @return the computed image with the watermark
def ss_watermark(img, watermark, alpha):
    imgShape= img.shape
    if(imgShape[0] != imgShape[1]):
        raise ValueError('A sqare image is necessary to compute the ss watermarking')
    watermarkLen = len(watermark)
    if(imgShape[0]*imgShape[1] < watermarkLen):
        raise ValueError('The watermark need to be smaller than the number of the pixel in the image')

    transformMat = dct(dct(img,axis=0, norm='ortho'),axis=1, norm='ortho')
    transform = transformMat.flatten()
    transformToSort = abs(transform.copy())
    transformToSort[0] = 0
    indexes = k_largest_index_argsort(transformToSort , watermarkLen)

    transformWatermarked = transform.copy()
    for i in range(0, watermarkLen):
        transformWatermarked[indexes[i]] += transform[indexes[i]] * watermark[i] * alpha

    transformWatermarkedMat = transformWatermarked.reshape(imgShape[0], imgShape[1])
    imgWatermarked = idct(idct(transformWatermarkedMat,axis=0, norm='ortho'),axis=1, norm='ortho')
    return imgWatermarked

# @brief Compute the similitude of the Spread Spectrum watermarking
#
# @param originalImg        the original image
# @param watermarkedImg     the original image
# @param watermark          the watermark to check
# @param alpha              the constant for the multiplicative watermarking
#
# @return the watermark extracted
def ss_simWatermark(originalImg, watermarkedImg, alpha):
    watermarkLen = 1024
    originalImgShape = originalImg.shape
    if(originalImgShape != watermarkedImg.shape):
        raise ValueError('The two images need to have the same shape')

    #if(originalImg == watermarkedImg):
        #return 0
    transformMat = dct(dct(originalImg,axis=0, norm='ortho'),axis=1, norm='ortho')
    transform = transformMat.flatten()
    transformToSort = abs(transform.copy())
    transformToSort[0] = 0
    indexes = k_largest_index_argsort(transformToSort , watermarkLen)
    transformWatMat = dct(dct(watermarkedImg,axis=0, norm='ortho'),axis=1, norm='ortho')
    transformWatermarked = transformWatMat.flatten()
    watermarkExt = np.empty((watermarkLen))
    for i in range(0, watermarkLen):
        if(transform[indexes[i]] == 0) :
            watermarkExt[i] = 0
        else:
            watermarkExt[i] = (transformWatermarked[indexes[i]] - transform[indexes[i]]) / (transform[indexes[i]] * alpha)
    #sim = (np.inner(watermark, watermarkExt) / np.sqrt(np.inner(watermarkExt, watermarkExt)))
    return watermarkExt #sim


# @brief Calculate the N zigzag index of a 2D square matrix
#
# @param coord      the coordinates, array with two values
# @param dim        the side dimension of the matrix
#
# @return the zigzag index
def zigZagCoordToN(coord, dim):
    if(coord[0] > dim or coord[1] > dim):
        raise ValueError('Coord out of dim')

    reverse = False
    diag = coord[0] + coord[1]
    if(diag >= dim):
        reverse = True
        coord[0] = dim - coord[0] - 1
        coord[1] = dim - coord[1] - 1
        diag = coord[0] + coord[1]
    k = int(diag*(diag+1)/2)
    returnThis = k + coord[diag % 2]
    if(reverse):
        returnThis = dim*dim -returnThis - 1
    return returnThis

# @brief Calculate the zigzag coordinates of a 2D square matrix
#
# @param N          the zigzag index
# @param dim        the side dimension of the matrix
#
# @return the coordinates (array with two values)
def nToZigZagCoord(N, dim):
    if(N > dim*dim):
        raise ValueError('N out of dim')

    reverse = False
    if(N > dim*dim/2):
        reverse = True
        N=(dim*dim)-N-1
    diag = math.floor((math.sqrt(1+8*N)+1)/2)
    k = int(N - (diag-1)*diag/2)
    returnThis = [0, 0]
    returnThis[diag % 2] = diag-k-1
    returnThis[1 - diag % 2] = k
    if(reverse):
        returnThis[0] = dim - returnThis[0] - 1
        returnThis[1] = dim - returnThis[1] - 1
    return returnThis

# @brief convert a matrix to matrix read in Zig Zag mode
#
# @param originalImg          Matrix to convert
#
# @return Same matrix but read in zig zag (Scorrelated)
def convertToZigZagMatrix(originalImg):
    newImg = np.zeros(originalImg.shape)
    newImg = newImg.flatten()
    for i in range(originalImg.shape[0]*originalImg.shape[1]):
        coords = nToZigZagCoord(i,originalImg.shape[0])
        newImg[i] = originalImg[coords[0],coords[1]]

    newImg = newImg.reshape(originalImg.shape[0],originalImg.shape[1])
    return newImg

# @brief convert a matrix to matrix read in Normal mode (if before was converted to ZigZag matrix)
#
# @param originalImg          Matrix to convert
#
# @return Same matrix but read in XY mode
def convertToXYMatrix(originalImg):
    newImg = np.zeros(originalImg.shape)
    originalImg = originalImg.flatten()
    for i in range(len(originalImg)):
        coords = nToZigZagCoord(i,newImg.shape[0])
        newImg[coords[0],coords[1]] = originalImg[i]

    return newImg

# @brief Calculate the variance of an image
#
# @param img        the image
#
# @return the variance of the provided image
def variance(img):
    return ndimage.variance(img)

# @brief Get the coordinates of the sqare with the most variance
#
# @param img        the image to check
# @param div        the number of divisions for each side
#
# @return the coordinates of the sqare with the most variance (array with two values)
def getMostVarImages(img, div):
    imgShape = img.shape
    if(imgShape[0] != imgShape[1]):
        raise ValueError('A sqare image is necessary to get the mostVar')

    minDim = int(imgShape[0]/div)
    numImages = div**2
    variances = np.empty((numImages))
    for i in range(0, div):
        for j in range(0, div):
            variances[i+(j*div)] = variance(img[i*minDim:(i+1)*minDim,j*minDim:(j+1)*minDim])
    indexes = k_largest_index_argsort(variances, numImages)
    coords = np.empty((numImages, 2))
    for i in range(0, numImages):
        coords[i, 0]= int(indexes[i]%div)
        coords[i, 1]= int(indexes[i]/div)
    return coords

# @brief Spread Spectrum watermarking only on the sqares with most variance
#
# @param img            the original image
# @param watermark      the watermark normal distributed with mean equal to zero
# @param alpha          the constant for the multiplicative watermarking
# @param div            the number of divisions for each side
# @param numSide        the root sqare of the number of squares in which the watermark is embedded
#
# @return the computed image with the watermark
def ss_watermarkHF(img, watermark, alpha, div, numSide):
    imgShape = img.shape
    minDim = int(imgShape[0]/div)
    coords = getMostVarImages(img, div)
    imgToWatermark = np.zeros((minDim*numSide, minDim*numSide))
    offset = 0
    for i in range(0, (numSide**2)):
        x = int(i/numSide)
        y = int(i%numSide)
        xMin = int(x*minDim)
        xMax = int((x+1)*minDim)
        yMin = int(y*minDim)
        yMax = int((y+1)*minDim)
        xF = coords[i + offset, 0]
        yF = coords[i + offset, 1]
        xFMin = int(xF*minDim)
        xFMax = int((xF+1)*minDim)
        yFMin = int(yF*minDim)
        yFMax = int((yF+1)*minDim)
        imgToWatermark[xMin : xMax, yMin : yMax] = img[xFMin : xFMax, yFMin : yFMax]
    watermarkedImgHF = ss_watermark(imgToWatermark, watermark, alpha)
    imgWatermarkedHF1 = img.copy() #np.zeros((imgShape[0], imgShape[0])) #img.copy()
    for i in range(0, (numSide**2)):
        x = int(i/numSide)
        y = int(i%numSide)
        xMin = int(x*minDim)
        xMax = int((x+1)*minDim)
        yMin = int(y*minDim)
        yMax = int((y+1)*minDim)
        xF = coords[i + offset, 0]
        yF = coords[i + offset, 1]
        xFMin = int(xF*minDim)
        xFMax = int((xF+1)*minDim)
        yFMin = int(yF*minDim)
        yFMax = int((yF+1)*minDim)
        imgWatermarkedHF1[xFMin : xFMax, yFMin : yFMax] = 0

    imgWatermarkedHF2 = np.zeros((imgShape[0], imgShape[0])) #img.copy()
    for i in range(0, (numSide**2)):
        x = int(i/numSide)
        y = int(i%numSide)
        xMin = int(x*minDim)
        xMax = int((x+1)*minDim)
        yMin = int(y*minDim)
        yMax = int((y+1)*minDim)
        xF = coords[i + offset, 0]
        yF = coords[i + offset, 1]
        xFMin = int(xF*minDim)
        xFMax = int((xF+1)*minDim)
        yFMin = int(yF*minDim)
        yFMax = int((yF+1)*minDim)
        imgWatermarkedHF2[xFMin : xFMax, yFMin : yFMax] = watermarkedImgHF[xMin : xMax, yMin : yMax]

    imgWatermarkedHF = imgWatermarkedHF1 + imgWatermarkedHF2

    imgWatermarkedScramble = np.zeros((minDim*numSide, minDim*numSide))
    for i in range(0, (numSide**2)):
        x = int(i/numSide)
        y = int(i%numSide)
        xMin = int(x*minDim)
        xMax = int((x+1)*minDim)
        yMin = int(y*minDim)
        yMax = int((y+1)*minDim)
        xF = coords[i + offset, 0]
        yF = coords[i + offset, 1]
        xFMin = int(xF*minDim)
        xFMax = int((xF+1)*minDim)
        yFMin = int(yF*minDim)
        yFMax = int((yF+1)*minDim)
        imgWatermarkedScramble[xMin : xMax, yMin : yMax] = imgWatermarkedHF[xFMin : xFMax, yFMin : yFMax]
    return imgWatermarkedHF

# @brief Compute the similitude of the Spread Spectrum watermarking only on the sqares with most variance
#
# @param originalImg        the original image
# @param watermarkedImg     the original image
# @param watermark          the watermark to check
# @param alpha              the constant for the multiplicative watermarking
# @param div                the number of divisions for each side
# @param numSide            the root sqare of the number of squares in which the watermark is embedded
#
# @return the watermark extracted
def ss_simWatermarkHF(originalImg, watermarkedImg, alpha, div, numSide):
    imgShape = originalImg.shape
    minDim = int(imgShape[0]/div)
    coords = getMostVarImages(originalImg, div)
    imgOriginalScramble = np.zeros((minDim*numSide, minDim*numSide))
    imgWatermarkedScramble = np.zeros((minDim*numSide, minDim*numSide))
    offset = 0
    for i in range(0, (numSide**2)):
        x = int(i/numSide)
        y = int(i%numSide)
        xF = coords[i + offset, 0]
        yF = coords[i + offset, 1]
        imgOriginalScramble[int(x*minDim) : int((x+1)*minDim), int(y*minDim) : int((y+1)*minDim)] = originalImg[int(xF*minDim) : int((xF+1)*minDim), int(yF*minDim) : int((yF+1)*minDim)]
        imgWatermarkedScramble[int(x*minDim) : int((x+1)*minDim), int(y*minDim) : int((y+1)*minDim)] = watermarkedImg[int(xF*minDim) : int((xF+1)*minDim), int(yF*minDim) : int((yF+1)*minDim)]
    extractedWat = ss_simWatermark(imgOriginalScramble, imgWatermarkedScramble, alpha)
    return extractedWat

# @brief Show the squares with more variance
#
# The image is divided in sqares which are sorted considering their variance.
# An RGB image in which the square with more variance are more red is returned.
# The intensity of the red is proportional to the position of the block in the
# previously sorted array of sqares.
#
# @param img        the original image
# @param div        the number of divisions for each side
#
# @return an RGB image in which the square with more variance are more red
def showImgVarianceSort(img, div):
    imgShape = img.shape
    coords = getMostVarImages(img, div)
    minDim = int(imgShape[0]/div)
    numImages = div**2
    outImg = np.empty((imgShape[0], imgShape[0], 3), dtype=np.uint8)
    outImg[:, :, 0] = img
    for i in range(0, numImages):
        mulFactor =  (i/(numImages))
        minX = int(coords[i, 0]*minDim)
        maxX = int((coords[i, 0]+1)*minDim)
        minY = int(coords[i, 1]*minDim)
        maxY = int((coords[i, 1]+1)*minDim)
        outImg[minX:maxX, minY:maxY, 2] = outImg[minX:maxX, minY:maxY, 1] = (img[minX:maxX, minY:maxY]*mulFactor)
    return outImg

# @brief Show the squares with more variance
#
# The image is divided in sqares. An RGB image in which the square with
# more variance are more red is returned.
# The intensity of the red is proportional to the variance of the block.
#
# @param img        the original image
# @param div        the number of divisions for each side
#
# @return an RGB image in which the square with more variance are more red
def showImgVariance(img, div):
    imgShape = img.shape
    if(imgShape[0] != imgShape[1]):
        raise ValueError('A sqare image is necessary to get the mostVar')

    outImg = np.empty((imgShape[0], imgShape[0], 3), dtype=np.uint8)
    outImg[:, :, 0] = img
    minDim = int(imgShape[0]/div)
    variances = np.empty((div,div))
    for i in range(0, div):
        for j in range(0, div):
            variances[i, j] = variance(img[i*minDim:(i+1)*minDim,j*minDim:(j+1)*minDim])
    maxVariance = np.amax(np.amax(variances))
    for i in range(0, div):
        for j in range(0, div):
            mulFactor = 1 - (variances[i, j]/maxVariance)
            minX = i*minDim
            maxX = (i+1)*minDim
            minY = j*minDim
            maxY = (j+1)*minDim
            outImg[minX:maxX, minY:maxY, 2] = outImg[minX:maxX, minY:maxY, 1] = (img[minX:maxX, minY:maxY]*mulFactor)
    return outImg



# @brief Destry image watermarking
#
# @param img            the original image
# @param watermark      the watermark normal distributed with mean equal to zero
#
# @return the computed image with the watermark
def di_watermark(img, watermark):
    imgShape= img.shape
    if(imgShape[0] != imgShape[1]):
        raise ValueError('A sqare image is necessary to compute the ss watermarking')
    watermarkLen = len(watermark)
    if(imgShape[0]*imgShape[1] < watermarkLen):
        raise ValueError('The watermark need to be smaller than the number of the pixel in the image')

    transformMat = dct(dct(img,axis=0, norm='ortho'),axis=1, norm='ortho')
    transform = transformMat.flatten()
    transformToSort = abs(transform.copy())
    indexes = k_largest_index_argsort(transformToSort , watermarkLen)

    transformWatermarked = transform.copy()
    for i in range(0, watermarkLen):
        transformWatermarked[indexes[i]] = watermark[i]

    transformWatermarkedMat = transformWatermarked.reshape(imgShape[0], imgShape[1])
    imgWatermarked = idct(idct(transformWatermarkedMat,axis=0, norm='ortho'),axis=1, norm='ortho')
    return imgWatermarked

# @brief Extract watermark of destroy image watermarking
#
# @param originalImg        the original image
# @param watermarkedImg     the original image
# @param watermark          the watermark to check
#
# @return the watermark extracted
def di_getWatermark(originalImg, watermarkedImg):
    originalImgShape = originalImg.shape
    watermarkLen = 1024
    if(originalImgShape != watermarkedImg.shape):
        raise ValueError('The two images need to have the same shape')

    if(originalImgShape[0]*originalImgShape[1] < watermarkLen):
        raise ValueError('The watermark need to be smaller than the pixel of the image')
    #if(originalImg == watermarkedImg):
        #return 0
    transformMat = dct(dct(originalImg,axis=0, norm='ortho'),axis=1, norm='ortho')
    transform = transformMat.flatten()
    transformToSort = abs(transform.copy())
    indexes = k_largest_index_argsort(transformToSort , watermarkLen)
    transformWatMat = dct(dct(watermarkedImg,axis=0, norm='ortho'),axis=1, norm='ortho')
    transformWatermarked = transformWatMat.flatten()
    watermarkExt = np.empty((watermarkLen))
    for i in range(0, watermarkLen):
        watermarkExt[i] = transformWatermarked[indexes[i]]
    return watermarkExt

############### DWT DCT SVD WATERMARK ##################
# Parameters
blockSizeDwtDctSvd = 8
alphaDwtDctSvd = 2.0

# @param dctMatrix      matrix of 32x32 values (((512 / 2) / 8) ; same)
# @param watermarkRow   32 values of watermark to insert
# @return               matrix with shaoe of 1/8 of fromDwt
def svdInsert(dctMatrix,watermarkRow):
    # apply SVD to the result matrix. The only matrix that we need is "s" (according to dctMatrix = USV^(T))
    u,s,v=np.linalg.svd(dctMatrix)
    s_mod = s.copy()
    for i in range(s.size):
    #     if ( i > 0 and (s[i-1] - s[i]) < alphaDwtDctSvd*1 ) :
    #         if ( s[i] - alphaDwtDctSvd*1 > 0 ) :
    #             s[i-1] = s[i-1] + alphaDwtDctSvd*1
    #             s[i] = s[i] - alphaDwtDctSvd*1
    #         else:
    #             s[i-1] = s[i-1] + alphaDwtDctSvd*1
    #     if ( s[i] < alphaDwtDctSvd*1 and watermarkRow[i] < 0 ) :
    #         watermarkRow[i] = 0
        s_mod[i] = s[i] + watermarkRow[i]*alphaDwtDctSvd
    # Inverse SVD. Compute the diagonal matrix of s to compute the operation
    watermarkedMatrix = np.matmul(u,np.matmul(np.diag(s_mod),v))

    # u,s_1,v=np.linalg.svd(watermarkedMatrix)

    return watermarkedMatrix

# @input fromDwt can represent the LL, LH, HL, HH components from DWT
# @return 32 matrixes with shape of 1/8 of fromDwt
def dctCoeffMatrixes(fromDwt):
    if(fromDwt.shape[0] != fromDwt.shape[1]):
        raise ValueError('fromDwt input is not a square matrix')

    numBlockSide = int(fromDwt.shape[0]/blockSizeDwtDctSvd)
    dctMatrixSortWatermarked = np.empty((32, numBlockSide, numBlockSide))

    for i in range(numBlockSide):
        for j in range(numBlockSide):
            minX = int(i * blockSizeDwtDctSvd)
            maxX = int((i+1) * blockSizeDwtDctSvd)
            minY = int(j * blockSizeDwtDctSvd)
            maxY = int((j+1) * blockSizeDwtDctSvd)
            dctMatrix = dct(dct(fromDwt[minX:maxX, minY:maxY],axis=0, norm='ortho'),axis=1, norm='ortho')
            dctMatrixFlat = dctMatrix.flatten()
            maxDctCoeff = k_largest_index_argsort(abs(dctMatrixFlat), 32)
            for k in range(32):
                dctMatrixSortWatermarked[k,i,j] = dctMatrixFlat[maxDctCoeff[k]]

    return dctMatrixSortWatermarked

# @return inverse of dctCoeffMatrixes, return HL/LL/.. of watermarked image
def idctCoeffMatrixes(originalFromDwt, watermarkedMatrixes):
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
            dctMatrixFlat = dctMatrix.flatten()
            maxDctCoeff = k_largest_index_argsort(abs(dctMatrixFlat), 32)
            for k in range(32):
                dctMatrixFlat[maxDctCoeff[k]] = watermarkedMatrixes[k,i,j]
            dctMatrix = dctMatrixFlat.reshape(blockSizeDwtDctSvd,blockSizeDwtDctSvd)
            toInvDwt[minX:maxX, minY:maxY] = idct(idct(dctMatrix,axis=0, norm='ortho'),axis=1, norm='ortho')

    return toInvDwt

# @param originalImg        Original image with shape of (512,512)
# @param watermark          Watermark with shape of (32,32)
# @return watermarkImage    Waterarked Image
def embeddedWatermarkDwtDctSvd(originalImg, watermark):
    watermark = watermark.reshape(32,32)
    # if (originalImg.shape != (512,512)) or (watermark.shape != (32,32)) :
    #     raise ValueError("Size of images aren't standard.")

    # Compute dwt of original image
    coeff = pywt.dwt2(originalImg, 'haar')   # Old: originalImg
    LL, (LH, HL, HH) = coeff

    ## LL
    # Find 32 matrixes of higher components of LH component
    dctMatrixes = dctCoeffMatrixes(LL)

    # insert on every coefficents matrix a line of watermark
    for i in range(len(dctMatrixes)):
        dctMatrixes[i] = svdInsert(dctMatrixes[i],watermark[i,:8]) # watermark[i]

    # Find the modified LL component
    LL = idctCoeffMatrixes(LL, dctMatrixes)
    ##
    ## HL
    # Find 32 matrixes of higher components of LH component
    dctMatrixes = dctCoeffMatrixes(HL)

    # insert on every coefficents matrix a line of watermark
    for i in range(len(dctMatrixes)):
        dctMatrixes[i] = svdInsert(dctMatrixes[i],watermark[i,8:16])


    # Find the modified HL component
    HL = idctCoeffMatrixes(HL, dctMatrixes)
    ##
    ## LH
    # Find 32 matrixes of higher components of LH component
    dctMatrixes = dctCoeffMatrixes(LH)

    # insert on every coefficents matrix a line of watermark
    for i in range(len(dctMatrixes)):
        dctMatrixes[i] = svdInsert(dctMatrixes[i],watermark[i,16:24])

    # Find the modified LH component
    LH = idctCoeffMatrixes(LH, dctMatrixes)
    ##
    ## HH
    # Find 32 matrixes of higher components of HH component
    dctMatrixes = dctCoeffMatrixes(HH)

    # insert on every coefficents matrix a line of watermark
    for i in range(len(dctMatrixes)):
        dctMatrixes[i] = svdInsert(dctMatrixes[i],watermark[i,24:32])

    # Find the modified HH component
    HH = idctCoeffMatrixes(HH, dctMatrixes)
    ##

    # compute inverse of idw2 with watermarked components
    watermarkedImage=pywt.idwt2((LL,(LH,HL,HH)),'haar')

    return watermarkedImage

def extractWatermarkDWT(dwt_original,dwt_watermarked):
    watermarkToReturn = np.zeros((32,8)) #np.zeros((32,32))

    # Find the size of matrix of block blockSizeDwtDctSvdxblockSizeDwtDctSvd
    numBlockSide = int(dwt_original.shape[0]/blockSizeDwtDctSvd)
    dctMatrixSortWatermarked = np.empty((32, numBlockSide, numBlockSide))
    dctMatrixSortOriginal = np.empty((32, numBlockSide, numBlockSide))

    for i in range(numBlockSide):
        for j in range(numBlockSide):
            minX = int(i * blockSizeDwtDctSvd)
            maxX = int((i+1) * blockSizeDwtDctSvd)
            minY = int(j * blockSizeDwtDctSvd)
            maxY = int((j+1) * blockSizeDwtDctSvd)
            dctMatrixFlatOriginal = dct(dct(dwt_original[minX:maxX, minY:maxY],axis=0, norm='ortho'),axis=1, norm='ortho')
            dctMatrixWatermarked = dct(dct(dwt_watermarked[minX:maxX, minY:maxY],axis=0, norm='ortho'),axis=1, norm='ortho')
            dctMatrixFlatOriginal = dctMatrixFlatOriginal.flatten()
            dctMatrixWatermarked = dctMatrixWatermarked.flatten()
            maxDctCoeff = k_largest_index_argsort(abs(dctMatrixFlatOriginal), 32)
            for k in range(32):
                dctMatrixSortWatermarked[k,i,j] = dctMatrixWatermarked[maxDctCoeff[k]]
                dctMatrixSortOriginal[k,i,j] = dctMatrixFlatOriginal[maxDctCoeff[k]]

    for i in range(len(dctMatrixSortOriginal)):
        u_original,s_original,vh_original=np.linalg.svd(dctMatrixSortOriginal[i])
        u_watermarked,s_watermarked,vh_watermarked=np.linalg.svd(dctMatrixSortWatermarked[i])
        #
        # for y in range(s_original.size):
        #     if ( y > 0 and (s_original[y-1] - s_original[y]) < alphaDwtDctSvd*1 ) :
        #         if ( s_original[y] - alphaDwtDctSvd*1 > 0 ) :
        #             s_original[y-1] = s_original[y-1] + alphaDwtDctSvd*1
        #             s_original[y] = s_original[y] - alphaDwtDctSvd*1
        #         else:
        #             s_original[y-1] = s_original[y-1] + alphaDwtDctSvd*1
        watermarkToReturn[i] = (s_watermarked - s_original) / (alphaDwtDctSvd)

    return watermarkToReturn


def extractWatermarkDwtDctSvd(original, watermarkedImage):
    # if (original.shape != (512,512)) or (watermarkedImage.shape != (512,512)) :
    #     raise("Size of images aren't standard.")

    # Do dwt transform to original image
    originalCoeff = pywt.dwt2(original, 'haar')
    LL_original, (LH_original, HL_original, HH_original) = originalCoeff

    # Do dwt transformof the watermarked image
    watermarkedCoeff = pywt.dwt2(watermarkedImage, 'haar')
    LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = watermarkedCoeff
    extractedWatermarkLL = extractWatermarkDWT(LL_original,LL_watermarked)
    extractedWatermarkLH = extractWatermarkDWT(LH_original,LH_watermarked)
    extractedWatermarkHL = extractWatermarkDWT(HL_original,HL_watermarked)
    extractedWatermarkHH = extractWatermarkDWT(HH_original,HH_watermarked)

    # returnWatermark = extractedWatermarkLL*0.45+extractedWatermarkLH*0.2+extractedWatermarkHL*0.2+extractedWatermarkHH*0.15
    returnWatermark = np.zeros((32,32))
    returnWatermark[:,:8] = extractedWatermarkLL
    returnWatermark[:,8:16] = extractedWatermarkHL
    returnWatermark[:,16:24] = extractedWatermarkLH
    returnWatermark[:,24:32] = extractedWatermarkHH
    returnWatermark = returnWatermark.round()
    returnWatermark = returnWatermark.flatten()
    return returnWatermark

############### DWT DCT SVD WATERMARK ##################

############# OUR FINAL METHOD ##################
# Parameters
blockSizeDwtDctSvd = 8

#                LL    LH   HL   HH
alphaPaper    = [30, 20, 20, 20]  # old: all 8 9 9 10
extractWeight = [0.45, 0.2, 0.2, 0.15] # old: [0.45, 0.2, 0.2, 0.15]

############# COMMON FUNCTION #############

# @input fromDwt can represent the LL, LH, HL, HH components from DWT
# @return 32 matrixes with shape of 1/8 of fromDwt
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

############# EMBEDDED FINAL METHOD #############
# Idea of 19.10 ( Paper method in all blocks of image, and SS in the HF bloks)
# @param originalImg        Original image with shape of (512,512)
# @param watermark          Watermark with shape of (32,32)
# @param ssNum              Number of SS watermark to insert into image
# @param alphaSS            Alpha of SS
# @return watermarkImage    Waterarked Image
def embeddedFinalMethod(originalImg, watermark, imageName='NULL'):
    # Insert Watermark wit our method
    watermarkedImgA = embeddedPaper(originalImg, watermark)
    watermarkedImg = np.rint(watermarkedImgA).astype(int)
    if imageName != 'NULL' :
        cv2.imwrite('../Watermarked/'+imageName+'_panebbianco.bmp', watermarkedImgA)
    return watermarkedImg


# @param originalImg        Original image with shape of (512,512)
# @param watermark          Watermark with shape of (32,32)
# @return watermarkImage    Waterarked Image
def embeddedPaper(originalImg, watermark):
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
    Mp = svdInsertPaper(dctMatrix, watermark, alphaPaper[0])

    # Find the modified LL component
    LL_star = idctCoeffMatrix(LL, Mp)

    ## LH
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(LH)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsertPaper(dctMatrix, watermark, alphaPaper[1])

    # Find the modified LH component
    LH = idctCoeffMatrix(LH, Mp)
    ##

    ## HL
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(HL)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsertPaper(dctMatrix, watermark, alphaPaper[2])

    # Find the modified HL component
    HL = idctCoeffMatrix(HL, Mp)
    ##

    ## HH
    # Find 32 matrixes of higher components of LH component
    dctMatrix = dctCoeffMatrix(HH)

    # insert on every coefficents matrix a line of watermark
    Mp = svdInsertPaper(dctMatrix, watermark, alphaPaper[3])

    # Find the modified HH component
    HH = idctCoeffMatrix(HH, Mp)
    ##

    # compute inverse of idw2 with watermarked components
    watermarkedImage=pywt.idwt2((LL_star,(LH,HL,HH)),'haar')


    return watermarkedImage

def svdInsertPaper(dctMatrix, watermark, alphaPaperLocal):
    # apply SVD to the result matrix. The only matrix that we need is "s" (according to dctMatrix = USV^(T))
    u,s,v = np.linalg.svd(dctMatrix)
    s_mod = np.diag(s) + watermark * alphaPaperLocal

    dcDwtWat = np.matmul(u,np.matmul(s_mod,v))

    return dcDwtWat

############# DETECTION FINAL METHOD #############
def detection(input1, input2, input3):
    original = cv2.imread(input1, 0)
    watermarked = cv2.imread(input2, 0)
    attacked = cv2.imread(input3, 0)

    output1 = 0
    watermark = extractWatermarkPaper(originalImg, watermarked)
    attackedWat = extractWatermarkPaper(originalImg, attacked)
    if(similarity(watermark,attackedWat) > 9.17):
      output1 = 1

    output2 = wpsnr(watermarked, attacked)
    return output1, output2

def extractWatermarkPaper(original, watermarkedImage):
    if (original.shape != (512,512)) or (watermarkedImage.shape != (512,512)) :
        rise("Size of images aren't standard.")

    # Do dwt transform to original image
    originalCoeff = pywt.dwt2(original, 'haar')
    LL_original, (LH_original, HL_original, HH_original) = originalCoeff

    # Do dwt transformof the watermarked image
    watermarkedCoeff = pywt.dwt2(watermarkedImage, 'haar')
    LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = watermarkedCoeff

    extractedWatermarkLL = extractPaperDWT(LL_original,LL_watermarked, alphaPaper[0])
    extractedWatermarkLH = extractPaperDWT(LH_original,LH_watermarked, alphaPaper[1])
    extractedWatermarkHL = extractPaperDWT(HL_original,HL_watermarked, alphaPaper[2])
    extractedWatermarkHH = extractPaperDWT(HH_original,HH_watermarked, alphaPaper[3])

    returnWatermark = extractedWatermarkLL*extractWeight[0]+extractedWatermarkLH*extractWeight[1]+extractedWatermarkHL*extractWeight[2]+extractedWatermarkHH*extractWeight[3]
    # returnWatermark = returnWatermark.round()
    returnWatermark = returnWatermark.flatten()
    return returnWatermark

def extractPaperDWT(dwt_original,dwt_watermarked, alphaPaperLocal):
    dctMatrixWatermarked = dctCoeffMatrix(dwt_watermarked)
    dctMatrixOriginal = dctCoeffMatrix(dwt_original)

    u, s, v = np.linalg.svd(dctMatrixOriginal)
    s_mod_star = np.matmul(np.matmul(u.transpose(),dctMatrixWatermarked),v.transpose())

    watermarkToReturn = (s_mod_star - np.diag(s)) / (alphaPaperLocal)

    return watermarkToReturn

#################### UTILITIES FOR BATTLE ###########################
# Return list with all files in a path
# Example:
# ImageToAttack = imageOnFolder('../imagesToAttack/')
#
def imageOnFolder(mypath) :
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return onlyfiles

# @return attacked image
# Example:
#   f.attackOnlyInSpace(toAttackImg,xMin,xMax,yMin,yMax,functionToUse,ParametersToFntToUse)
def attackOnlyInSpace(img,xMin,xMax,yMin,yMax,attackFunction,parameters) :
    attacked = img.copy()
    # attacked[xMin:xMax,yMin:yMax] = np.zeros((xMax-xMin,yMax-yMin)) #attackFunction(attacked[xMin:xMax,yMin:yMax],parameters)
    attacked[xMin:xMax,yMin:yMax] = attackFunction(attacked[xMin:xMax,yMin:yMax],parameters)
    return attacked


#################### UTILITIES FOR BATTLE ###########################
