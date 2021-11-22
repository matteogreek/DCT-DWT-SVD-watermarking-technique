import importlib
import os
import random
#from detection_panebbianco import *
from embed_panebbianco import *
import functions as f
from cv2 import resize

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
########################## attacks ######################################
def awgn(img, std, seed=123):
    mean = 0.0  # some constant
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def blur(img, sigma):
    from scipy.ndimage.filters import gaussian_filter
    attacked = gaussian_filter(img, sigma)
    return attacked

def sharpening(img, sigma, alpha=1):
    import scipy
    from scipy.ndimage import gaussian_filter
    import matplotlib.pyplot as plt

    # print(img/255)
    filter_blurred_f = gaussian_filter(img, sigma)

    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    from scipy.signal import medfilt
    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  cv2.imwrite('tmp.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), QF])
  attacked = cv2.imread('tmp.jpg', 0)
  os.remove('tmp.jpg')
  return attacked

def random_attack(img):
    i = random.randint(1, 6)
    if i == 1:
        attacked = awgn(img, 5.0, 123)
    elif i == 2:
        attacked = blur(img, [3, 2])
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, [3, 5])
    elif i == 5:
        attacked = resizing(img, 0.5)
    elif i == 6:
        attacked = jpeg_compression(img, 75)
    return attacked

def random_mark(mark_size):
    fakemark = np.random.uniform(0.0, 1.0, mark_size)
    fakemark = np.uint8(np.rint(fakemark))
    return fakemark
def attack_name(numAttack):
    if numAttack == 0:
        return "awgn"
    elif numAttack == 1:
        return "blur"
    elif numAttack == 2:
        return "sharpening"
    elif numAttack == 3:
        return "median"
    elif numAttack == 4:
        return "resizing"
    elif numAttack == 5:
        return "jpeg"
######################################################################

imagesToAttack = f.imageOnFolder('../imagesToAttack/')
# mark = np.load('../Utilities/panebbianco.npy')
results = []
thisImageResults = []
attacksFunctions = [awgn, blur, sharpening, median, resizing, jpeg_compression]
str_arr =   [1,   0.5,  1,  1, 1, 90] # parametri partenza
alpha_arr = [0.5, 0.5, 0.5, 2, -0.5, -5] # incrementi

for path in imagesToAttack:
    thisImageResults = []
    groupName = path.split('_')[0]
    imageName = path.split('_')[1]
    originalPath = "../originalImages/" + str(imageName)
    watermarkedPath = "../imagesToAttack/" + str(path)
    print(f'Testing {imageName} of the group {groupName}...')

    # Read image
    watermarked = cv2.imread(watermarkedPath, 0)

    #Embed Watermark
    #watermarked = f.ss_watermark(im,mark,0.01)
    # watermarked = embeddedFinalMethod(im, mark)
    # waterWpsnr = f.wpsnr(im, watermarked)
    # print("watermarked image wpsnr: " + str(waterWpsnr))

    res_att = np.copy(watermarked)
    for c in range(6):
        wpsnr = 36
        found = 1
        strength = str_arr[c]
        alpha = alpha_arr[c]
        failed_att = 0
        while found == 1 and wpsnr >= 35 and failed_att == 0:
            strength += alpha
            print(attack_name(c))
            res_att = attacksFunctions[c](watermarked, strength)
            res_att = np.rint(res_att).astype(int)
            cv2.imwrite('tmp.bmp', res_att)
            gd = __import__("detection_" + groupName)
            #import detection_A as detection
            found, wpsnr = gd.detection(originalPath, watermarkedPath, 'tmp.bmp')

            """
            wpsnr = f.wpsnr(watermarked, res_att)
            if f.similarity(mark, f.ss_simWatermark(im, res_att, 0.01)) > 12:
                found = 1
            else:
                found = 0
            """
            if wpsnr < 35:
                failed_att = 1
            print("found:"+str(found))
            print("wpsnr:"+str(wpsnr))

            if strength == 0 and c==4:
                failed_att=1
        if failed_att == 0:
            res = {
                "imagePath": watermarkedPath,
                "imageName": imageName,
                "groupName": groupName,
                "methodName": attack_name(c),
                "methodCode": c,
                "WPSNR": wpsnr,
                "params": strength,
            }
            thisImageResults.append(res)
            results.append(res)

    if thisImageResults: # If there is at least one attack
        # Save the images with the best attack
        best_attack = sorted(thisImageResults, key=lambda x:x["WPSNR"])[-1]
        watermarked = cv2.imread(best_attack["imagePath"], 0)
        attackedImage = attacksFunctions[best_attack["methodCode"]](watermarked, best_attack["params"])
        cv2.imwrite('../attackedimages/panebbianco_' +  best_attack["groupName"] + "_" + best_attack["imageName"], res_att)
        saveThis = [best_attack["imageName"], best_attack["groupName"], best_attack["WPSNR"], f'{best_attack["methodName"]} param: {best_attack["params"]}']
        append_list_as_row('../attackedimages/attacks.csv', saveThis)

print(results)
print("\n")
for res in results:
    print(f'imge: {res["imageName"]}\ngroup: {res["groupName"]}\nmethod: {res["methodName"]}\nWPSNR: {res["WPSNR"]}\nparams: {res["params"]}\n')
