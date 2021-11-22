import os
import cv2
import matplotlib.pyplot as plt
import functions as f

######################################################################

imagesToAttack = f.imageOnFolder('../imagesToAttack/')
i = 0

for path in imagesToAttack:
    thisImageResults = []
    groupName = path.split('_')[0]
    imageName = path.split('_')[1]
    originalPath = "../originalImages/" + str(imageName)
    watermarkedPath = "../imagesToAttack/" + str(path)
    watermarked = cv2.imread(watermarkedPath, 0)
    # cv2.imwrite('../imagesToAttack/panebbianco_'+groupName+'.bmp', watermarked)

    print(f'Testing {imageName} of the group {groupName}...')

    original = cv2.imread(originalPath, 0)
    watermarked = cv2.imread(watermarkedPath, 0)
    diff = original.astype(int) - watermarked
    i += 1
    plt.subplot(3, 3, i)
    plt.title(f'{imageName}_{groupName}')
    plt.imshow(diff)
plt.show()
