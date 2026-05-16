import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt



path = "/Volumes/SSK Media/visual_odometry/dataset/sequences/00"
image = "000000.png"

img1 = cv.imread(f'{path}/image_0/{image}', cv.IMREAD_UNCHANGED) #queryimage # left image
img2 = cv.imread(f'{path}/image_1/{image}', cv.IMREAD_UNCHANGED) #trainimage # right image


def nothing(x):
    pass

cv.namedWindow('disp', cv.WINDOW_NORMAL)
cv.resizeWindow('disp',600,1300)

# Creating an object of StereoBM algorithm
stereo = cv.StereoBM_create()
#stereo = cv.StereoSGBM.create(mode=cv.STEREO_SGBM_MODE_SGBM_3WAY)

cv.createTrackbar('minDisparity','disp',0,25,nothing)
cv.createTrackbar('numDisparities','disp',6,17,nothing)
cv.setTrackbarMin('numDisparities', 'disp', 1)
cv.createTrackbar('disp12MaxDiff','disp',0,25,nothing)
cv.createTrackbar('preFilterCap','disp', 1, 62, nothing)
cv.setTrackbarMin('preFilterCap', 'disp', 1)
cv.createTrackbar('blockSize','disp',11,50,nothing)
cv.createTrackbar('textureThreshold','disp',0,100,nothing)
cv.createTrackbar('uniquenessRatio','disp',0,100,nothing)
cv.createTrackbar('speckleRange','disp',0,100,nothing)
cv.createTrackbar('speckleWindowSize','disp',0,25,nothing)


if type(stereo) == cv.StereoBM:
    cv.createTrackbar('preFilterType','disp',0,1,nothing)
    cv.createTrackbar('preFilterSize','disp',0,25,nothing)

else:
    cv.createTrackbar('P1','disp', 8, 40, nothing)
    cv.createTrackbar('P2','disp', 32, 40, nothing)


 

while True:
    
    # Updating the parameters based on the trackbar positions
    numDisparities = cv.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv.getTrackbarPos('blockSize','disp')*2+5
    preFilterCap = cv.getTrackbarPos('preFilterCap','disp')
    uniquenessRatio = cv.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv.getTrackbarPos('minDisparity','disp')

    if type(stereo) == cv.StereoBM:
        preFilterType = cv.getTrackbarPos('preFilterType','disp')
        preFilterSize = cv.getTrackbarPos('preFilterSize','disp')*2 + 5
        textureThreshold = cv.getTrackbarPos('textureThreshold','disp')
    else:
        P1 = cv.getTrackbarPos('P1','disp')*11**2
        P2 = cv.getTrackbarPos('P2','disp')*11**2
        mode = cv.getTrackbarPos('mode','disp')
     


    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)

    if type(stereo) == cv.StereoBM:
        stereo.setPreFilterType(preFilterType)
        stereo.setPreFilterSize(preFilterSize)
        stereo.setTextureThreshold(textureThreshold)
    else:
        stereo.setP1(P1)
        stereo.setP2(P2)
        #stereo.setMode(mode)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(img1, img2).astype(np.float32)/16

    disparity = cv.normalize(disparity, None, 0, 1.0, cv.NORM_MINMAX, dtype=cv.CV_32F)
 
    # Displaying the disparity map
    cv.imshow("disp", disparity)
 
    # Close window using esc key
    if cv.waitKey(1) == 27:
      break


print()
print()

print(f'numDisparities: {numDisparities}')
print(f'blockSize: {blockSize}')
print(f'preFilterCap: {preFilterCap}')
print(f'uniquenessRatio: {uniquenessRatio}')
print(f'speckleRange: {speckleRange}')
print(f'speckleWindowSize: {speckleWindowSize}')
print(f'disp12MaxDiff: {disp12MaxDiff}')
print(f'minDisparity: {minDisparity}')

if type(stereo) == cv.StereoBM:
    print(f'preFilterType: {preFilterType}')
    print(f'preFilterSize: {preFilterSize}')
    print(f'textureThreshold: {textureThreshold}')
else:
    print(f'P1: {P1}')
    print(f'P2: {P2}')
    print(f'mode: {mode}')
