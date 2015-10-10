import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

"""
	grabCut.py

	This script is going to take in a camera capture image and coordinate on this image.
	The aim is to get a decent segmented block of the image that can be matched up
	with a list of known objects and finally identify what kind of object that coordinate
	was pointing at.

	This initial version:
	(1) Performs a basic contour and edge detection to find top 5 largest rectangles that
	    highly likely be one of objects that we are interested in.
	(2) Take average of areas of these rectangles, and assume that our "point of interest"
	    is a square centered at the coordinate, with an area equal to this average.
	(3) Use this square as initial mask for grabCut algorithm, which will give us a piece
	    of image which we can use for identification.

	NOTE: If better algorithm / methodology is available, then please make this script better.

"""

# This image will also be provided from camera module
img = cv2.imread('Sample_6.png')              # img.shape : (486, 648, 3)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 35, 125)
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(map(cv2.contourArea, cnts), reverse=True)[:5]
avgSize = sum(cnts)/len(cnts)

sq = math.sqrt(avgSize)

# Assume following coordinate is given.
(x,y) = (324, 223)

halfSq = int(sq/2)
rect = (x-halfSq,y-halfSq,x+halfSq,y+halfSq)
cv2.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(200,0,0),2)

cv2.imshow('img',img)
cv2.waitKey(0)

mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (486, 648)

bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)

# this modifies mask 
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

# If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

# adding additional dimension for rgb to the mask, by default it gets 1
# multiply it with input image to get the segmented image
img_cut = img*mask2[:,:,np.newaxis]

# Obviously, following lines will not be necessary for functional code.
plt.subplot(211),plt.imshow(img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(212),plt.imshow(img_cut)
plt.title('Grab cut'), plt.xticks([]), plt.yticks([])
plt.show()