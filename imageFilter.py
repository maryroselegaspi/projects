# -*- coding: utf-8 -*-
"""
Create instagram filter using opencv
"""

import cv2
import numpy as np


# dummy funtion that does nothing
def dummy(value):
    pass

# define convulution kernels
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
gaussian_kernel1 = cv2.getGaussianKernel(3,0)
gaussian_kernel2 = cv2.getGaussianKernel(5,0)
box_kernel = np.array([[1,1,1,],[1,1,1],[1,1,1]], np.float32)/9.0 # averaging kernel

kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel] # add all kernels above

# read an image, make a gray scale copy
image_original = cv2.imread('panagbenga.jpg')
gray_original = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

# create the UI (window and track)
cv2.namedWindow('Image Filter')

#arguments:create trackbarName, windowName, value (initial value), count(max value), onChange(event handler)
cv2.createTrackbar('contrast', 'Image Filter', 1, 100, dummy)
# name=brightness, initial value=50, max value =100, event handler=dummy
cv2.createTrackbar('brightness', 'Image Filter', 50, 100, dummy)
cv2.createTrackbar('filter', 'Image Filter', 0, len(kernels)-1, dummy) # TODO: update max value to number of filters
cv2.createTrackbar('grayscale', 'Image Filter', 0, 1, dummy)

# main UI loop
count = 1 # for saving
while True:
    # read all of the trackbar values
    grayscale = cv2.getTrackbarPos('grayscale', 'Image Filter')
    contrast = cv2.getTrackbarPos('contrast', 'Image Filter')
    brightness = cv2.getTrackbarPos('brightness', 'Image Filter')
    kernel_idx = cv2.getTrackbarPos('filter','Image Filter')
    
    # apply filters
    color_modified = cv2.filter2D(image_original, -1, kernels[kernel_idx])
    gray_modified = cv2.filter2D(gray_original, -1, kernels[kernel_idx])
    
    # apply brightness and contrast
    color_modified = cv2.addWeighted(color_modified, contrast, np.zeros_like(image_original), 0 , (brightness - 50))
    gray_modified = cv2.addWeighted(gray_modified, contrast, np.zeros_like(gray_original), 0 , (brightness - 50))
 
    # wait for keypress(100 milliseconds)
    key =  cv2.waitKey(100)
    if key == ord('q'): #convert character into integer
        break # quit while and control
    elif key == ord('s'):
        
        # save image
        if grayscale == 0:
            cv2.imwrite('output-{}.png'.format(count), color_modified)
        else:
            cv2.imwrite('output-{}.png'.format(count), gray_modified)
        count += 1
    
    # Show the image
    if grayscale == 0:
 
        cv2.imshow('Image Filter', color_modified)
    else:
        cv2.imshow('Image Filter', gray_modified)



# window cleanup
cv2.destroyAllWindows()

