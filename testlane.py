import cv2
import numpy as np

def _birdview_transform(img):
    """Apply bird-view transform to the image
    """
    IMAGE_H = 480
    IMAGE_W = 640
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W+160, 0]])
    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H)) # Image warping
    return warped_img

def _find_lane_center(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Create a binary image by throttling the image. 
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) 
    
    #Determine the distance transform. 
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5) 
    
    # Make the distance transform normal. 
    dist_output = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX) 
    return dist_output
img = cv2.imread('./imgs/0.929116062845199.png')
warped_img = _birdview_transform(img)


  
# Display the distance transform 
cv2.imshow('Distance Transform', dist_output) 
cv2.waitKey(0) 

# cv2.imshow('1', img)
# cv2.imshow('2', warped_img)


# cv2.waitKey(0)