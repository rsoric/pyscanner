# importing the required libraries
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import imutils

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

# function for auto adjusted canny edge detection
def auto_canny(image, sigma=0.30):
	
    # compute the median of the single channel pixel intensities
	v = np.median(image)
	
    # apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	canny = cv2.Canny(image, lower, upper)
	
    # return the edged image
	return canny

# function to find edges on an image using gaussian blur and canny edge detection 
def find_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1)
    canny = auto_canny(blurred)
    return canny

# function to get the contours of the image
def getContours(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    return cnts

# function to increase brightness   
def increase_brightness(image, value=30):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return image



# function which displays the window of the scan preview
def showDocumentWithContours(image, cnts):
    # create copy of source image to display
    imageToDrawOn = image.copy()

    # loop accross the contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.026 * peri, True)
        
        #if we find 4 contours, it should be our document
        if len(approx) == 4:
            screenCnt = approx

            #draw the contours in blue
            cv2.drawContours(imageToDrawOn, [screenCnt], -1, (0, 0, 255), 3)
            break
    # resize the image so it fits in screen    
    image_resized = imutils.resize(imageToDrawOn, height = 720)

    # draw an overlay so the user finds it easier to place the document
    cv2.imshow('Document',image_resized)

#function to get scanned image from given contours, if they are found
def scan(image_original,cnts,img_counter):
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            screenCnt = approx

            #warp the image to fill the document
            warped = four_point_transform(image_original, screenCnt.reshape(4, 2))

            #increase brightness and contrast
            adjusted = increase_brightness(warped)

            #sharpen the image
            sharpened = cv2.filter2D(adjusted, -1, kernel)
            

            #show the image and save it
            cv2.imshow('Scanned image',sharpened)
            img_counter+=1
            img_name = "scan_{}.png".format(img_counter)
            cv2.imwrite(img_name, sharpened)
            break
        else:
            print("Contour not found!")
                


# set up video capture feed
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1980)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

img_counter = 0

# main loop
while True:


    ret, image = cam.read()
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if not ret:
        print("failed to grab frame")
        break

    image_edged = find_edges(image)
    cnts = getContours(image_edged)

    showDocumentWithContours(image,cnts)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC was pressed
        print("Closing...")
        break
    elif k%256 == 32:
        # SPACE was pressed
        scan(image,cnts,img_counter)


cam.release()
cv2.destroyAllWindows()