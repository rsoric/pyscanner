# Importing the required libraries
from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2
import imutils

# Function to show the scanned document
def scan(image):
    ratio = image.shape[0] / 720.0
    orig = image.copy()
    image = imutils.resize(image, height = 720)
    # convert the image to grayscale, blur it, and find edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 200)
    # save the contours
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # if our contour has four points, it should be the scanned document
        if len(approx) == 4:
            screenCnt = approx
            # show the contour on the display
            cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
            # four point transform the detected document
            warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
            # convert the warped image to grayscale
            warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            # show the scanned document
            cv2.imshow('warped',warped)
            break
        else:
            # alert the user so they can try to adjust the paper so the contours get detected
            print("Contour not found!")

# Similar function to scan() but only for displaying each frame with detected contours
def showcontours(image):
    ratio = image.shape[0] / 720.0
    orig = image.copy()
    image = imutils.resize(image, height = 720)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow('edged',edged)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            cv2.drawContours(image, [screenCnt], -1, (255, 0, 0), 2)
            break
        else:
            print("Contour not found!")
    # show the individual frame
    cv2.imshow('image',image)
                
def crop_center(img,cropx,cropy):
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx]

# load webcam feed and set dimensions to full HD
cam = cv2.VideoCapture(0)
cam.set(3,1920)
cam.set(4,1080)

img_counter = 0


# infinitely show teh recieved frame from the webcam until a key is pressed
while True:
    ret, frame = cam.read()
    frame = crop_center(frame,1280,720)

    if not ret:
        print("failed to grab frame")
        break
    showcontours(frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC was pressed
        print("Closing...")
        break
    elif k%256 == 32:
        # SPACE was pressed
        img_name = "scan_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        scan(frame)


cam.release()

cv2.destroyAllWindows()