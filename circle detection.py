# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:05:44 2019

@author: dmase
"""
import numpy as np
import cv2
img_color = cv2.imread('challenge.png')
img = cv2.medianBlur(img_color,5) #blurs image to make processing easier
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts to grayscale
img = np.invert(img)
#clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(8,8))
#img = clahe.apply(img)
cv2.imshow('image',img) #opens image
cv2.waitKey(1000) #waits for image to render
color = img
rows = img.shape[0]
print(rows,img.shape[1])

def f1(circles):
    if circles is not None: 
        print('found')
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(img, center, radius, (255, 0, 255), 1)
    else: print('not found')
    cv2.imshow("detected circles", img)
    cv2.imwrite('detected circles.png', img) 
    cv2.waitKey(1000)
    return(radius,center)
def f2(circles,r):
    if circles is not None: 
        print('found')
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(img, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = r
            cv2.circle(img, center, radius, (255, 0, 255), 1)
    # Simple binary threshold
    else: print('not found')
    cv2.imshow("detected circles", img)
    cv2.imwrite('detected circles.png', img) 
    cv2.waitKey(1000)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, rows/8,
                           param1 = 140, param2 = 50, minRadius = 20, maxRadius = 60) #parameters for inner circle 
print(circles)
height,width = img.shape
mask = np.zeros((height,width), np.uint8)
radius , center = f1(circles)[0], f1(circles)[1]  #innermost circle
print(center)
r_offset1 = radius + int(radius/1.69) #outer inner circle radius is a multiple of inner radius
r_offset2 = radius + int(radius*1.3) #inner outer cirlce
r_offset3 = radius + int(radius*2.5) #outermost circle
f2(circles,r_offset1)
f2(circles,r_offset2)
f2(circles,r_offset3)
print(r_offset1, r_offset2, r_offset3) #feed this data into temp reader to form boundary circles

#circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, rows/8,
#                           param1 = 100, param2 = 41, minRadius = 40, maxRadius = 70)
#print(circles)
#f1(circles)
