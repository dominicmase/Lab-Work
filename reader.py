# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:22:10 2019

@author: dmase
"""
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob #library assositated with file identification
import os  #''

def circle_finder():
    img_color = cv2.imread('test.png') #test.png is created in last file function
    img = cv2.medianBlur(img_color,5) #blurs image to make processing easier
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts to grayscale
    img = np.invert(img)
#    cv2.imshow('image',img) #opens image
    cv2.waitKey(1000) #waits for image to render

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
#        cv2.imshow("detected circles", img)
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
#        cv2.imshow("detected circles", img) #visual (if not working uncomment)
#        cv2.imwrite('detected circles.png', img) 
#        cv2.waitKey(1000)
    
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT,1, rows/8,
                               param1 = 140, param2 = 50, minRadius = 20, maxRadius = 60) #parameters for inner circle 
    print(circles)
    height,width = img.shape
    radius , center = f1(circles)[0], f1(circles)[1]  #innermost circle
    print(center)
    r_offset1 = radius + int(radius/1.69) #outer inner circle radius is a multiple of inner radius
    r_offset2 = radius + int(radius*1.3) #inner outer cirlce
    r_offset3 = radius + int(radius*2.5) #outermost circle
    f2(circles,r_offset1)
    f2(circles,r_offset2)
    f2(circles,r_offset3)
    print(r_offset1, r_offset2, r_offset3) #feed this data into temp reader to form boundary circles
    return(radius,r_offset1,r_offset2,r_offset3,center)

def last_file(): #opens last file in folder since it will have the highest contrast, then uses that file to id circles
    list_of_files = glob.glob('C:/Users/dmase/Desktop/thisisthedata/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    print(latest_file)
    data = genfromtxt(latest_file, delimiter=',',skip_header = 3) #should open last file because it will have the highest contrast
    data = np.delete(data,0,1)
    data = np.delete(data,-1,1)
    plt.imshow(data)
    plt.imsave('test.png',data)
    return(list_of_files)
def iso(circle_info,list_of_files): #isolates important regions of image. 
    xc = circle_info[4][0]
    yc = circle_info[4][1]
    r, r1 , r2 , r3 = circle_info[0], circle_info[1], circle_info[2], circle_info[3]
    print(list_of_files)
    for i in range(0,len(list_of_files)-100,4):
        data = genfromtxt(list_of_files[i], delimiter=',',skip_header = 3) #should open last file because it will have the highest contrast
        data = np.delete(data,0,1)
        data = np.delete(data,-1,1)
#        plt.imshow(data)
        plt.imsave('test.png',data)
        img = mpimg.imread('test.png')
        X = np.linspace(0,len(img[0,:]),len(img[0,:]))
        Y = np.linspace(0,len(img[:,1]),len(img[:,1]))
        X , Y = np.meshgrid(X , Y) 
        grid = np.sqrt((X-xc)**2+(Y-yc)**2)
        print(list_of_files[i])
        for i in range(0,len(grid[0,:])): #blacks out everything not of intrest
            for j in range(0,len(grid[:,0])):
                if grid[j,i] < r and grid[j,i] < r1 or grid[j,i] > r1 and grid[j,i] < r2 or grid[j,i] > r3: 
                    img[j,i] = 0
        plt.imshow(img)
        plt.show()
#    plt.imsave('isolated.png',img)
list_of_files = last_file()
circle_info = circle_finder() #shape (radius,r_offset1,r_offset2,r_offset3,center)
iso(circle_info,list_of_files)
#print(circle_info)