# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:28:48 2019

@author: dmase
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
img = mpimg.imread('challenge.png')
circle = [147.5,104.5,27.2]
xc = circle[0]
yc = circle[1]
r, r1 , r2 , r3 = circle[2], 42, 62, 90 #boundary circles (hardcoded for now...)
N = 100
X = np.linspace(0,len(img[0,:]),len(img[0,:]))
Y = np.linspace(0,len(img[:,1]),len(img[:,1]))
X , Y = np.meshgrid(X , Y) 
grid = np.sqrt((X-xc)**2+(Y-yc)**2)
plt.imshow(grid) #creates a density plot of radius from center  
plt.show()
for i in range(0,len(grid[0,:])): #blacks out everything not of intrest
    for j in range(0,len(grid[:,0])):
        if grid[j,i] < r and grid[j,i] < r1 or grid[j,i] > r1 and grid[j,i] < r2 or grid[j,i] > r3:
            img[j,i] = 0
plt.imshow(img)
plt.imsave('isolated.png',img)