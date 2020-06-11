#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created on Tue apr 21 14:37:53 2020

@author: bryantiernan
Name: Bryan Tiernan
ID: 16169093 
"""

##############################################################################
###############################   Imports  ###################################
##############################################################################

import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pylab as pylab
import imutils
from PIL import Image

##############################################################################
##############################  Functions  ###################################
##############################################################################

def generateHarrisMatrix(img, sigma = 2):
    x = np.zeros(img.shape)
    filters.gaussian_filter(img, 1, (0,1), output = x)
    y = np.zeros(img.shape)
    filters.gaussian_filter(img, 1, (1,0), output = y)
    a = filters.gaussian_filter(x * x, sigma)
    b = filters.gaussian_filter(x * y, sigma)
    c = filters.gaussian_filter(y * y, sigma)
    detM = (a * c) - (b ** 2)
    traceM = (a + c)
    return detM / traceM



def HPoints(HImg, threshold = 0.1, minDist = 10):
    # Find the top corner candidates above a threshold
    cornerThreshold = HImg.max() * threshold
    HImgThresholded = (HImg > cornerThreshold)
    # Find the co-ordinates of these candidates and their response values
    coords = np.array(HImgThresholded.nonzero()).T
    Val = np.array([HImg[c[0],c[1]] for c in coords])
    index = np.argsort(Val)
    # pointLocations stored in bool img 
    pointLocations = np.zeros(HImg.shape, dtype = 'bool')
    pointLocations[minDist:-minDist, minDist:-minDist] = True
    # nonmax suppression based on the pointLocations array
    realCoords = []
    for i in index[::-1]:
        r, c = coords[i]
        if pointLocations[r, c]:
            realCoords.append((r, c))
            pointLocations[r-minDist:r+minDist, c-minDist:c+minDist] = False  
    return realCoords


def plotHPoints(image, interestPoints):
    pyl.imshow(image, cmap='gray')
    pyl.plot([p[1] for p in interestPoints], [p[0] for p in interestPoints], 'ro')
    pyl.axis('off')
    pyl.show()            
           
    
    
def findDescriptors(image, interestPoints, width = 5):
    descriptors = []
    for coords in interestPoints:
        arg = image[coords[0] - width:coords[0] + width + 1, coords[1] - width:coords[1] + width + 1].flatten()
        arg -= np.mean(arg)
        arg /= np.linalg.norm(arg)
        descriptors.append(arg)
    return descriptors



def matchDescriptors(descriptors1, descriptors2, threshold = 0.95):
    arr1 = np.asarray(descriptors1, dtype = np.float32)
    arr2 = np.asarray(descriptors2, dtype = np.float32).T # Transposed
    responseMatrix = np.dot(arr1, arr2)
    originalMatrix = Image.fromarray(responseMatrix * 255)
    pairs = []
    for r in range(responseMatrix.shape[0]):
        rowMaximum = responseMatrix[r][0]
        for c in range(responseMatrix.shape[1]):
            if (responseMatrix[r][c] > threshold) and (responseMatrix [r][c] > rowMaximum):
                pairs.append((r,c))
            else:
                responseMatrix[r][c] = 0  
    thresholdedMatrix = Image.fromarray(responseMatrix * 255)
    return  originalMatrix, thresholdedMatrix, pairs



def plotMatches(img1, img2, IPoints1, IPoints2, pairs):
    rows1 = img1.shape[0]
    rows2 = img2.shape[0]
    if rows1 < rows2:
        img1 = np.concatenate((img1, np.zeros((rows2 - rows1, img1.shape[1]))), axis=0)
    elif rows2 < rows1:
        img2 = np.concatenate((img2, np.zeros((rows1 - rows2, img2.shape[1]))), axis=0)
    image3 = np.concatenate((img1, img2), axis=1)
    pyl.imshow(image3, cmap="gray")
    column1 = img1.shape[1]
    for i in range(len(pairs)):
        index1, index2 = pairs[i]
        pyl.plot([IPoints1[index1][1], IPoints2[index2][1] + column1],
                 [IPoints1[index1][0], IPoints2[index2][0]], 'g')
    pyl.axis('off')
    pyl.show()
    
    
    
def RANSAC(matches, coords1, coords2, matchDist=1.6):
    dist = matchDist ** 2
    offsets = np.zeros((len(matches), 2))
    for i in range(len(matches)):
        index1, index2 = matches[i]
        offsets[i, 0] = coords1[index1][0] - coords2[index2][0]
        offsets[i, 1] = coords1[index1][1] - coords2[index2][1]
    best_match_count = -1
    best_row_offset, best_col_offset = 1e6, 1e6
    for i in range(len(offsets)):
        match_count = 1.0
        offi0 = offsets[i, 0]
        offi1 = offsets[i, 1]
        if (offi0 - best_row_offset) ** 2 + (offi1 - best_col_offset) ** 2 >= dist:
            sum_row_offsets, sum_col_offsets = offi0, offi1
            for j in range(len(matches)):
                if j != i:
                    offj0 = offsets[j, 0]
                    offj1 = offsets[j, 1]
                    if (offi0 - offj0) ** 2 + (offi1 - offj1) ** 2 < dist:
                        sum_row_offsets += offj0
                        sum_col_offsets += offj1
                        match_count += 1.0
            if match_count >= best_match_count:
                best_row_offset = sum_row_offsets / match_count
                best_col_offset = sum_col_offsets / match_count
                best_match_count = match_count
                
    return best_row_offset, best_col_offset, best_match_count



def compImgs(img1, img2, rowOffset, colOffset):
    rowOffset = int(rowOffset)
    colOffset = int(colOffset)
    compImg = Image.new(img1.mode, (img1.width + abs(colOffset), img1.width + abs(rowOffset)))  
    compImg.paste(img1, (0, compImg.height - img1.height))
    compImg.paste(img2, (colOffset, compImg.height - img1.height + rowOffset))
    pyl.figure('Final composite Image')
    pyl.imshow(compImg)
    pyl.axis('off')
    pyl.show()
    return compImg


##############################################################################
##############################  Statements  ##################################
##############################################################################     

j = ["balloon","arch","al"]
for i in j:
    print ("\nReading Images")            
    HImg1 = (np.array(Image.open(i+'1.png').convert('L'), dtype=np.float32))
    HImg2 = (np.array(Image.open(i+'2.png').convert('L'), dtype=np.float32))
    imutils.imshow(HImg1)
    imutils.imshow(HImg2)
    
    
    
    print ("\nFinding Harris Matrices")
    img1 = generateHarrisMatrix(HImg1, 2)
    img2 = generateHarrisMatrix(HImg2, 2)
    imutils.imshow(img1)
    imutils.imshow(img2)
    
    
    
    print ("\nFinding Interest Points for both images")
    IPoints1 = HPoints(img1)
    IPoints2 = HPoints(img2)
    print ("Found " + str(len(IPoints1)) +  " interest points in image 1.")
    print ("Found " + str(len(IPoints2)) +  " interest points in image 2.")
    plotHPoints(HImg1, IPoints1)
    plotHPoints(HImg2, IPoints2)
    
    
    
    print ("Find image descriptors")
    descriptors1 = findDescriptors(HImg1, IPoints1)
    descriptors2 = findDescriptors(HImg2, IPoints2)
    
    
    
    print ("Find matches between Descriptors")
    originalMatrix, thresholdedMatrix, pairsList = matchDescriptors(descriptors1, descriptors2)
    
    
    
    print ("Response matrix before and after thresholding: ")
    pyl.subplot(121)
    pyl.imshow(originalMatrix)
    pyl.subplot(122)
    pyl.imshow(thresholdedMatrix)
    pyl.show()
    
    
    
    print ("Plot the matches between the two images:")
    result = plotMatches(HImg1, HImg2, IPoints1, IPoints2, pairsList)
    
    
    
    print ("RANSAC:")
    rowOffset, colOffset, bestMatches = RANSAC(pairsList, IPoints1, IPoints2)
    print ('Best match count: ' + str(bestMatches))
    print ('Row offset: ' + str(rowOffset))
    print ('column offset: ' + str(colOffset)) 
    
    
    
    print ("\n\tComposite image")
    colourimg1 = Image.open(i+'1.png')
    colourimg2 = Image.open(i+'2.png')
    final = compImgs(colourimg1, colourimg2, rowOffset, colOffset)