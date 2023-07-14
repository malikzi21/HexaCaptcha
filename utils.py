import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from sklearn.cluster import KMeans
import pickle as pkl
import math
from sklearn.model_selection import train_test_split

def remove_stray_lines(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Dilate then erode to remove noise
    # Stronger dilation than erosion
    kernel1 = np.ones((3,3),np.uint8)
    kernel2 = np.ones((4,4),np.uint8)
    img = cv.dilate(img,kernel2,iterations = 1)
    img = cv.erode(img,kernel1,iterations = 1)
    return img

def morph_grad_and_threshold(img):
    # Morphological gradient
    kernel3 = np.ones((6,6),np.uint8)
    img = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel3)
    img = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    return img

def find_largest_connected_component(img):
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8, ltype=cv.CV_32S)
    largest = 0
    largest_area = 0
    for i in range(1, nlabels):
        area = stats[i, cv.CC_STAT_AREA]
        if area > largest_area:
            largest_area = area
            largest = i
    img[labels == largest] = 0
    return img

def get_parent(hierarchy, index):
    return hierarchy[0][index][3]


def contour_based_segmentation(img):

    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    useful_contours = []
    outer_contours = []
    for i in range(len(contours)):
        level = 1
        parent = get_parent(hierarchy, i)
        while parent != -1:
            level  = level + 1
            parent = get_parent(hierarchy, parent)
        if level == 1:
            outer_contours.append(contours[i])
        elif level == 2:
            useful_contours.append(contours[i])
        else:
            # Fit polygon to contour
            epsilon = 0.12*cv.arcLength(contours[i],True)
            approx = cv.approxPolyDP(contours[i],epsilon,True)
            # If area of polygon is close to contour area, it is a useful contour
            if(approx is not None and cv.contourArea(approx) > 0.9*cv.contourArea(contours[i])):
                useful_contours.append(contours[i])
    
    
    # Solid fill the contours
    # Sort useful contours by area
    outer_contours = sorted(outer_contours, key=cv.contourArea, reverse=True)
    # Find the largest decreasing area
    useful_contours.append(outer_contours[0])
    for i in range(1,len(outer_contours)):
        if cv.contourArea(outer_contours[i]) < cv.contourArea(outer_contours[i-1]) * 0.1:
            break
        useful_contours.append(outer_contours[i])
    # Add inner contours
    empty_img = np.zeros(img.shape, np.uint8)

    cv.drawContours(empty_img, useful_contours, -1, (255,0,0), -1)
    img = empty_img
    return img



def better_image_splitter(image,x_step = 5,y_step = 5):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image = cv.threshold(image, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]

    white_pixels = []

    for j in range(0,image.shape[1],x_step):
        for i in range(0,image.shape[0],y_step):
            if image[i,j] == 255:
                white_pixels.append([i,j])
    centers = []
    curr_x = 0
    curr_y = 0
    num = 0
   
    for i in range(1,len(white_pixels)):
        if(white_pixels[i][1] - white_pixels[i-1][1] > 35):
            centers.append([curr_x/num,curr_y/num])
            curr_x = 0
            curr_y = 0
            num = 0
        else:
            curr_x = curr_x + white_pixels[i][0]
            curr_y = curr_y + white_pixels[i][1]
            num = num + 1
    centers.append([curr_x/num,curr_y/num])
    centers = sorted(centers, key=lambda x: x[1])
    if(len(centers) != 4):
        return
    split_images = []
    for i in range(4):
        x = centers[i][1] - 70
        y = centers[i][0] - 70
        x = math.floor(x)
        y = math.floor(y)
        if(x < 0):
            x = 0
        if(y < 0):
            y = 0
        split_images.append(image[y:y+140, x:x+140])
    return split_images

def process(img):
    img = remove_stray_lines(img)
    img = morph_grad_and_threshold(img)
    img = find_largest_connected_component(img)
    img = contour_based_segmentation(img)
    images = better_image_splitter(img)
    if(images is None):
        return [],0
    if(len(images) != 4):
        pass
    return images[3], len(images)



