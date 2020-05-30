# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:29:00 2020

@author: siddh
"""
    
import time
import sys

toolbar_width = 40

import csv
import os , sys
import cv2 as cv
import numpy as np
from time import sleep
from time import time
from skimage import filters
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def blueRatioHistogram(img):
 # t1 = time()
  #img = cv.imread('A00_01.jpg')
  red = img[:,:,2]
  blue = img[:,:,0]
  green = img[:,:,1]

  red = tf.convert_to_tensor(red)
  green = tf.convert_to_tensor(green)
  blue = tf.convert_to_tensor(blue)

  blue = tf.to_float(blue)
  red = tf.to_float(red)
  green = tf.to_float(green)
  #100 * b
  b100 = tf.multiply(blue,100.)

  #r+g
  r_g = tf.add(red,green)

  #r+g+b
  r_g_b = tf.add(r_g,blue)

  one = tf.constant([[1.]])

  #r+g+b+1
  r_g_b_1 = tf.add(r_g_b,one)

  #r+g+1
  r_g_1 = tf.add(r_g,one)

  #factor1 = (100*b)/(r+g+1)
  factor1 = tf.div(b100,r_g_1)

  #256
  t56 = tf.multiply(one,255.)

  #factor2 = 256/(r+g+b+1)
  factor2 = tf.div(t56,r_g_b_1)

  #brh = factor1*factor2
  brh = tf.multiply(factor1,factor2)
  #normalising brh and scaling to 256
  maxb =brh
  a = tf.reduce_max(maxb,[0,1])
  brh = tf.div(brh,a)
  brh = tf.multiply(brh,255.)
  brh = tf.round(brh)
  brh = tf.cast(brh,tf.uint8)


  with tf.Session() as sess:
    brh = sess.run(brh)
    #equal = sess.run(equal)
    #maxa = sess.run(a)
  #t2 = time()
 # t = (t2 - t1)
  #print("Time taken is "+str(t)+"us")
  return brh

def OTSU_treshold(img):
    return cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]

def open_image(img):
    kernel = np.ones((5,5),np.uint8)
    morph = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
#    new_kernel = np.ones((1,1),np.uint8)
#   morph = cv.dilate(img,new_kernel,iterations = 1)
    return morph
P = [1,0,-1,0,1,-1,1,-1]
Q = [0,1,0,-1,1,-1,-1,1]


    
def connected(i,j,blob,image,is_found):
    blob.append((i,j))
    is_found[i][j] = 1
    
    for p in P:
        for q in Q:
            x = i+p
            y = j+q
            if (x<image.shape[0] and x>=0 and y<image.shape[1] and y>=0 and is_found[x][y]==0 and image[x][y]>0):
                connected(x,y,blob,image,is_found)
    return
    
def blob_detect(image):

    print('Start.')
    blobs = []
    is_found = np.zeros(image.shape)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            if image[i][j]>0 and is_found[i][j]==0:
                print('\rNumber of nuclei detected: [%d]'%i, end="")
                blob = [] 
                connected(i,j,blob,image,is_found)
                blobs.append(blob)
    print('end.')
    return blobs

def get_means(blobs):
    means = []
    for blob in blobs:
        a = [x[0] for x in blob]
        b = [y[1] for y in blob]
        x_mean = sum(a)//len(a)
        y_mean = sum(b)//len(b)
        means.append((x_mean, y_mean))
        
    return means

def is_mitotic(blob, m_cell_blobs):
    for pixel in [(y,x) for x,y in blob]:
        for m_cell_blob in m_cell_blobs:
            if pixel in m_cell_blob:
                return True
    return False

def classify_cells(blobs, m_cell_blobs):
    detected_mcells = []
    non_mitotic = []
    for i, blob in enumerate(blobs):
        if is_mitotic(blob, m_cell_blobs):
            detected_mcells.append(i)
        else:
            non_mitotic.append(i)
    return detected_mcells, non_mitotic

def get_mcells(path):
    path = str(path)
    m_cells = []

    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i,row in enumerate(csv_reader):
            m_cell = []    
            for i in range(0,len(row)-1,2):
                m_cell.append((int(row[i]), int(row[i+1])))
            m_cells.append(m_cell)
    return m_cells

def pad_side(padded, side,patch):
    for x,y in side:
        padded[x:x+50,y:y+50] = patch
    
import numexpr as ne

def bincount_numexpr_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    eval_params = {'a0':a2D[:,0],'a1':a2D[:,1],'a2':a2D[:,2],
                   's0':col_range[0],'s1':col_range[1]}
    a1D = ne.evaluate('a0*s0*s1+a1*s0+a2',eval_params)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)

  
    
def pad_image(img):
    clr = bincount_numexpr_app(img)
    padded = cv.copyMakeBorder(img, 50, 50, 50, 50, cv.BORDER_CONSTANT, value = [int(x) for x in list(clr)])
    return padded


def segment(img, means,m_cell_means):
    cells = []
    m_cells = []
    for i,(x,y) in  enumerate(means):
        x = x+50
        y = y+50
        x1 = x-40
        y1 = y-40
        
        x2 = x+40
        y2 = y+40
        cell = img[x1:x2,y1:y2]
        cells.append(cell)
    for i,(y,x) in  enumerate(m_cell_means):
        x = x+50
        y = y+50
        x1 = x-40
        y1 = y-40
        
        x2 = x+40
        y2 = y+40
        m_cell = img[x1:x2,y1:y2]
        m_cells.append(m_cell)
    
    return cells,m_cells

def get_blob_img(blobs,means, opened):
    cv.imwrite('opened.bmp', opened)
    opened = cv.imread('opened.bmp')
    for blob in blobs:
        clr = tuple(np.random.choice(range(256), size=3))
        for x,y in blob:
            opened[x,y] = clr
            
        for mean in means:
            color = (0,0,255)
            opened[mean[0],mean[1]] = color  
    return opened            


