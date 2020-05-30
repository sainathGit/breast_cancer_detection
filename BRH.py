# -*- coding: utf-8 -*-
"""
Created on Fri May 15 19:29:00 2020

@author: siddh
"""


import os , sys
import cv2 as cv
import numpy as np
from time import sleep
import tensorflow as tf
from time import time
from skimage import filters


def blueRatioHistogram(img):
  t1 = time()
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
  brh = tf.cast(brh,tf.int32)


  with tf.Session() as sess:
    brh = sess.run(brh)
    #equal = sess.run(equal)
    #maxa = sess.run(a)
  t2 = time()
  t = (t2 - t1)
  #print("Time taken is "+str(t)+"us")
  return brh

def OTSU_treshold(img):
    return cv.threshold(img,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)[1]