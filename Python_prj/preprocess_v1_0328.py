# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:12:52 2019

@author: Lijiwei
"""
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
import shelve
from MNIST_process import load_train_images,load_train_labels,load_test_images,load_test_labels

'''3*3'''
#def preprocess_right(image):
#    image_processed=zeros((26,26))
#    for i in range(0,26):
#        for j in range(0,26):
#            image_processed[i,j] = image[i,j+2] and image[i+1,j+1] and image[i+2,j]
#    return image_processed
#    
#def preprocess_left(image):
#    image_processed=zeros((26,26))
#    for i in range(0,26):
#        for j in range(0,26):
#            image_processed[i,j] = image[i,j] and image[i+1,j+1] and image[i+2,j+2]
#    return image_processed    
#    
#def preprocess_vertical(image):
#    image_processed=zeros((26,28))
#    for i in range(0,26):                                                                         
#        for j in range(0,28):
#            image_processed[i,j] = image[i,j] and image[i+1,j] and image[i+2,j]
#    return image_processed    
#    
#def preprocess_horizonal(image):
#    image_processed=zeros((28,26))
#    for i in range(0,28):                                                                         
#        for j in range(0,26):
#            image_processed[i,j] = image[i,j] and image[i,j+1] and image[i,j+2]
#    return image_processed    
        
#'''4*4 AND'''
#def preprocess_right(image):
#    image_processed=zeros((25,25))
#    for i in range(0,25):
#        for j in range(0,25):
#            image_processed[i,j] = image[i+3,j] and image[i+2,j+1] and image[i+1,j+2] and image[i,j+3]
#    return image_processed
#    
#def preprocess_left(image):
#    image_processed=zeros((25,25))
#    for i in range(0,25):
#        for j in range(0,25):
#            image_processed[i,j] = image[i,j] and image[i+1,j+1] and image[i+2,j+2] and image[i+3,j+3]
#    return image_processed    
#    
#def preprocess_vertical(image):
#    image_processed=zeros((25,28))
#    for i in range(0,25):                                                                         
#        for j in range(0,28):
#            image_processed[i,j] = image[i,j] and image[i+1,j] and image[i+2,j] and image[i+3,j]
#    return image_processed    
#    
#def preprocess_horizonal(image):
#    image_processed=zeros((28,25))
#    for i in range(0,28):                                                                         
#        for j in range(0,25):
#            image_processed[i,j] = image[i,j] and image[i,j+1] and image[i,j+2] and image[i,j+3]
#    return image_processed    
  
'''k_size*k_size AND'''
def preprocess_right(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(0,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_size-k-1]
    return image_processed
    
def preprocess_left(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(0,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k]
    return image_processed    
    
def preprocess_vertical(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(0,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_middle]
    return image_processed    
    
def preprocess_horizonal(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(0,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k_middle,j+k]
    return image_processed    
    
#%%  
def pooling(image):
    (m,n)=image.shape#get the size of image
    step=2#2*2
#    m_pool=(step-m%step+m)//step
#    n_pool=(step-n%step+n)//step
    m_pool=m//step
    n_pool=n//step
    image_temp=zeros((m_pool*step,n_pool*step))
#    image_temp[0:m,0:n]=image
    image_temp[0:m_pool*step,0:n_pool*step]=image[0:m_pool*step,0:n_pool*step]
    image_pooled=zeros((m_pool,n_pool))
    
    for i in range(0,m_pool):
        for j in range(0,n_pool):
            image_pooled[i,j] = image_temp[i*step,j*step] or image_temp[i*step+1,j*step]\
            or image_temp[i*step+1,j*step+1] or image_temp[i*step,j*step+1] 
    return image_pooled
            

               
                
                
            
    
                
#%%    
def run():


    # 查看前十个数据及其标签以读取是否正确
    # for i in range(2):
    #     print train_labels[i],test_labels[i]
    #     fig=plt.figure()
    #     ax=fig.add_subplot(211)
    #     plt.imshow(train_images[i], cmap='gray')
    #     ax = fig.add_subplot(212)
    #     plt.imshow(test_images[i], cmap='gray')
    #     plt.show()


    abc=train_images[1]
    plt.imshow(abc, cmap='gray')
    plt.show()
    print 'initial image done'
    
    
    abc = (abc > 127) # binaryzation
    k_size=5
    abc1=preprocess_right(abc,k_size)
    plt.imshow(abc1, cmap='gray')
    plt.show()
    print 'preprocess_right done'

    abc8=pooling(abc1)
    plt.imshow(abc8, cmap='gray')
    plt.show()
    print 'pooling done'    
    
    abc2=preprocess_left(abc,k_size)
    plt.imshow(abc2, cmap='gray')
    plt.show()
    print 'preprocess_left done'

    abc7=pooling(abc2)
    plt.imshow(abc7, cmap='gray')
    plt.show()
    print 'pooling done'    
    
    abc3=preprocess_horizonal(abc,k_size)
    plt.imshow(abc3, cmap='gray')
    plt.show()
    print 'preprocess_horizonal done'    
    
    abc5=pooling(abc3)
    plt.imshow(abc5, cmap='gray')
    plt.show()
    print 'pooling done'    
    
    abc4=preprocess_vertical(abc,k_size)
    plt.imshow(abc4, cmap='gray')
    plt.show()
    print 'preprocess_vertical done'    
    
    abc6=pooling(abc4)
    plt.imshow(abc6, cmap='gray')
    plt.show()
    print 'pooling done'    
    
    
#%%    
if __name__ == '__main__':
    #%%
    train_images = load_train_images()
    # train_labels = load_train_labels()
    # test_images = load_test_images()
    # test_labels = load_test_labels()
    #%%
    run()