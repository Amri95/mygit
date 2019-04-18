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
        
'''4*4 AND'''
def preprocess_right(image):
    image_processed=zeros((25,25))
    for i in range(0,25):
        for j in range(0,25):
            image_processed[i,j] = image[i+3,j] and image[i+2,j+1] and image[i+1,j+2] and image[i,j+3]
    return image_processed
    
def preprocess_left(image):
    image_processed=zeros((25,25))
    for i in range(0,25):
        for j in range(0,25):
            image_processed[i,j] = image[i,j] and image[i+1,j+1] and image[i+2,j+2] and image[i+3,j+3]
    return image_processed    
    
def preprocess_vertical(image):
    image_processed=zeros((25,28))
    for i in range(0,25):                                                                         
        for j in range(0,28):
            image_processed[i,j] = image[i,j] and image[i+1,j] and image[i+2,j] and image[i+3,j]
    return image_processed    
    
def preprocess_horizonal(image):
    image_processed=zeros((28,25))
    for i in range(0,28):                                                                         
        for j in range(0,25):
            image_processed[i,j] = image[i,j] and image[i,j+1] and image[i,j+2] and image[i,j+3]
    return image_processed    
  
#%%  
def pooling(image):
    (m,n)=image.shape#get the size of image
    step=3#2*2
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
         
         
def preprocess(image,input_threshold):
    # features extracting and pooling
    image_binary = (image > input_threshold) # binaryzation
    #feature extracting
    image_feture_1=preprocess_right(image_binary)
    image_feture_2=preprocess_left(image_binary)    
    image_feture_3=preprocess_horizonal(image_binary)    
    image_feture_4=preprocess_vertical(image_binary)
    #pooling
    image_pool_1=pooling(image_feture_1)    
    image_pool_2=pooling(image_feture_2)
    image_pool_3=pooling(image_feture_3)  
    image_pool_4=pooling(image_feture_4)
    #generating input for SNN    
    image_vector=zeros(size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4))
    m=0
    n=size(image_pool_1)    
    image_vector[m:n] = image_pool_1.reshape(size(image_pool_1), order='c')
    m=size(image_pool_1)
    n=size(image_pool_1)+size(image_pool_2)
    image_vector[m:n] = image_pool_2.reshape(size(image_pool_2), order='c')
    m=size(image_pool_1)+size(image_pool_2)  
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)
    image_vector[m:n] = image_pool_3.reshape(size(image_pool_3), order='c')    
    m=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)  
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)
    image_vector[m:n] = image_pool_4.reshape(size(image_pool_4), order='c')
    
    return image_vector

#%%    
def run():
    ##pattern parameter
    input_threshold=127 # the binary threshold of input pattern
    '''Simulation parameter setting'''
    Batch_num=600
    Sample_num=60000
    Epoch_N=3
    Batch_N = Sample_num / Batch_num * Epoch_N

    Gw = zeros((272, 10))

    Acc=zeros(Batch_N)
    '''Data pre-processing and Training'''
    for Epoch in range(0,Epoch_N):
        sequence_in = np.random.permutation(Sample_num) # Generate the random sequence of trainning pattern input
        # training in a batch
        for Batch in range(0,Sample_num/Batch_num):
            #trainning
            for n in range(Batch*Batch_num,(Batch+1)*Batch_num):
                input_sample=train_images[sequence_in[n]]
                input_label=train_labels[sequence_in[n]]
                input_sample=preprocess(input_sample,input_threshold)#.astype(np.int_)
                output_signal = np.dot(input_sample, Gw)# calculation
                out_sort = np.argsort(-output_signal, 0)  # 获取序列顺序
                # print out_sort[0],input_label
                if out_sort[0] != input_label:
                    Gw[:, input_label] += input_sample
                'forgetting during a sample'
                Gw *=0.99999999999
            'Accuracy Rate Test'
            correct=0
            for n in range(0,10000):
                input_sample=test_images[n]
                input_label=test_labels[n]
                input_sample=preprocess(input_sample,input_threshold)
                output_signal = np.dot(input_sample, Gw)# calculation
                out_sort = np.argsort(-output_signal, 0)  # 获取序列顺序
                if out_sort[0] == input_label:
                    correct +=1
            print correct
            Acc[Batch+Epoch*(Sample_num/Batch_num)]=correct
        Gw *= 0.99    
    
 
    
    
#%%    
if __name__ == '__main__':
    #%%
    #Loading MNIST dataset
    global train_images
    global train_labels
    global test_images
    global test_labels
    
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()

    #%%
    run()