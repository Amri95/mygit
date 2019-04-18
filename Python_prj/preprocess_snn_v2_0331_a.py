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

'''k_size*k_size AND'''
def preprocess_right_upper(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(k_size//2,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_size-k-1]
    return image_processed
    
def preprocess_right_down(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(0,k_size//2+1):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_size-k-1]
    return image_processed    
    
def preprocess_left_upper(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(0,k_size//2+1):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k]
    return image_processed    
    
def preprocess_left_down(image,k_size):
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):
        for j in range(0,29-k_size):
            for k in range(k_size//2,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k]
    return image_processed 
    
def preprocess_vertical_upper(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(0,k_size//2+1):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_middle]
    return image_processed    

def preprocess_vertical_down(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(k_size//2,k_size):
                image_processed[i,j]=image_processed[i,j] and image[i+k,j+k_middle]
    return image_processed    
    
def preprocess_horizonal_left(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(0,k_size//2+1):
                image_processed[i,j]=image_processed[i,j] and image[i+k_middle,j+k]
    return image_processed   
    
def preprocess_horizonal_right(image,k_size):
    k_middle=k_size//2+1
    image_processed=ones((29-k_size,29-k_size))
    for i in range(0,29-k_size):                                                                         
        for j in range(0,29-k_size):
            for k in range(k_size//2,k_size):
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
         
         
def preprocess(image,input_threshold):
    # features extracting and pooling
    image_binary = (image > input_threshold) # binaryzation
    #feature extracting
    k_size=5
    
    image_feature_1=preprocess_right_upper(image_binary,k_size)
    image_feature_2=preprocess_right_down(image_binary,k_size)    
    image_feature_3=preprocess_left_upper(image_binary,k_size)    
    image_feature_4=preprocess_left_down(image_binary,k_size)

    image_feature_5=preprocess_vertical_upper(image_binary,k_size)
    image_feature_6=preprocess_vertical_down(image_binary,k_size)    
    image_feature_7=preprocess_horizonal_left(image_binary,k_size)    
    image_feature_8=preprocess_horizonal_right(image_binary,k_size)    
    
    #pooling
    image_pool_1=pooling(image_feature_1)    
    image_pool_2=pooling(image_feature_2)
    image_pool_3=pooling(image_feature_3)  
    image_pool_4=pooling(image_feature_4)
    
    image_pool_5=pooling(image_feature_5)    
    image_pool_6=pooling(image_feature_6)
    image_pool_7=pooling(image_feature_7)  
    image_pool_8=pooling(image_feature_8)
    #generating input for SNN    
    image_vector_size = size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
                       +size(image_pool_5)+size(image_pool_6)+size(image_pool_7)+size(image_pool_8)    
    image_vector=zeros(image_vector_size)
    
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
    
    m=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)  
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)
    image_vector[m:n] = image_pool_4.reshape(size(image_pool_5), order='c')
    
    m=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)  
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)+size(image_pool_6)
    image_vector[m:n] = image_pool_4.reshape(size(image_pool_6), order='c')
    
    m=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)+size(image_pool_6)
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)+size(image_pool_6)+size(image_pool_7)
    image_vector[m:n] = image_pool_4.reshape(size(image_pool_7), order='c')
    
    m=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)+size(image_pool_6)+size(image_pool_7)
    n=size(image_pool_1)+size(image_pool_2)+size(image_pool_3)+size(image_pool_4)\
        +size(image_pool_5)+size(image_pool_6)+size(image_pool_7)+size(image_pool_8)
    image_vector[m:n] = image_pool_4.reshape(size(image_pool_8), order='c')
    
    return image_vector

def dataset_preprocess():
    #Loading MNIST dataset
#    global train_images
    global train_labels
#    global test_images
    global test_labels
    global train_images_preprocessed
    global test_images_preprocessed
    global vector_size

    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    ##pattern parameter
    input_threshold=127 # the binary threshold of input pattern
    sample=preprocess(train_images[0],input_threshold)
    vector_size = size(sample)
    train_images_preprocessed = zeros((60000,vector_size))
    test_images_preprocessed = zeros((10000,vector_size))
    for n in range(0,60000):
        train_images_preprocessed[n] = preprocess(train_images[n],input_threshold)
    print  'preprocess train_image done'
    
    for n in range(0,10000):
        test_images_preprocessed[n] = preprocess(test_images[n],input_threshold)
    print  'preprocess test_image done'

    

#%%    
def run():
    # call the function to preprocess dataset

    '''Simulation parameter setting'''
    Batch_num=6000
    Sample_num=60000
    Epoch_N=30
    Batch_N = Sample_num / Batch_num * Epoch_N
    
    Gw = zeros((vector_size, 10))

    Acc=zeros(Batch_N)
    
    '''Data pre-processing and Training'''
    for Epoch in range(0,Epoch_N):
        sequence_in = np.random.permutation(Sample_num) # Generate the random sequence of trainning pattern input
        # training in a batch
        for Batch in range(0,Sample_num/Batch_num):
            #trainning
            for n in range(Batch*Batch_num,(Batch+1)*Batch_num):
                input_sample=train_images_preprocessed[sequence_in[n]]
                input_label=train_labels[sequence_in[n]]
                input_sample_signal=input_sample + (input_sample == 0) * (-0.01)
                output_signal = np.dot(input_sample_signal, Gw)# calculation
                out_sort = np.argsort(-output_signal, 0)  # 获取序列顺序
                # print out_sort[0],input_label
                if out_sort[0] != input_label:
                    Gw[:, int(input_label)] += input_sample
                'forgetting during a sample'
                Gw *=0.9999999999999
            'Accuracy Rate Test'
            correct=0
            for n in range(0,10000):
                input_sample=test_images_preprocessed[n]
                input_label=test_labels[n]
                input_sample_signal=input_sample + (input_sample == 0) * (-0.01)
                output_signal = np.dot(input_sample_signal, Gw)# calculation
                out_sort = np.argsort(-output_signal, 0)  # 获取序列顺序
                if out_sort[0] == input_label:
                    correct +=1
            print correct
            Acc[Batch+Epoch*(Sample_num/Batch_num)]=correct
        Gw *= 0.999   
    
 
    
    
#%%    
if __name__ == '__main__':
    #%%
    dataset_preprocess()


    #%%
    run()