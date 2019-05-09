# encoding: utf-8
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
import shelve
from MNIST_process_py3 import load_train_images,load_train_labels,load_test_images,load_test_labels

'''dataset loading'''
train_images = load_train_images()
train_labels = load_train_labels()
test_images = load_test_images()
test_labels = load_test_labels()

#%%
'''Simulation parameter setting'''
Batch_num = 600
Sample_num = 60000
Epoch_N = 100
Batch_N = Sample_num / Batch_num * Epoch_N
'''hidden layer parameter'''
num_hidden_neurons = 800
param_refractory_period = 100
'''weight change parameter'''
pulse_p= 1
pulse_n= -1
a_u=0.5
b_u=1/a_u
ap=1e-1
bp=1.029e1
ad=a_u*1e-1
bd=b_u*1.029e1
'''Synapses array initialization '''
Gw_L1 = np.random.uniform(0.01, 0.2, [784, num_hidden_neurons])  # initializing synapses of the first layer
Gw_L2 = np.zeros([num_hidden_neurons,10])
'''Neurons initialization'''
threshold_input = 0.5
refractory_period = np.zeros(num_hidden_neurons)

active_status = np.zeros(num_hidden_neurons)#this value increases when a certain neuron fires, otherwise it decreases
#labels = set(np.linspace(0,9,10).astype(int))
#out_label = np.zeros(1)
'''Data pre-processing and Training'''
for Epoch in range(0, Epoch_N):
    print('Epoch=%d' % Epoch)
    sequence_in = np.random.permutation(Sample_num) # Generate the random sequence of trainning pattern input
    # training in a batch
    for Batch in range(0, int(Sample_num/Batch_num)):
        # trainning
#        print(Batch)
        for n in range(Batch*Batch_num, (Batch+1)*Batch_num):
            input_sample = train_images[sequence_in[n]]/255
            input_label = train_labels[sequence_in[n]].astype(int)
            '''sort'''
            input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
            '''Binaryzation of input pattern'''
            input_sample = (input_sample > threshold_input) * pulse_p + (input_sample <= threshold_input) * pulse_n#(-0.145)  # binaryzation
            '''Dot production, with Gw_L1'''
            neuron_l1_signal = np.dot(input_sample, Gw_L1)  # calculation
            neuron_l1_sort = np.argsort(-neuron_l1_signal, 0)
            refractory_period += -1
            np.clip(refractory_period,0,num_hidden_neurons,refractory_period)
            for i in range(0,num_hidden_neurons):
                if refractory_period[neuron_l1_sort[i]]==0 :
                    neuron_fire_l1 = neuron_l1_sort[i]
                    temp_neuron_status = active_status[neuron_fire_l1]
                    # active_status change
                    temp_neuron_status += 1
                    if temp_neuron_status > 2.5 :
                        refractory_period[neuron_fire_l1]=param_refractory_period
                        temp_neuron_status = 0# reset this neuron                                    
                    break
            active_status = active_status*0.8
            active_status[neuron_fire_l1] = temp_neuron_status
#            print('neuron num=%d,label=%d' % (neuron_fire_l1,input_label))
            '''Weight update'''
            for i in range(0,784):
                x0 = Gw_L1[i, neuron_fire_l1]
                if input_sample[i]>0 :
                    dx = ap*np.exp(-bp*x0)
                    x = x0+dx
                else:
                    dx = ad*np.exp(-bd*(1-x0))
                    x = x0-dx
                Gw_L1[i, neuron_fire_l1] = x
            np.clip(Gw_L1,0,1,Gw_L1)
           
            '''Second layer'''
            for i in range(0,10):
                x0 = Gw_L2[neuron_fire_l1,i]
                if i==input_label:
                    dx = ap*np.exp(-bp*x0)
                    x = x0+dx
                else:
                    dx = ad*np.exp(-bd*(1-x0))
                    x = x0-dx   
                Gw_L2[neuron_fire_l1,i] = x
            np.clip(Gw_L2,0,1,Gw_L2)       
    correct=0
    for n in range(0,10000):
        input_sample = test_images[n]/255
        input_label = test_labels[n].astype(int)
        
        #first layer
        input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
        '''Binaryzation of input pattern'''
        input_sample = (input_sample > threshold_input) * pulse_p + (input_sample <= threshold_input) *pulse_n#(-0.145)  # binaryzation
        '''Dot production, with Gw_L1'''
        neuron_l1_signal = np.dot(input_sample, Gw_L1)  # calculation
        neuron_l1_sort = np.argsort(-neuron_l1_signal, 0)
        
        #second layer
        neuron_l2_sample = np.ones(num_hidden_neurons)*(0)
        for i in range(0,17):
            neuron_l2_sample[neuron_l1_sort[i]] = np.exp(-i*0.39)#1-0.08*i
        neuron_l2_signal = np.dot(neuron_l2_sample,Gw_L2)
        neuron_l2_sort = np.argsort(-neuron_l2_signal,0)
        neuron_l2_out = neuron_l2_sort[0]
        if neuron_l2_out == input_label :
            correct +=1
    accuracy=correct/10000
    print(accuracy)

#%%  
correct=0
for n in range(0,10000):
    input_sample = test_images[n]/255
    input_label = test_labels[n].astype(int)
    
    #first layer
    input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
    '''Binaryzation of input pattern'''
    input_sample = (input_sample > threshold_input) * pulse_p + (input_sample <= threshold_input) *pulse_n#(-0.145)  # binaryzation
    '''Dot production, with Gw_L1'''
    neuron_l1_signal = np.dot(input_sample, Gw_L1)  # calculation
    neuron_l1_sort = np.argsort(-neuron_l1_signal, 0)
    
    #second layer
    neuron_l2_sample = np.ones(num_hidden_neurons)*(0)
    for i in range(0,17):
        neuron_l2_sample[neuron_l1_sort[i]] = np.exp(-i*0.39)#1-0.08*i
    neuron_l2_signal = np.dot(neuron_l2_sample,Gw_L2)
    neuron_l2_sort = np.argsort(-neuron_l2_signal,0)
    neuron_l2_out = neuron_l2_sort[0]
    if neuron_l2_out == input_label :
        correct +=1
accuracy=correct/10000
print(accuracy)
#%%    
Gw_L2
n=61    
print(Gw_L2[n,:])
image_temp= Gw_L1[:,n].reshape((28,28))
plt.imshow(image_temp)