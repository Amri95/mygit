# encoding: utf-8
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
import shelve
from MNIST_process import load_train_images,load_train_labels,load_test_images,load_test_labels

def ANL_synapse(W0,n_pulse):# Function of weight updating
    ap = 1e-4
    bp = 1e-5
    if n_pulse > 0: # if the pulse exists
        # weight_change = ap*exp(-bp*W0)
        weight_change=0.01
        W1 = W0+weight_change
        if W1 > 1:
            W1 = 1
    else:
        W1 = W0

    return W1
def Weight_array_change(W_array0,V_in):# Updating procedure of weight array
    W_array = W_array0
    # array_num=np.size(W_array0)
    H_w=np.size(W_array0, 0)#hang row
    for h_w in range(0, H_w):
        W_array[h_w] = ANL_synapse(W_array0[h_w], V_in[h_w])
    return  W_array
    # if array_num==H_w:
    #     for h_w in range(0, H_w):
    #             W_array[h_w] = ANL_synapse(W_array0[h_w], V_in[h_w])
    # else:
    #     V_w=np.size(W_array0,1)#column
    #     for h_w in range(0,H_w):
    #         for v_w in range(0,V_w):
    #             W_array[h_w,v_w]=ANL_synapse(W_array0[h_w,v_w],V_in[h_w])

def snn_2layer_test(test_images, test_labels, Gw_L1, Gw_L2, vth_neuron_l1):
    correct = 0
    for n in range(0, 10000):
        input_sample = test_images[n]/255
        input_label = test_labels[n]

        # '''Binaryzation'''
        # input_sample = (input_sample > 0.5) * 1 + (input_sample <= 0.5) * (-0.1)  # binaryzation
        # input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
        '''sort'''
        input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
        sort_input_temp = np.argsort(-input_sample, 0)
        threshold_input_num = sort_input_temp[100]
        threshold_input = input_sample[threshold_input_num]  # find the threshold of the pixel value as active

        '''Binaryzation'''
        # input_sample = (input_sample > 0.5) * 1 + (input_sample <= 0.5) * (-0.1)  # binaryzation
        input_sample = (input_sample > threshold_input) * 1 + (input_sample <= threshold_input) * (-0.1)  # binaryzation
        '''Dot production'''
        neuron_L1_signal = np.dot(input_sample, Gw_L1)  # calculation

        '''Leaky and Fire, Layer 1'''
        neuron_fire_l1 = (neuron_L1_signal >= vth_neuron_l1) * 1 + (neuron_L1_signal < vth_neuron_l1) * (-0.1)

        '''Layer-2'''
        input_L2 = neuron_fire_l1
        neuron_L2_signal = np.dot(input_L2, Gw_L2)
        neuron_L2_sort = np.argsort(-neuron_L2_signal, 0)
        neuron_num_L2 = neuron_L2_sort[0]

        if neuron_num_L2 == input_label:
            correct += 1
    return correct


def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    '''Simulation parameter setting'''
    Batch_num = 600
    Sample_num = 60000
    Epoch_N = 4
    Batch_N = Sample_num / Batch_num * Epoch_N

    '''Synapses array initialization '''
    Gw_L1 = np.random.uniform(0.14, 0.16, [784, 100])  # initializing synapses of the first layer
    Gw_L2 = zeros((100, 10))  # initializing synapses of the second layer

    '''Neurons initialization'''
    vth_neuron_l1 = 15*ones(100)
    vth_min = 0.1
    vth_max = 100

    Acc = zeros(Batch_N)

    '''Data pre-processing and Training'''
    for Epoch in range(0, Epoch_N):
        sequence_in = np.random.permutation(Sample_num) # Generate the random sequence of trainning pattern input
        if Epoch == 1:
            print Epoch

        # training in a batch
        for Batch in range(0, Sample_num/Batch_num):
            # trainning
            for n in range(Batch*Batch_num, (Batch+1)*Batch_num):
                input_sample = train_images[sequence_in[n]]/255
                input_label = train_labels[sequence_in[n]]

                '''sort'''
                input_sample = input_sample.reshape(784, order='c')  # reshape the data array from 28*28 to 784*1
                sort_input_temp = np.argsort(-input_sample, 0)
                threshold_input_num = sort_input_temp[100]
                threshold_input = input_sample[threshold_input_num] # find the threshold of the pixel value as active


                '''Binaryzation'''
                # input_sample = (input_sample > 0.5) * 1 + (input_sample <= 0.5) * (-0.1)  # binaryzation
                input_sample = (input_sample > threshold_input) * 1 + (input_sample <= threshold_input) * (-0.1)  # binaryzation

                '''Dot production'''
                neuron_L1_signal = np.dot(input_sample, Gw_L1)  # calculation
                neuron_fire_l1 = (neuron_L1_signal >= vth_neuron_l1) * 1 + (neuron_L1_signal < vth_neuron_l1) * (-0.1)

                # choose fired neurons of layer 1, as the input pattern of the 2nd layer
                for num_of_neuron in range(0, 100):# change the threshold of neurons
                    temp_vth_neuron_l1 = vth_neuron_l1[num_of_neuron]
                    if neuron_fire_l1[num_of_neuron] != 1:
                        temp_vth_neuron_l1 *= 0.9
                        if temp_vth_neuron_l1 > vth_max:
                            temp_vth_neuron_l1 = 1
                    else:
                        temp_vth_neuron_l1 *= 1.1
                        if temp_vth_neuron_l1 < vth_min:
                            temp_vth_neuron_l1 = 0.1
                    vth_neuron_l1[num_of_neuron] = temp_vth_neuron_l1

                '''Weight change'''
                for label_input in range(0, 784):
                    if input_sample[label_input] == 1:
                        for label_output in range(0, 100):
                            if neuron_fire_l1[label_output] == 1:
                                W0 = Gw_L1[label_input, label_output]
                                # weight_change = ap*exp(-bp*W0)
                                weight_change = 0.1
                                W1 = W0 + weight_change
                                if W1 > 1:
                                    W1 = 1
                                Gw_L1[label_input, label_output] = W1

                'forgetting during a sample'
                Gw_L1 *= 0.99999999
        
                '''Layer-2'''
                input_L2 = neuron_fire_l1
                neuron_L2_signal = np.dot(input_L2, Gw_L2)
                neuron_L2_sort = np.argsort(-neuron_L2_signal, 0)
                neuron_num_L2 = neuron_L2_sort[0]
                '''weight change'''
                if neuron_num_L2 != input_label:
                    for label_input_l2 in range(0, 100):
                        if input_L2[label_input_l2] == 1:
                            W0 = Gw_L2[int(label_input_l2), int(input_label)]
                            # weight_change = ap*exp(-bp*W0)
                            weight_change = 0.1
                            W1 = W0 + weight_change
                            if W1 > 1:
                                W1 = 1
                            Gw_L2[int(label_input_l2), int(input_label)] = W1
                Gw_L2 *= 0.99999999
            '''Accuracy Rate Test'''
            correct = snn_2layer_test(test_images, test_labels, Gw_L1, Gw_L2, vth_neuron_l1)
            print correct
            # Acc[Batch + Epoch * (Sample_num / Batch_num)] = snn_2layer_test(test_images, test_labels, Gw_L1_A, Gw_L1_B, Gw_L1_C, Gw_L1_D, Gw_L1_E, Gw_L2)
        Gw_L1 *= 0.999
        Gw_L2 *= 0.999

if __name__ == '__main__':
    run()
