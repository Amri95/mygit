# encoding: utf-8
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
import shelve

from MNIST_process import load_train_images,load_train_labels,load_test_images,load_test_labels

def run():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    # test_labels = load_test_labels()

    Batch_num=600
    Sample_num=60000
    Epoch_N=3

    Gw = mat(zeros((784, 10)))

    sequence_in=np.random.permutation(Sample_num)

    for Epoch in range(0,Epoch_N):
        print Epoch
        for n_batch in range(0,Sample_num/Batch_num-1):
            for n_sample in range(n_batch*Batch_num,(n_batch+1)*Batch_num-1):
                input_sample = mat(train_images[sequence_in[n_sample]])
                input_label=int(mat(train_labels[sequence_in[n_sample]]))
                input_sample = test_images[1].reshape(784, order='c')
                input_sample =(input_sample > 0.5).astype(np.int_)# binaryzation
                output_neuron=input_sample * Gw
                fired_neuron=int(np.argmax(output_neuron))
                if fired_neuron!=input_label:
                    input_sample=input_sample.reshape(-1,1)
                    Gw[:,input_label] = input_sample*0.1+Gw[:,input_label]
                # else:
                #     print fired_neuron, input_label





    # print sequence_in
    # output_signal=input_signal * Gw
    # print input_signal
    #
    # binary_image=(input_signal > 0.5).astype(np.int_)# binaryzation
    # print binary_image

if __name__ == '__main__':
    run()