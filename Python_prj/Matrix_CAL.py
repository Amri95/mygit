# encoding: utf-8
from numpy import *
import numpy as np
import struct
import matplotlib.pyplot as plt
import pickle
import shelve
from MNIST_process import load_train_images,load_train_labels,load_test_images,load_test_labels


def run():
    # train_images = load_train_images()
    # train_labels = load_train_labels()
    # test_images = load_test_images()
    test_labels = load_test_labels()

    weight=mat(zeros((784,10)))+0.01
    input_signal= test_labels[1]

    output_signal=input_signal * weight
    print output_signal

if __name__ == '__main__':
    run()
