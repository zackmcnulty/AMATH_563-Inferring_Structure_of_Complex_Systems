% Zachary McNulty
% AMATH 563: Inferring Structure of Complex Systems
% HW 1

%% Loading the MNIST Data
clear all; close all; clc;

A_train_images = load_MNIST_file("input_files/train-images-idx3-ubyte" ,"image");
B_train_labels = load_MNIST_file("input_files/train-labels-idx1-ubyte" , "label");
A_test_images = load_MNIST_file("input_files/t10k-images-idx3-ubyte" ,"image");
B_test_labels = load_MNIST_file("input_files/t10k-labels-idx1-ubyte" , "label");

%%