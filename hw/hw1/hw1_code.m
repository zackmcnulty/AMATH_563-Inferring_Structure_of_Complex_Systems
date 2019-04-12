% Zachary McNulty
% AMATH 563: Inferring Structure of Complex Systems
% HW 1

%% Part 0: Loading the MNIST Data
clear all; close all; clc;

% load_MNIST_file is a helper function that helps deal with the formatting
% of these data files. I take the tranpose to make images/labels in the
% rows
A_train_images = load_MNIST_file("input_files/train-images-idx3-ubyte" ,"image").';
B_train_labels = load_MNIST_file("input_files/train-labels-idx1-ubyte" , "label").';
A_test_images = load_MNIST_file("input_files/t10k-images-idx3-ubyte" ,"image").';
B_test_labels = load_MNIST_file("input_files/t10k-labels-idx1-ubyte" , "label").';

% smaller datasets for testing
train_size = 60000; % max is 60000
test_size = 10000; % max is 10000
A_train_images = A_train_images(1:train_size, :);
A_test_images = A_test_images(1:test_size, :);
B_test_labels = B_test_labels(1:test_size, :);
B_train_labels = B_train_labels(1:train_size, :);

%% Part 1: Mapping between images and digits.
close all; clc;

% dimensions of A and B do not line up for AX = b? should we store the
% images and labels as rows? Thats what I have done for now.

% Standard MATLAB backslash: AX = b

% X is our classifier. Note that as each row of A is an image and each row
% of B its corresponding label, we have (image i) X = (label i) so X tells
% us how to weight weight each pixel. Furthermore, (image i) (colum X_j) =
% B_ij --> tells us how much image i resembles digit j
X_train = A_train_images \ B_train_labels; 

[predicted_labels_backslash, error_backslash] = ...
    predict_labels(X_train, A_test_images, B_test_labels, 'backslash');
error_backslash

%% matrix L2 Norm (psuedoinverse)
close all; clc; clearvars -except A_test_images A_train_images B_test_labels B_train_labels

X = pinv(A_train_images)*B_train_labels;
[predicted_labels_2norm, error_2norm] = ...
    predict_labels(X, A_test_images, B_test_labels, '2 norm');
error_2norm

%% matrix fro norm (L2 norm element by element)
% close all; clc; clearvars -except A_test_images A_train_images B_test_labels B_train_labels
% clear cvx_problem;
% 
% m = size(A_train_images, 2);
% n = size(B_train_labels, 2);
% cvx_begin
%     variable X(m,n)
%     minimize norm(A_train_images*X - B_train_labels,'fro')
% cvx_end
% 
% [predicted_labels_fro_norm, error_fro_norm] = ...
%     predict_labels(X, A_test_images, B_test_labels, 'fro norm');
% error_fro_norm

%% Part 2: Promoting Sparsity & Dimensionality Reduction


% Lasso
close all; clc; clearvars -except A_test_images A_train_images B_test_labels B_train_labels

%% L1 norm across all elements
close all; clc; clearvars -except A_test_images A_train_images B_test_labels B_train_labels
clear cvx_problem;

% Frobenius norm is the sum of the squared entries in the matrix vs the
% matrix 2-norm (i.e. norm(X, 2)) which gives the max singular value
lambda = 0.01;
lambda = 100;
m = size(A_train_images, 2);
n = size(B_train_labels, 2);
cvx_begin
    variable X(m,n)
    minimize norm(A_train_images*X - B_train_labels, 'fro') + lambda*sum(sum(abs(X)))
cvx_end

[predicted_labels_1norm, error_1norm] = ...
    predict_labels(X, A_test_images, B_test_labels, '1 norm');
error_1norm



%% Part 3: Assessing Low-Rank approximation of Data




%% Part 4: Analysis on single digits at a time