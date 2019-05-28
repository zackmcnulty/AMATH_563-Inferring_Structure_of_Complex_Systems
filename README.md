# AMATH 563: Inferring Structure of Complex Systems

## Setting Up CVX
- Open MATLAB
- In the MATLAB console/terminal type:
``` 
	cd cvx-w64/cvx
	cvx_setup
```


# Lecture Topics

### Regularization and Promoting Sparsity
- Solving overdetermined & underdetermined systems
- L1 Norm vs L2 Norm
- Gradient descent, LASSO, and QR Decomposition
- Regularization and the benefits of sparsity


### Model Discovery

- SINDy: Discovering models for ODEs & PDEs using data
- Time-Embeddings: Discovering hidden (latent) variables
- Pareto Frontiers
- Cross-validation
- k fold cross validation
- Information Criteria (Bayesian,KL, AIC) 

### Data Assimilation
- Kalman Filters & Model auto-correction


### Dynamic Mode Decomposition
- Model discovery in low-data settings 

### Clustering and Classification
- Unsupervised Methods
    - K-Means
    - Hierarchical Clustering
- Supervised Methods
    - Support Vector Machines
    - Classification/Decision Trees

### Neural Networks
- Feedforward neural networks
- Convolutional neural networks
- Regularization in Neural Networks


### Randomized Linear Algebra
- Extracting low-rank structure in Big-data

## Homework Topics

##### HW1: Regression and Sparsity
- MNIST database (hand-drawn digits)
- Digit recognition using linear regression
- Promoting Sparsity with L1 Norm
- MATLAB optimization with CVX

##### HW2: Model Discovery
- Nonlinear sparse regression
- Discovery of dynamical systems through data
- Model Assessment: Information Criterion and KL Divergence

##### HW3: Clustering and Classification with Neural Networks
- Lorenz System, Reaction-Diffusion Equations, kuramoto sivashinsky equations
- Future State Predictions
