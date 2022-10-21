
# Gradient Descent

Gradient descent is an iterative machine learning optimization algorithm to reduce the cost function so that we have models that makes accurate predictions.

Cost function(C) or Loss function measures the difference between the actual output and predicted output from the model. Cost function are a convex function.
![Logo](https://latex.codecogs.com/svg.image?\theta&space;_{j}&space;:=&space;\theta&space;_{j}&space;-&space;\alpha&space;\frac{\partial&space;}{\partial&space;\theta&space;_{j}}J\left&space;(&space;\theta&space;&space;\right&space;))
There’s an important parameter α which scales the gradient and thus controls the step size. In machine learning, it is called learning rate and have a strong influence on performance.

1. The smaller learning rate the longer GD converges, or may reach maximum iteration before reaching the optimum point
2. If learning rate is too big the algorithm may not converge to the optimal point (jump around) or even to diverge completely.

#### In summary, Gradient Descent method’s steps are:

1. choose a starting point (initialisation)
2. calculate gradient at this point
3. make a scaled step in the opposite direction to the gradient (objective: minimise)
4. repeat points 2 and 3 until one of the criteria is met:
* maximum number of iterations reached
* step size is smaller than the tolerance (due to scaling or a small gradient).


### This function takes 5 parameters:
1. starting point - in our case, we define it manually but in practice, it is often a random initialisation
2. gradient function - has to be specified before-hand
3. learning rate - scaling factor for step sizes
4. maximum number of iterations
5. tolerance to conditionally stop the algorithm (in this case a default value is 0.01)
#### 
#### 
The animation below shows steps taken by the GD algorithm for learning rates of 0.1 and 0.8. As you see, for the smaller learning rate, as the algorithm approaches the minimum the steps are getting gradually smaller. For a bigger learning rate, it is jumping from one side to another before converging.
![Logo](https://miro.medium.com/max/720/1*v5bc1TzeMKpzTAgorjOoHQ.gif)

### Different types of Gradient descents are

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini batch Gradient Descent

### Batch Gradient Descent
In batch gradient we use the entire dataset to compute the gradient of the cost function for each iteration of the gradient descent and then update the weights.

Since we use the entire dataset to compute the gradient convergence is slow.

If the dataset is huge and contains millions or billions of data points then it is memory as well as computationally intensive.
#### Advantages of Batch Gradient Descent
* Theoretical analysis of weights and convergence rates are easy to understand
#### Disadvantages of Batch Gradient Descent
* Perform redundant computation for the same training example for large datasets
* Can be very slow and intractable as large datasets may not fit in the memory
* As we take the entire dataset for computation we can update the weights of the model for the new data

### Stochastic Gradient descent
In stochastic gradient descent we use a single datapoint or example to calculate the gradient and update the weights with every iteration.

we first need to shuffle the dataset so that we get a completely randomized dataset. As the dataset is randomized and weights are updated for each single example, update of the weights and the cost function will be noisy jumping all over the place as shown below

Random sample helps to arrive at a global minima and avoids getting stuck at a local minima.

Learning is much faster and convergence is quick for a very large dataset.

#### Advantages of Stochastic Gradient Descent
* Learning is much faster than batch gradient descent
* Redundancy is computation is removed as we take one training sample at a time for computation
* Weights can be updated on the fly for the new data samples as we take one training sample at a time for computation
#### Disadvantages of Stochastic Gradient Descent
* As we frequently update weights, Cost function fluctuates heavily

### Mini Batch Gradient descent
Mini-batch gradient is a variation of stochastic gradient descent where instead of single training example, mini-batch of samples is used.

Mini batch gradient descent is widely used and converges faster and is more stable.

Batch size can vary depending on the dataset.

As we take a batch with different samples,it reduces the noise which is variance of the weight updates and that helps to have a more stable converge faster.
#### Advantages of Min Batch Gradient Descent
* Reduces variance of the parameter update and hence lead to stable convergence
* Speeds the learning
* Helpful to estimate the approximate location of the actual minimum
#### Disadvantages of Mini Batch Gradient Descent
* Loss is computed for each mini batch and hence total loss needs to be accumulated across all mini batches


1. The smaller learning rate the longer GD converges, or may reach maximum iteration before reaching the optimum point
2. If learning rate is too big the algorithm may not converge to the optimal point (jump around) or even to diverge completely.

#### In summary, Gradient Descent method’s steps are:

1. choose a starting point (initialisation)
2. calculate gradient at this point
3. make a scaled step in the opposite direction to the gradient (objective: minimise)
4. repeat points 2 and 3 until one of the criteria is met:
* maximum number of iterations reached
* step size is smaller than the tolerance (due to scaling or a small gradient).


### This function takes 5 parameters:
1. starting point - in our case, we define it manually but in practice, it is often a random initialisation
2. gradient function - has to be specified before-hand
3. learning rate - scaling factor for step sizes
4. maximum number of iterations
5. tolerance to conditionally stop the algorithm (in this case a default value is 0.01)
### Different types of Gradient descents are

* Batch Gradient Descent
* Stochastic Gradient Descent
* Mini batch Gradient Descent

### Batch Gradient Descent
In batch gradient we use the entire dataset to compute the gradient of the cost function for each iteration of the gradient descent and then update the weights.

Since we use the entire dataset to compute the gradient convergence is slow.

If the dataset is huge and contains millions or billions of data points then it is memory as well as computationally intensive.
#### Advantages of Batch Gradient Descent
* Theoretical analysis of weights and convergence rates are easy to understand
#### Disadvantages of Batch Gradient Descent
* Perform redundant computation for the same training example for large datasets
* Can be very slow and intractable as large datasets may not fit in the memory
* As we take the entire dataset for computation we can update the weights of the model for the new data

### Stochastic Gradient descent
In stochastic gradient descent we use a single datapoint or example to calculate the gradient and update the weights with every iteration.

we first need to shuffle the dataset so that we get a completely randomized dataset. As the dataset is randomized and weights are updated for each single example, update of the weights and the cost function will be noisy jumping all over the place as shown below

Random sample helps to arrive at a global minima and avoids getting stuck at a local minima.

Learning is much faster and convergence is quick for a very large dataset.

#### Advantages of Stochastic Gradient Descent
* Learning is much faster than batch gradient descent
* Redundancy is computation is removed as we take one training sample at a time for computation
* Weights can be updated on the fly for the new data samples as we take one training sample at a time for computation
#### Disadvantages of Stochastic Gradient Descent
* As we frequently update weights, Cost function fluctuates heavily

### Mini Batch Gradient descent
Mini-batch gradient is a variation of stochastic gradient descent where instead of single training example, mini-batch of samples is used.

Mini batch gradient descent is widely used and converges faster and is more stable.

Batch size can vary depending on the dataset.

As we take a batch with different samples,it reduces the noise which is variance of the weight updates and that helps to have a more stable converge faster.
#### Advantages of Min Batch Gradient Descent
* Reduces variance of the parameter update and hence lead to stable convergence
* Speeds the learning
* Helpful to estimate the approximate location of the actual minimum
#### Disadvantages of Mini Batch Gradient Descent
* Loss is computed for each mini batch and hence total loss needs to be accumulated across all mini batches
