# Neural-Netwrok-implementation-from-scratch
Implementation of the multi layer neural network from scratch (in python)

## How to Train the model

##### Step 1: Import the dl_toolkit
```python
from dl_toolkit import *

# allowed hyperparameter values are as follows
allowed_activation_function = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
allowed_weight_init = ['zero', 'random', 'he', 'xavier']
allowed_optimizer = ['gradient_descent', 'gradient_descent_with_momentum', 'NAG', 'AdaGrad', 'RMSProp', 'Adam']
allowed_regularization = ['l1', 'l2']

```
##### Step 2: Initializing the model hyperparameters 
```python3

# initilaize kwargs dictionary to send additional parameters
kwargs = {
    'beta' : 0.9,             # for momentum
    'gamma' : 0.9,            # for RMSProp and Adam
    'epsilon' : 10**-8,       # for optimizers (adding into denominator to prevent divide by 0 error)
    'lamda' : 0               # regularization term
}  


# Initializing the model and assigning hyperparameters
mnn = MLPClassifier(layers = [784, 256, 128, 64, 10],       # 784 is input features and 10 is no. of classes
                    learning_rate = 0.001, 
                    activation_function = "tanh", 
                    optimizer = 'RMSProp',
                    weight_init = "xavier", 
                    regularization = 'l2',
                    batch_size = 64, 
                    num_epochs = 100,
                    dropouts = 0.2, 
                    **kwargs
                    )

```
##### Step 3: Downloading the dataset (MNIST in this example)
```python3

# downloading the MNIST dataset from sklearn
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

x, y = mnist.data, mnist.target
x, y = x.astype(np.int32).to_numpy(), y.astype(np.int32).to_numpy()   # converting x, y to int32 numpy array

# splitting the data in train and test (Stratified sampling)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, stratify = y, random_state = 42)

```

##### Step 4: Train (i.e. run) the model
```python3
# running the model
mnn.fit(X_train, y_train)

# OR mnn.fit(X_train, y_train, X_test, y_test)  # if you want testing accuracy and entropy also after each epoch

```


<hr>

## Some basic functions are as follows

1. __init__(layers, learning_rate, activation_function, optimizer, weight_init, regularization, batch_size, num_epochs, dropouts, **kwargs)

    - layers: A list of number of neurons in each layer, starting from input layer,
    intermediate hidden layers and output layer. For example, a list consisting of [784, 200,
    50, 10] means the number of input features to the NN is 784, and it consists of 2 hidden
    layers with 200, 50 neurons in hidden layers 1 and 2, and it has 10 neurons in the output
    layer. Since we are using MNIST dataset, the number of neurons in the input and output
    layer will remain constant i.e. 784 for input and 10 for output. No default value is there,
    this is a necessary argument.
    - learning_rate: Learning rate of the neural network. Default value = 1e-5.
    - activation_function: A string containing the name of the activation function to be
    used in the hidden layers. For the output layer use Softmax activation function. Default
    value = “relu”.
    - optimizer: A string containing the name of the optimizer to be used by the network.
    Default value = “gradient_descent”.
    - Weight_init: “random”, “he” or “xavier”: String defining type of weight initialization
    used by the network. Default value = “random”.
    - Regularization: A string containing the type of regularization. The accepted values
    can be “l1”, “l2”, “batch_norm”, and “layer_norm” . The default value is “l2”.
    - Batch_size: An integer specifying the mini batch size. By default the value is 64.
    - Num_epochs: An integer with a number of epochs the model should be trained for.
    - dropout: An integer between 0 to 1 describing the percentage of input neurons to be
    randomly masked.
    - **kwargs: A dictionary of additional parameters required for different optimizers. By
    default it is None, however, you must initialize different parameters of optimizers with
    some valid input value for convergence. (This parameter will not be used in the test file)
    
    Output: (void)

2. fit(X, Y)
    - X: a numpy array of shape (num_examples, num_features).
    - Y: a numpy array of shape (num_examples): This array contains the classification labels of the task.
    
    Output: (void)
    
    Note: This function should log the loss after some minibatches (you can choose this
    arbitrarily) and after the complete epoch.
    
3. predict(X):
    - X: a numpy array of shape (num_examples, num_features)
    
    Output: numpy array of shape (num_examples) with classification labels of each class.
    
4. predict_proba(X):
    - X: a numpy array of shape (num_examples, num_features)
    
    Output: numpy array of shape (num_examples, num_classes): This 2d matrix contains the probabilities of each class for all the examples.
    
5. get_params():
    
    Output: An array of 2d numpy arrays. This array contains the weights of the model.
    
6. score(X, y):
    - X: a numpy array of shape (num_examples, num_features): This 2d matrix contains the complete dataset.
    - Y: a numpy array of shape (num_examples): This array contains the classification labels of the task.
    
    Output: (float) Classification accuracy given X and y.
