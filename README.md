# Neural-Netwrok-implementation-from-scratch
Implementation of the multi layer neural network from scratch (in python)

## How to use it 

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
x, y = x.astype(np.int32), y.astype(np.int32)   # converting x, y to int32 type

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
