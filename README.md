# Gradient Descent and Linear Algebra Review

In this notebook, we review:
* Referencing sklearn documentation
* `learning rate` - a standard hyperparameter for gradient descent algorithms
    * How learning rate impacts a model's fit.
* Introductory elements of linear algebra.
    * Vector shape
    * Dot products
    * Numpy arrays

In the next cell we import the necessary packages for the notebook and load in a dataset containing information about diatebetes patients. 

**Data Understanding**

The documentation for this dataset provides the following summary:

> *"Ten baseline variables, age, sex, body mass index, average blood
pressure, and six blood serum measurements were obtained for each of n =
442 diabetes patients, as well as the response of interest, a
quantitative measure of disease progression one year after baseline."*


```python
# Sklearn's gradient descent linear regression model
from sklearn.linear_model import SGDRegressor

# Pandas and numpy
import pandas as pd
import numpy as np

# Train test split
from sklearn.model_selection import train_test_split

# Load Data
from sklearn.datasets import load_diabetes
data = load_diabetes()
df = pd.DataFrame(data['data'], columns=data['feature_names'])
df['target'] = data['target']

# Jupyter configuration
%config Completer.use_jedi = False

df.head(3)
```

# Gradient Descent 

## 1. Set up a train test split

In the cell below, please create a train test split for this dataset, setting `target` as the response variable and all other columns as independent variables.
* Set the random state to 2021


```python
# Your code here
```

## 2. Initialize an SGDRegressor model

Now, initialize an `SGDRegressor` model.
* Set the random state to 2021


```python
# Your code here
```

## 3. Fit the model

In the cell below, fit the model to the training data.


```python
# Your code here
```

At this point in the program, you may have become accustomed to ignoring pink warning messages –– mostly because `pandas` returns many unhelpful warning messages. 

It is important to state that, generally, you should not default to ignoring warning messages. In this case the above pink warning message is quite informative!

The above warning message tells us that our model failed to converge. This means that our model did not find the minima of the cost curve, which we usually want! The warning offers the suggestion:
> *"Consider increasing max_iter to improve the fit."*


`max_iter` is an adjustable hyperparameter for the `SGDRegressor` model.

Let's zoom in on this parameter for a second.


```python
# Run this cell unchanged
from src.questions import *
question_4.display()
```

## 5. Update the max_iter

In the cell below, initialize a new `SGDRegessor` model with `max_iter` set to 10,000. 
* Set the random state to 2021


```python
# Your code here
```

The model converged! This tells us that the model just needed to run for longer to reach the minima of the cost curve. 


But how do you find the necessary number of iterations? 

In the cell below, we have written some code that shows you how to find the required number of iterations programmatically. This code is mostly being provided in case you ever need it, so don't stress if it feels intimidating!

In truth, there is a different hyperparameter we tend to use to help our models converges. 


```python
# Run this cell unchanged
import warnings

# Loop over a range of numbers between 1000 10,000
for i in range(1000, 10000, 500):
    # Catch the ConvergenceWarning
    with warnings.catch_warnings():
        # If a warning is produced, throw an error instead
        warnings.filterwarnings('error')
        # Place the model fit inside a try except block to catch the error
        try:
            model = SGDRegressor(max_iter=i, random_state=2021)
            model.fit(X_train, y_train)
            # If the model fits without a ConvergenceWarning stop the for loop
            break
        except Warning:
            # If the model returns a ConvergenceWarning, move to the next iteration.
            continue
            
# Print the number of iterations that allowed the model to converge
print('Max iterations needed for convergence:', i)
```

### Let's zoom in on the *learning rate*!

## 6. What is the default setting for alpha (learning rate) for the `SGDRegressor`? - Multi choice


```python
# Run this cell unchanged
question_6.display()
```

## 7. Update the alpha to .01 and set the max_iter to 2000


```python
# Your code here
```

## 8. The model converged - True or False


```python
# Run this cell unchanged
question_8.display()
```

## 9. Select the answer that best describes how alpha impacts a model's fit


```python
# Run this cell unchanged
question_9.display()
```

# Linear Algebra 

## 10. When finding the dot product for two vectors, the length of the vectors must be the same.


```python
# Run this cell unchanged
question_10.display()
```

## 11. Please select the solution for the dot product of the following vectors.

$vector_1 = \begin{bmatrix} 10&13\\ \end{bmatrix}$

$vector_2= \begin{bmatrix} -4&82\\ \end{bmatrix}$



```python
# Run this cell unchanged
question_11.display()
```

## 12. How do you turn a list into a numpy array?


```python
# Run this cell unchanged

question_12.display()
```

## 13. Please find the dot product of the following vectors


```python
vector_1 = [
               [ 0.80559827,  0.29916789,  0.39630405,  0.92797795, -0.13808099],
               [ 1.7249222 ,  1.59418491,  1.95963002,  0.64988373, -0.08225951],
               [-0.50472891,  0.74287965,  1.8927091 ,  0.33783705,  0.94361808],
               [ 0.99034854, -1.0526394 , -0.33825968, -0.40148036,  1.81821604],
               [-0.7298026 , -0.88302624,  0.49319177, -0.02758864,  0.33430167],
               [ 0.85938167, -0.71149948, -1.8434118 ,  0.89097775,  0.53842254]
                                                                                    ]


vector_2 = [
              [ 0.13288805],
              [-2.50839814],
              [-0.90620828],
              [ 0.09841538],
              [ 1.86783262],
              [ 1.98903307]
                               ]
```


```python
# Your code here
```
