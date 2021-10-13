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




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>bp</th>
      <th>s1</th>
      <th>s2</th>
      <th>s3</th>
      <th>s4</th>
      <th>s5</th>
      <th>s6</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.038076</td>
      <td>0.050680</td>
      <td>0.061696</td>
      <td>0.021872</td>
      <td>-0.044223</td>
      <td>-0.034821</td>
      <td>-0.043401</td>
      <td>-0.002592</td>
      <td>0.019908</td>
      <td>-0.017646</td>
      <td>151.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.001882</td>
      <td>-0.044642</td>
      <td>-0.051474</td>
      <td>-0.026328</td>
      <td>-0.008449</td>
      <td>-0.019163</td>
      <td>0.074412</td>
      <td>-0.039493</td>
      <td>-0.068330</td>
      <td>-0.092204</td>
      <td>75.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.085299</td>
      <td>0.050680</td>
      <td>0.044451</td>
      <td>-0.005671</td>
      <td>-0.045599</td>
      <td>-0.034194</td>
      <td>-0.032356</td>
      <td>-0.002592</td>
      <td>0.002864</td>
      <td>-0.025930</td>
      <td>141.0</td>
    </tr>
  </tbody>
</table>
</div>



# Gradient Descent 

## 1. Set up a train test split

In the cell below, please create a train test split for this dataset, setting `target` as the response variable and all other columns as independent variables.
* Set the random state to 2021


```python
X = df.drop('target', axis = 1)
y = df.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2021)
```

## 2. Initialize an SGDRegressor model

Now, initialize an `SGDRegressor` model.
* Set the random state to 2021


```python
model = SGDRegressor(random_state=2021)
```

## 3. Fit the model

In the cell below, fit the model to the training data.


```python
model.fit(X_train, y_train)
```

    /Users/joel/opt/anaconda3/envs/python3/lib/python3.9/site-packages/sklearn/linear_model/_stochastic_gradient.py:1225: ConvergenceWarning: Maximum number of iteration reached before convergence. Consider increasing max_iter to improve the fit.
      warnings.warn("Maximum number of iteration reached before "





    SGDRegressor(random_state=2021)



At this point in the program, you may have become accustomed to ignoring pink warning messages –– mostly because `pandas` returns many unhelpful warning messages. 

It is important to state that, generally, you should not default to ignoring warning messages. In this case the above pink warning message is quite informative!

The above warning message tells us that our model failed to converge. This means that our model did not find the minima of the cost curve, which we usually want! The warning offers the suggestion:
> *"Consider increasing max_iter to improve the fit."*


`max_iter` is an adjustable hyperparameter for the `SGDRegressor` model.

Let's zoom in on this parameter for a second.


```python
from src.questions import *
question_4.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto')), RadioButtons(layout=Layout(flex_flow='column…


## 5. Update the max_iter

In the cell below, initialize a new `SGDRegessor` model with `max_iter` set to 10,000. 
* Set the random state to 2021


```python
model = SGDRegressor(max_iter=10000, random_state=2021)
model.fit(X_train, y_train)
```




    SGDRegressor(max_iter=10000, random_state=2021)



The model converged! This tells us that the model just needed to run for longer to reach the minima of the cost curve. 


But how do you find the necessary number of iterations? 

In the cell below, we have written some code that shows you how to find the required number of iterations programmatically. This code is mostly being provided in case you ever need it, so don't stress if it feels intimidating!

In truth, there is a different hyperparameter we tend to use to help our models converges. 

### Let's zoom in on the *learning rate*!

## 6. What is the default setting for alpha (learning rate) for the `SGDRegressor`? - Multi choice


```python
# Run this cell unchanged
question_6.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


## 7. Update the alpha to .01 and set the max_iter to 2000


```python
model = SGDRegressor(max_iter=2000, alpha=0.01, random_state=2021)
model.fit(X_train, y_train)
```




    SGDRegressor(alpha=0.01, max_iter=2000, random_state=2021)



## 8. The model converged - True or False


```python
# Run this cell unchanged
question_8.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


## 9. Select the answer that best describes how alpha impacts a model's fit


```python
# Run this cell unchanged
question_9.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


# Linear Algebra 

## 10. When finding the dot product for two vectors, the length of the vectors must be the same.


```python
# Run this cell unchanged
question_10.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


## 11. Please select the solution for the dot product of the following vectors.

$vector_1 = \begin{bmatrix} 10&13\\ \end{bmatrix}$

$vector_2= \begin{bmatrix} -4&82\\ \end{bmatrix}$



```python
# Run this cell unchanged
question_11.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


## 12. How do you turn a list into a numpy array?


```python
# Run this cell unchanged

question_12.display()
```


    VBox(children=(Output(layout=Layout(bottom='5px', width='auto'), outputs=({'output_type': 'display_data', 'met…


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
vector_1 = np.array(vector_1)
vector_2 = np.array(vector_2)

np.dot(vector_1.T, vector_2)
```




    array([[-3.31869275],
           [-7.80043543],
           [-9.35675419],
           [-0.13185929],
           [ 1.207176  ]])


