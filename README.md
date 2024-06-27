# Linear-Regression--Boston-House-Value-Prediction


This project demonstrates the use of linear regression to predict the median value of owner-occupied homes in the Boston area using various predictors.

## Dataset

The dataset contains the following columns:

1. **crim**: per capita crime rate by town.
2. **zn**: proportion of residential land zoned for lots over 25,000 sq.ft.
3. **indus**: proportion of non-retail business acres per town.
4. **chas**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5. **nox**: nitrogen oxides concentration (parts per 10 million).
6. **rm**: average number of rooms per dwelling.
7. **age**: proportion of owner-occupied units built prior to 1940.
8. **dis**: weighted mean of distances to five Boston employment centres.
9. **rad**: index of accessibility to radial highways.
10. **tax**: full-value property-tax rate per 10,000 dollars.
11. **ptratio**: pupil-teacher ratio by town.
12. **black**: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
13. **lstat**: lower status of the population (percent).
14. **medv**: median value of owner-occupied homes in $1000s.

## Steps to Run the Analysis

### 1. Import Necessary Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
```

### 2. Load the Data

```python
data = pd.read_csv('path/to/your/boston.csv')  # Update the path to your dataset
data.head()
```

### 3. Data Visualization and Correlation Analysis

```python
fig = plt.figure(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True)
plt.show()
```

### 4.Select Relevant Features Based on Correlation Analysis

```python
data2 = data[['indus', 'rm', 'lstat', 'medv']]
```

### 5. Check for Linearity

```python
fig = plt.figure(figsize=(15, 15))
plt.subplot(2, 3, 1)
plt.scatter(data2['indus'], data2['medv'])
plt.subplot(2, 3, 2)
plt.scatter(data2['rm'], data2['medv'])
plt.subplot(2, 3, 3)
plt.scatter(data2['lstat'], data2['medv'])
plt.show()
```

### 6. Split the Data into Training and Testing Sets

```python
X = pd.DataFrame(data2[['indus', 'rm', 'lstat']])
y = pd.DataFrame(data2['medv'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

### 7. Train the Linear Regression Model

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

### 8. Model Evaluation

```python
y_pred = model.predict(X_test)

# Mean Absolute Error
mae = metrics.mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Mean Squared Error
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Root Mean Squared Error
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

# R-Squared
r2 = metrics.r2_score(y_test, y_pred)
print("R-Squared:", r2)
```

### 9. Calculate Adjusted R-Squared

```python
n = len(X_test)
k = X_test.shape[1]
adjusted_r2 = 1 - ((1 - r2) * (n - 1)) / (n - k - 1)
print("Adjusted R-Squared:", adjusted_r2)
```

