import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#Load the dataset from Scikit-Learn
from sklearn.datasets import load_boston

boston_dataset = load_boston()
boston_dataset.keys()

#Invoke pandas
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
boston['MEDV'] = boston_dataset.target

#Preprocess the data
# check for missing values in all the columns
boston.isnull().sum()


#Visualization
# set the size of the figure
sns.set(rc={'figure.figsize':(8,5)})

# plot a histogram showing the distribution of the target values
sns.distplot(boston['MEDV'], bins=30)
#plt.show()

# compute the pair wise correlation for all columns
correlation_matrix = boston.corr().round(2)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
#plt.show()

plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
plt.show()
