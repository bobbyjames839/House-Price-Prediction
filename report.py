import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error


data = pd.read_csv('train.csv')

print(data['SalePrice'].mean())

numerical_data = data.select_dtypes(include = ['number'])
'''
for item in data.columns:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.hist(data[item].dropna(), bins = 100)
  ax.set_xlabel(item)
  ax.set_ylabel('Freq')
  plt.show()

for item in numerical_data.columns:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(numerical_data['SalePrice'], numerical_data[item])
  ax.set_xlabel('Sale Price')
  ax.set_ylabel(item)
  plt.show()'''

non_numerical_data = data.select_dtypes(exclude = ['number'])
'''
for column in non_numerical_data.columns:
    unique_values = non_numerical_data[column].dropna().unique()
    
    plt.figure(figsize=(12, 8))
    
    for value in unique_values:
        subset = numerical_data[data[column] == value]['SalePrice'].dropna()
        plt.hist(subset, bins=200, alpha=0.5, label=f'{column} = {value}', histtype='step', linewidth=2)

    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Sale Price by {column}')
    plt.legend()
    plt.show()'''

'''null_features = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

null_data = data.copy()
for item in null_features:
    null_data[item] = null_data[item].astype(str).fillna('NaN')

for item in null_features:
    fig = plt.figure()
    ax = fig.add_subplot()
    sns.boxplot(x=null_data['SalePrice'], y=null_data[item], ax=ax)
    ax.set_xlabel('Sale Price')
    ax.set_ylabel(item)
    plt.show()'''

'''correlation_matrix = numerical_data.corr()

plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, 
            xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Correlation Matrix of Numerical Features')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

saleprice_correlation = correlation_matrix['SalePrice'].sort_values(ascending=False)
print(saleprice_correlation)'''


nominal_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
ordinal_features = ['Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
'''
label_encoder = LabelEncoder()
label_data = pd.DataFrame()
for col in ordinal_features:
    label_data[col] = label_encoder.fit_transform(data[col])

for i in range(len(ordinal_features)):
  ordinal_comparison_data = pd.concat([data[ordinal_features].iloc[:, i], label_data.iloc[:, i]], axis = 1)
  ordinal_comparison_data = ordinal_comparison_data.drop_duplicates()
  print(ordinal_comparison_data)
  print(ordinal_features[i])'''

'''
for column in non_numerical_data.columns:
    unique_values = non_numerical_data[column].dropna().unique()
    
    plt.figure(figsize=(12, 8))
    
    for value in unique_values:
        subset = numerical_data[data[column] == value]['SalePrice'].dropna()
        plt.hist(subset, bins=200, alpha=0.5, label=f'{column} = {value}', histtype='step', linewidth=2)

    plt.xlabel('Sale Price')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Sale Price by {column}')
    plt.legend()
    plt.show()'''

for column in non_numerical_data.columns:
    unique_values = non_numerical_data[column].dropna().unique()
    n_unique_values = len(unique_values)
    ncols = min(n_unique_values, 3)  # Number of columns in the grid
    nrows = (n_unique_values + ncols - 1) // ncols  # Number of rows in the grid

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    for ax, value in zip(axes, unique_values):
        subset = numerical_data[data[column] == value]['SalePrice'].dropna()
        ax.hist(subset, bins=200, alpha=0.5, label=f'{column} = {value}', histtype='step', linewidth=2)
        ax.set_xlabel('Sale Price')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Histogram of Sale Price by {column} = {value}')
        ax.legend()

    # Hide any unused subplots
    for i in range(n_unique_values, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()