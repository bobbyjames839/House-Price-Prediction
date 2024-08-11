import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
data = pd.read_csv('train.csv')

#display columns with more than 500 null values
'''for item in data.columns:
  nullitems = data[item].isna().sum()
  if nullitems > 500:
    print(item)'''

#histograms of all the features and label
'''for item in data.columns[]:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.hist(data[item].dropna(), bins = 100)
  ax.set_xlabel(item)
  ax.set_ylabel('Freq')
  plt.show()'''

null_features = ['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']

#plot a boxplot of the null features to see how much they affect the label
'''for item in null_features:
  fig = plt.figure()
  ax = fig.add_subplot()
  sns.boxplot(x=data['SalePrice'], y=data[item])
  ax.set_xlabel('Sale Price')
  ax.set_ylabel(item)
  plt.show()'''

data = data.drop(columns=null_features)
data = data.drop(columns=['Id'])
numerical_data = data.select_dtypes(include=['number'])

#plot all of the numerical data against the label
'''for item in numerical_data:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(data['SalePrice'], data[item])
  ax.set_xlabel('Sale Price')
  ax.set_ylabel(item)
  plt.show()'''

#show the correlations between the numerical data and the label
'''correlation_matrix = numerical_data.corr()
saleprice_correlation = correlation_matrix['SalePrice'].sort_values(ascending=False)
print(saleprice_correlation)'''

non_numerical_data = data.select_dtypes(exclude=['number'])

#determine what is ordinal and what is nominal out of the non-numerical values 
'''for item in non_numerical_data.columns:
    fig = plt.figure()
    ax = fig.add_subplot()
    x = data['SalePrice']
    y = non_numerical_data[item]
    mask = ~y.isnull()
    ax.scatter(x[mask], y[mask])
    ax.set_xlabel('Sale Price')
    ax.set_ylabel(item)
    plt.show()'''

nominal_features = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
ordinal_features = ['Utilities', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'Functional', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

#visualising the effect of label encoding on the ordinal data 
'''label_encoder = LabelEncoder()
label_data = pd.DataFrame()
for col in ordinal_features:
    label_data[col] = label_encoder.fit_transform(data[col])

for i in range(len(ordinal_features)):
  ordinal_comparison_data = pd.concat([data[ordinal_features].iloc[:, i], label_data.iloc[:, i]], axis = 1)
  ordinal_comparison_data = ordinal_comparison_data.drop_duplicates()
  print(ordinal_comparison_data)
  print(ordinal_features[i])'''

#custom encoding the ordinal data

for item in data[ordinal_features]:
  print(data[item].drop_duplicates())

custom_ordinal_data = data[ordinal_features]

mapping_dict = {
    'Utilities': {'AllPub': 1, 'NoSeWa': 0},
    'ExterQual': {'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0},
    'ExterCond': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'BsmtQual': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'NaN': 0},
    'BsmtCond': {'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NaN': 0},
    'BsmtExposure': {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NaN': 0},
    'BsmtFinType1': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NaN': 0},
    'BsmtFinType2': {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NaN': 0},
    'HeatingQC': {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po': 0},
    'CentralAir': {'Y': 1, 'N': 0},
    'KitchenQual': {'Ex': 3, 'Gd': 2, 'TA': 1, 'Fa': 0},
    'Functional': {'Typ': 6, 'Min1': 5, 'Min2': 4, 'Mod': 3, 'Maj1': 2, 'Maj2': 1, 'Sev': 0},
    'GarageFinish': {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NaN': 0},
    'GarageQual': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NaN': 0},
    'GarageCond': {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NaN': 0},
    'PavedDrive': {'Y': 2, 'P': 1, 'N': 0}
}

for column, mapping in mapping_dict.items():
    custom_ordinal_data[column] = custom_ordinal_data[column].map(mapping)

#nominal data 
nominal_data = data[nominal_features]
for item in nominal_data.columns:
  print(nominal_data[item].nunique())
  print(item)


nominal_one_hot_features = ['MSZoning', 'LotShape', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition']
one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded_data_v1 = one_hot_encoder.fit_transform(data[nominal_one_hot_features])
one_hot_encoded_data = pd.DataFrame(one_hot_encoded_data_v1, columns=one_hot_encoder.get_feature_names_out(nominal_one_hot_features))




predicting_data = pd.concat([custom_ordinal_data, numerical_data, one_hot_encoded_data], axis=1)

for item in predicting_data.columns:
  print(predicting_data[item].isnull().sum())
  print(item)

predicting_data = predicting_data.fillna(0)
predicting_data = predicting_data.drop(columns=['SalePrice'])


x = predicting_data
y = data['SalePrice']

rf = RandomForestRegressor(random_state = 40)
lr = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

param_dist = {'bootstrap': [True, False],
  'max_depth': [10, 20, 30,None],
  'max_features': ['auto', 'sqrt'],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2, 5, 10],
  'n_estimators': [100, 150, 200, 400,],
  'max_samples': [0.5, 0.75, 1.0], 
  'warm_start': [True, False]  
 }

rf_grid = GridSearchCV(estimator=rf, param_grid=param_dist, cv=2, verbose=2, n_jobs=-1)
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_dist, n_iter=100, cv=3, verbose=2, random_state=40, n_jobs=-1)

lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

#print(f'Best Hyperparameters: {lr.best_params_}')
print(f'Error: {rmse}')

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(y_test, y_pred)
ax.set_xlim(0,500000)
ax.set_ylim(0,500000)
ax.plot([0, 500000], [0, 500000], color='red', linestyle='--')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()

# Feature importances
feature_importances = rf_grid.best_estimator_.feature_importances_
print(feature_importances)
features = x.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Group by the original feature names and sum their importances
original_features = [f.split('_')[0] for f in importances_df['Feature']]
importances_df['OriginalFeature'] = original_features
grouped_importances_df = importances_df.groupby('OriginalFeature').sum().reset_index()

# Sort the grouped importances
grouped_importances_df = grouped_importances_df.sort_values(by='Importance', ascending=False)

# Plot grouped feature importances
plt.figure(figsize=(10, 8))
plt.barh(grouped_importances_df['OriginalFeature'], grouped_importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Grouped Feature Importances')
plt.gca().invert_yaxis()
plt.show()