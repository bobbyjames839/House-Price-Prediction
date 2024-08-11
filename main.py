import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_columns', None)
data = pd.read_csv('train.csv')
#print(data.columns)

features = ['LotArea', 'Neighborhood', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 'KitchenQual', 'GarageArea', 'SaleCondition']


fig, axs = plt.subplots(2, 5, figsize=(20, 20))

for i, feature in enumerate(features):
    row = i // 5
    col = i % 5
    axs[row, col].scatter(data['SalePrice'], data[feature])
    axs[row, col].set_xlabel('House Price')
    axs[row, col].set_ylabel(feature)
    axs[row, col].set_title(feature)

plt.tight_layout()
plt.show()

fig = plt.figure()
ax = fig.add_subplot()
ax.hist(data['SalePrice'], bins = 200)
ax.set_xlabel('House Price')
ax.set_ylabel('Frequency')
plt.show()


#print(data[features].info())

nomial_categories = ['Neighborhood', 'HouseStyle', 'RoofStyle']
ordinal_categories = ['KitchenQual', 'SaleCondition']
one_hot_encoder = OneHotEncoder(sparse_output=False)
label_encoder = LabelEncoder()

one_hot_encoded = one_hot_encoder.fit_transform(data[nomial_categories])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(nomial_categories))

label_encoded_df = pd.DataFrame()
for col in ordinal_categories:
    label_encoded_df[col] = label_encoder.fit_transform(data[col])

features_data = data[features].drop(nomial_categories + ordinal_categories, axis=1)

encoded_df = pd.concat([features_data, one_hot_df, label_encoded_df], axis=1)

#print(encoded_df.head())

x = encoded_df
y = data['SalePrice']

rf = RandomForestRegressor(random_state = 40)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=40)

param_dist = {'bootstrap': [True, False],
  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
  'max_features': ['auto', 'sqrt'],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2, 5, 10],
  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
  'max_samples': [0.5, 0.75, 1.0], 
  'warm_start': [True, False]  
 }

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = param_dist, n_iter=100, cv=3, verbose=2, random_state=40, n_jobs=-1)

rf_random.fit(x_train, y_train)
y_pred = rf_random.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Best Hyperparameters: {rf_random.best_params_}')
print(f'Error: {rmse}')

'''fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(y_test, y_pred)
ax.set_xlim(0,500000)
ax.set_ylim(0,500000)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
plt.show()'''