import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
data = pd.read_csv('train.csv')

numerical_data = data.select_dtypes(include=['number'])

'''for item in numerical_data.columns:
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.scatter(data['SalePrice'], numerical_data[item])
  ax.set_xlabel('Sale Price')
  ax.set_ylabel(item)
  plt.show()'''

numerical_categorical_features = ['MSSubClass', 'OverallQual', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

'''for item in numerical_categorical_data:
    agg_data = data.groupby(item)['SalePrice'].mean().reset_index()
    fig = plt.figure()
    ax = fig.add_subplot()
    bars = ax.bar(agg_data[item], agg_data['SalePrice'], label='Sale Price')
    ax.set_xlabel(item)
    ax.set_ylabel('Average Sale Price')
    ax.legend()
    plt.show()'''

numerical_continous_data = numerical_data.drop(columns = numerical_categorical_features)
numerical_continous_data = numerical_continous_data.drop(columns = ['LotFrontage', 'GarageYrBlt', 'MasVnrArea', 'SalePrice', 'MiscVal', '3SsnPorch', 'LowQualFinSF', 'EnclosedPorch', 'BsmtFinSF2'])

target = 'SalePrice'
features = numerical_continous_data

X_train, X_test, y_train, y_test = train_test_split(features, numerical_data[target], test_size=0.2, random_state=42)

param_dist = {'bootstrap': [True, False],
  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
  'max_features': ['auto', 'sqrt'],
  'min_samples_leaf': [1, 2, 4],
  'min_samples_split': [2, 5, 10],
  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
  'max_samples': [0.5, 0.75, 1.0], 
  'warm_start': [True, False]  
 }

param_dist = {
    'n_estimators': [50, 100, 200, 400],          # Reduced the number of options
    'max_features': ['sqrt', 'log2'],   # Keeping it simple
    'max_depth': [10, 20, None],        # Few depth options
    'min_samples_split': [2, 5],        # Common choices
    'min_samples_leaf': [1, 2],  
    'max_samples': [0.5, 0.75, 1.0], 
    'warm_start': [True, False]    # Common choices
}

model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_grid = GridSearchCV(estimator=model, param_grid=param_dist, cv=3, verbose=2, n_jobs=-1)

rf_grid.fit(X_train, y_train)

y_pred = rf_grid.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')
print(f'Best Hyperparameters: {rf_grid.best_params_}')

# Get feature importances
feature_importances = rf_grid.best_estimator_.feature_importances_
features_list = features.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': features_list, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importances in Random Forest Regressor')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#miscval, 3ssnporch, lowqualfinsf, enclosedporch, bsmtfinsf2