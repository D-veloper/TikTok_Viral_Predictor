import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Display full tables in terminal window

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("../data sets/7. dataset_tiktok_v4 - final.csv")  # Load data set

# Extract desired features
selected_features = ['engagement', 'total_videos', 'day_posted', 'time_posted', 'video_duration', 'music_name', 'hashtags_used']
data = df[selected_features]

# Perform one-hot encoding for the categorical variables (day_posted and music_name)
data = pd.get_dummies(data, columns=['day_posted', 'music_name'])

# Sort predictor (X) and target (y) variables
X = data
y = df['video_plays']

# Standardize the extreme numerical features (Engagement and Total Videos)
scaler = StandardScaler()
features_to_scale = ['engagement', 'total_videos']
X[features_to_scale] = scaler.fit_transform(X[features_to_scale])

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize RandomizedSearchCV to find optimum setting
rf_random = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=42), param_distributions=param_grid,
                               n_iter=10, cv=5, random_state=42, n_jobs=-1)

# Fit the RandomizedSearchCV
rf_random.fit(X_train, y_train)

# Get the best parameters
best_params = rf_random.best_params_
# print('Best Parameters:', best_params)

# Initialize the Random Forest Regressor with the best parameters
rf_model = RandomForestRegressor(n_estimators=best_params['n_estimators'],
                                  max_depth=best_params['max_depth'],
                                  min_samples_split=best_params['min_samples_split'],
                                  min_samples_leaf=best_params['min_samples_leaf'],
                                  random_state=42)

# Train the random forest model
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Calculate the Mean Squared Error (It is large due to numbers involved. Convert to percentage for better comprehension)
rf_mse = mean_squared_error(y_test, y_pred)

# Calculate the variance of the target variable
rf_variance = np.var(y_test)

# Calculate MSE as a percentage of the variance
rf_mse_percentage = (rf_mse / rf_variance) * 100

# Get feature importance
feature_importance = rf_model.feature_importances_

# Create a DataFrame to display feature importance
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

print(fi_df.head(28))

#  Linear Regression

# Select features and target variable
features = ['engagement', 'total_videos', 'time_posted', 'video_duration', 'hashtags_used']
X = df[features]
y = df['video_plays']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error
lr_mse = mean_squared_error(y_test, y_pred)

# Calculate the variance of the target variable
lr_variance = np.var(y_test)

# Calculate MSE as a percentage of the variance
lr_mse_percentage = (lr_mse / lr_variance) * 100

print('Linear Regression vs. Random Forest MSE Percentage Comparison:')
print('Random Forest:', rf_mse_percentage)
print('Linear Regression:', lr_mse_percentage)

# Plot the data points and regression lines
plt.figure(figsize=(12, 8))

for i, feature in enumerate(features):
    plt.subplot(3, 2, i+1)
    plt.scatter(X[feature], y, color='blue', label='Data')
    plt.plot(X_test[feature], y_pred, color='red', linewidth=2, label='Regression Line')
    plt.xlabel(feature)
    plt.ylabel('Video Plays')
    plt.title(f'{feature} vs. Video Plays')
    plt.legend()

plt.tight_layout()
plt.show()
