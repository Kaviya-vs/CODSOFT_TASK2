import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the dataset
# Use the correct local path to your CSV file
data = pd.read_csv('C:/Users/Ram mrithyun jay/Desktop/kaviya data set/IMDb Movies India.csv', encoding='ISO-8859-1')


# Data Cleaning
# Combine the three actor columns into one
data['actors'] = data['Actor 1'] + ', ' + data['Actor 2'] + ', ' + data['Actor 3']
data = data[['Genre', 'Director', 'actors', 'Rating']].dropna()

# Feature Engineering - One-hot encode categorical features
X = data[['Genre', 'Director', 'actors']]
y = data['Rating']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Genre', 'Director', 'actors'])
    ])

# Build the pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
