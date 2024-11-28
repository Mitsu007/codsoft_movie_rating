import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load data
file_path = r'C:\Users\ACER\Desktop\Suhas\Internship\Mitu internship\IMDb_Movies_India.csv'
data = pd.read_csv(file_path, encoding='latin1')
data = data.dropna(subset=['Genre', 'Director', 'Actor 1', 'Rating'])

# Features and target
features = data[['Genre', 'Director', 'Actor 1']]
target = data['Rating']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# OneHotEncoding for categorical variables
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

# Model training
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train_encoded, y_train)

# Evaluate model
y_pred = regressor.predict(X_test_encoded)
print(f"Test RMSE: {mean_squared_error(y_test, y_pred):.2f}")

# User input for prediction
genre = input("Enter the genre of the movie: ")
director = input("Enter the director of the movie: ")
actor = input("Enter the lead actor of the movie: ")

sample_movie = pd.DataFrame({'Genre': [genre], 'Director': [director], 'Actor 1': [actor]})
sample_movie_encoded = encoder.transform(sample_movie)

# Predict rating
predicted_rating = regressor.predict(sample_movie_encoded)
print(f"Predicted Rating: {predicted_rating[0]:.2f}")
