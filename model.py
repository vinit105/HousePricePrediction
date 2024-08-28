import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
data = pd.read_csv('house.csv')
df = pd.DataFrame(data)
# Handle missing values (if any)
df.fillna(df.median(), inplace=True)

# Split data into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model (Random Forest Regressor)
model = RandomForestRegressor(random_state=24)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
train_data = X_train.join(y_train)
#print(train_data)
train_data.hist(figsize=(15, 8))  #show graphs
def predict_price(area, bhk, bathroom, is_furnished, parking, is_apartment):
    input_data = {
        'Area': [area],
        'BHK': [bhk],
        'Bathroom': [bathroom],
        'IsFurnished': [is_furnished],
        'Parking': [parking],
        'isApartment': [is_apartment]
    }
    user_input = pd.DataFrame(input_data)
    predicted_price = model.predict(user_input)
    return predicted_price[0]
joblib.dump(model, 'property_price_model.pkl')