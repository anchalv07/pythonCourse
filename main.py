import streamlit as st
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

X_train = [[2500, 3, 2, 2005, 1, 0, 0], [3000, 4, 3, 2010, 0, 1, 0]]  # Example features
y_train = [500, 600]  # Example labels in lakhs

model = LinearRegression()
model.fit(X_train, y_train)

with open('House Price Prediction.pkl', 'wb') as f:
    pickle.dump(model, f)

# File path to the model
model_file = 'House Price Prediction.pkl'

# Placeholder model creation
def create_placeholder_model():
    # Simple linear regression model as a placeholder
    model = LinearRegression()
    # Assuming the model takes 7 features: 4 for inputs and 3 for one-hot encoded neighborhoods
    X_placeholder = np.random.rand(10, 7)
    y_placeholder = np.random.rand(10)
    model.fit(X_placeholder, y_placeholder)
    return model

model = None
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
except (EOFError, FileNotFoundError, pickle.UnpicklingError):
    st.warning("The model file is empty or corrupted. Please upload a valid model file.")
    model = create_placeholder_model()  # Use placeholder model as a fallback

st.title("House Price Prediction")

st.header("Enter House Details:")

# Input fields
Square_feet = st.number_input("Square Feet", min_value=50, value=2500)
bedrooms = st.number_input("Bedrooms", min_value=1, value=15)
Bathrooms = st.number_input("Bathrooms", min_value=1, value=15)
Year_Built = st.number_input("Year Built", min_value=1950, value=2021)

Neighbor_hood = st.selectbox("Neighborhood", options=[
    'Rural', 'Suburb', 'Urban'
])

# One-hot encode the neighborhood
neighborhood_options = ['Rural', 'Suburb', 'Urban']
Neighbor_hood_encoded = [1 if Neighbor_hood == option else 0 for option in neighborhood_options]

# Combine all the inputs into a single array
input_data = np.array([Square_feet, bedrooms, Bathrooms, Year_Built] + Neighbor_hood_encoded)

if st.button("Predict House Price"):
    try:
        # Perform prediction
        prediction = model.predict(input_data.reshape(1, -1))

        # Display the prediction
        st.subheader(f"Predicted House Price: ₹{prediction[0]:,.2f} Lakh")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")




