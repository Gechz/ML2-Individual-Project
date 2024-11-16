import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

def run_simulation():
    st.markdown(f"The tensfor flow version is {(tf.__version__)}")
    # Load the trained Keras model
    model = tf.keras.models.load_model('IndProject/model1.keras')

    # Load the scalers
    with open('IndProject/scaler_y.pkl', 'rb') as file:
        scaler_y = pickle.load(file)

    with open('IndProject/scaler_x.pkl', 'rb') as file:
        scaler_x = pickle.load(file)

    # Streamlit app setup
    st.title('Housing Price Prediction App')

    # User input for the X features based on the data columns
    st.markdown("""
    **Note**: `MedInc` has been converted to a nominal value (not thousands).
    """)
    
    med_inc = st.number_input('Median Income (nominal)', min_value=500.0, max_value=20000.0, step=1000.0, format="%.1f")
    population = st.number_input('Population', min_value=50.0, max_value=50000.0, step=500.0, format="%.1f")
    
    house_age = st.slider('House Age', min_value=1, max_value=80, step=1, value=30)
    ave_rooms = st.slider('Average Number of Rooms', min_value=1, max_value=20, step=1, value=5)
    ave_bedrms = st.slider('Average Number of Bedrooms', min_value=1, max_value=35, step=1, value=1)
    ave_occup = st.slider('Average Occupancy', min_value=1, max_value=40, step=1, value=3)

    # Prepare input data
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup]
    })

    # Scale the input data using the scaler for X
    input_data_scaled = scaler_x.transform(input_data)

    # Predict using the model
    prediction = model.predict(input_data_scaled)
    prediction_rescaled = scaler_y.inverse_transform(prediction.reshape(-1, 1))

    # Ensure the prediction is non-negative and rounded
    final_prediction = max(0, round(prediction_rescaled[0][0]))

    # Display the result in a larger font
    st.markdown(f"<p style='font-size:24px;'>Predicted House Price in 1990: ${final_prediction:,}</p>", unsafe_allow_html=True)

    st.header("Considerations for Growth of Housing Market:")
    st.markdown("""
    - According to real estate trends, the average annual increase for California housing has varied, but has often ranged from about 3% to 7% per year, depending on market conditions.
    - For simplicity purposes, we will assume a 5% yearly increase""")
    st.subheader("Multiplier to be applied: 5.52")

    today_prediction = round(final_prediction * 5.52,2)
    st.markdown(f"<p style='font-size:24px;'>Predicted House Price in 1990: ${today_prediction:,}</p>", unsafe_allow_html=True)


    # Display house style based on the house age
    if house_age < 10:
        st.image('IndProject/modern_house.jpeg')  # Replace with the path to the modern house image
        st.write("Almost brand new house")
    elif 10 <= house_age <= 35:
        st.image('IndProject/mid_modern_house.jpg')  # Replace with the path to the mid-modern house image
        st.write("Not too old, not too new")
    else:
        st.image('IndProject/old_house.jpg')  # Replace with the path to the old house image
        st.write("Are you into old haunted houses?")

if __name__ == '__main__':
     run_simulation()
