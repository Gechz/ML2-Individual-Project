# California Housing Price Prediction App

This project is a **Streamlit web application** that predicts housing prices based on user input data. The application leverages a pre-trained Keras model and uses data preprocessing with scalers for accurate predictions.

## Project Overview

The app is designed to predict housing prices using features extracted from the California housing dataset. The model was trained on various features, including demographic and real estate data, to provide users with an estimation of house prices.

### Key Characteristics

- **User-friendly interface** for easy data input through Streamlit.
- **Scalable predictions** using a pre-trained Keras model.
- **Feature scaling** to ensure accurate model predictions.
- **Output rounding** and non-negative formatting for realistic price estimates.

## Data Features

The model uses the following input features for prediction:

1. **Median Income (`MedInc`)**: The median income in the block group.
2. **House Age (`HouseAge`)**: The average age of houses in the block group.
3. **Average Number of Rooms (`AveRooms`)**: The average number of rooms per household.
4. **Average Number of Bedrooms (`AveBedrms`)**: The average number of bedrooms per household.
5. **Population (`Population`)**: The total population in the block group.
6. **Average Occupancy (`AveOccup`)**: The average number of occupants per household.

## Target Variable

The target variable is a scalar value representing the **median house value for California** from the 1990 census. This variable was scaled during model training to improve prediction performance.

## Prerequisites

Ensure you have the following installed:

- **Python 3.x**
- **Streamlit**
- **TensorFlow**
- **Pandas**
- **scikit-learn**

## How to Run the App Locally

1. **Clone the repository** and navigate to the project directory.
2. **Place the following files** in the directory:
   - `model.keras`: The pre-trained Keras model.
   - `scaler_x.pkl`: The standard scaler for the input features.
   - `scaler_y.pkl`: The standard scaler for the target variable.
3. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
4. **Run the Streamlit app**:
   ```bash
   streamlit run app.py


The app prompts the user for the following inputs:

- Median Income
- House Age
- Average Number of Rooms
- Average Number of Bedrooms
- Population
- Average Occupancy

These inputs are then processed and scaled before being fed into the model for prediction.

Output
    - The app displays the predicted housing price as a non-negative, rounded value to make it more interpretable.
    - Example output:
      ```
      Predicted House Price: $350,000
      ```
 Model and Data Preparation
    - The model was trained using the **California housing dataset**.
    - Data was preprocessed using `StandardScaler` to normalize both the input features and target variable, ensuring effective model performance.

Notes
    - Ensure that the `model.keras` and `scaler` files are in the correct format and placed in the working directory.
    - The prediction model is optimized for the columns mentioned and may not generalize to other datasets without retraining.
    """)
