import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import streamlit as st

def ann_preparation():
    st.title("Artificial Neural Network Preparation")

    # Data Splitting
    st.subheader("Data Splitting")
    st.markdown("The dataset is split into training and testing sets using an 80/20 split.")
    st.code("""
    X_train_full, X_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    """)

    # Feature Scaling
    st.subheader("Feature Scaling")
    st.markdown("Standard scaling is applied to both the input features and the target variable to improve the model's performance.")
    st.code("""
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_full)
    X_test = scaler.transform(X_test)

    scalery = StandardScaler()
    y_train_full = scalery.fit_transform(y_train_full.values.reshape(-1, 1))
    y_test = scalery.transform(y_test.values.reshape(-1, 1))
    """)

    # Model Definition
    st.subheader("Model Definition")
    st.markdown("The ANN model is defined with two hidden layers and one output layer for regression.")
    st.code("""
    HL1 = 100
    HL2 = 50
    OL_Regression = 1
    n_features = len(x.columns)

    model1 = keras.models.Sequential([
        keras.layers.Dense(HL1, activation="relu", input_shape=[n_features]),
        keras.layers.Dense(HL2, activation="relu"),
        keras.layers.Dense(OL_Regression, activation="linear")
    ])
    """)

    # Displaying Model Summary as Image
    st.text("Model Summary:")
    st.image("IndProject/path_to_modelsumm.png")  # Replace this path with your image path if necessary

    # Model Compilation
    st.subheader("Model Compilation")
    st.markdown("The model is compiled using the Mean Squared Error loss function and the Adam optimizer.")
    st.code("""
    # Compile the model with Mean Squared Error loss and Adam optimizer
    model1.compile(loss="mean_squared_error", optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    """)

    # Callbacks
    st.subheader("Callbacks for Training")
    st.markdown("Callbacks such as checkpoint saving, early stopping, and TensorBoard logging are defined for efficient training.")
    st.code("""
    # Define callbacks including checkpoint and early stopping
    checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model1.keras", save_best_only=True)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    """)

    # Model Training
    st.subheader("Model Training")
    st.markdown("The model is trained with a validation split of 0.3 and a maximum of 100 epochs.")
    st.code("""
    # Fit the model with validation split at 0.3
    history = model1.fit(
        X_train_full, y_train_full, 
        epochs=100, 
        validation_split=0.3, 
        callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb]
    )
    """)

    # Model Evaluation
    st.subheader("Model Evaluation")
    st.markdown("Evaluate the trained model on the test data to determine its performance.")
    st.code("""
    # Evaluate the model on test data
    model1_test = model1.evaluate(X_test, y_test)
    """)

    st.markdown("The final evaluation metric is displayed after training, which due to the nature of the model, fluctuates every time it runs.")

# Call this function in your Streamlit multipage app to display the ANN preparation page.

if __name__ == '__main__':
    ann_preparation()
