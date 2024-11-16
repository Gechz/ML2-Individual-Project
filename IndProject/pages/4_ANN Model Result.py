import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def ann_model_result():
    st.header("Overall Insights")
    st.markdown("""
    - The Training and Validation loss graphs shows that overfitting was prevented before running 40 epochs.
    - The number of hidden layer nodes can influence forecasting power, but increasing this will harm computational power.
    - Error Metrics show a solid model, but still with room for improvement.
    - Improvements:
        - Add other categorical features such as location to improve forecasting power.
        - Add other numeric features such as number of floors.
    """)
    # Subheader for the loss plot
    st.subheader("Training and Validation Loss")
    st.markdown("""
    This plot shows the training and validation loss over epochs. The training loss curve indicates how well the model is learning on the training data, while the validation loss helps to monitor its generalization on unseen data. 
    A consistent gap between the curves or an increasing validation loss while training loss continues to decrease might indicate overfitting. This furthermore indicates that the model was stopped after roughly 35 epochs to prevent overfitting.
    """)
    st.image("IndProject/training_validation_loss.png")  # Replace with the correct path to your image
    st.code("""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Training and Validation Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    """)

    # Subheader for the standardized test predictions plot
    st.subheader("Test Predictions vs. True Values (Standardized)")
    st.markdown("""
    This scatter plot illustrates the predicted values against the true values from the test set, in their standardized form. A strong positive diagonal pattern indicates good predictive performance.
    """)
    st.image("IndProject/standardized_predictions.png")  # Replace with the correct path to your image
    st.code("""
    y_predict = model1.predict(X_test)
    plt.plot(y_test, y_predict, "^", color='r')
    plt.xlabel('Model Predictions')
    plt.ylabel('True Values')
    """)

    # Subheader for the original scale test predictions plot
    st.subheader("Test Predictions vs. True Values (Original Scale)")
    st.markdown("""
    This plot provides a more interpretable view by showing the true values against the predicted values on their original scale. A pattern close to a 45-degree line suggests accurate predictions.
    """)
    st.image("IndProject/original_scale_predictions.png")  # Replace with the correct path to your image
    st.code("""
    # Reshape y_predict to have the same number of features as the scaler expects
    y_predict_reshaped = y_predict.reshape(-1, 1)

    # Inverse transform using the new scaler
    y_predict_orig = scalery.inverse_transform(y_predict_reshaped)
    y_test_orig = scalery.inverse_transform(y_test)
    y_train_orig = scalery.inverse_transform(y_train_full)

    plt.plot(y_test_orig, y_predict_orig, "^", color='r')
    plt.xlabel('Model Predictions')
    plt.ylabel('True Values')
    """)

    # Subheader for test set error metrics
    st.subheader("Test Set Error Metrics")
    st.markdown("""
    The table below displays the evaluation metrics for the test set, including R quared (R^2), Root Mean Squared Error (RMSE), Mean Squared Error (MSE), and Mean Absolute Error (MAE). These metrics provide insights into the model's prediction accuracy on unseen data.
    """)
    #st.image("IndProject/test_results.png")  # Replace with the correct path to your image
    st.code("""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    from math import sqrt
    # Generate predictions on the test data
    y_test_predict = model1.predict(X_test)

    # Reshape y_test_predict to have the same number of features as the scaler expects
    y_test_predict_reshaped = y_test_predict.reshape(-1, 1)

    # Inverse transform the test predictions
    y_test_predict_orig = scalery.inverse_transform(y_test_predict_reshaped)

    # Calculate error metrics for the test set
    RMSE_test = round(float(format(np.sqrt(mean_squared_error(y_test_orig, y_test_predict_orig)), '.3f')), 2)
    MSE_test = round(mean_squared_error(y_test_orig, y_test_predict_orig), 2)
    MAE_test = round(mean_absolute_error(y_test_orig, y_test_predict_orig), 2)
    R2_test = round(r2_score(y_test_orig, y_test_predict_orig), 3)
    MAPE_test = round(mean_absolute_percentage_error(y_test_orig,y_test_predict_orig),3)

    print("TEST RESULTS")
    print('RMSE =', RMSE_test, '\nMSE =', MSE_test, '\nMAE =', MAE_test, '\nR2 =', R2_test, '\nMAPE = ', MAPE_test')
    """)

    # Subheader for train set error metrics
    st.subheader("Train Set Error Metrics")
    st.markdown("""
    The train set metrics provide an understanding of how well the model performs on the training data. This can help to identify if the model has overfit the data or is generalizing well.
    """)
    #st.image("IndProject/train_results.png")  # Replace with the correct path to your image
    st.code("""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
    from math import sqrt

    # Generate predictions on the training data
    y_train_predict = model1.predict(X_train_full)

    # Reshape y_train_predict to have the same number of features as the scaler expects
    y_train_predict_reshaped = y_train_predict.reshape(-1, 1)

    # Inverse transform the training predictions
    y_train_predict_orig = scalery.inverse_transform(y_train_predict_reshaped)

    # Calculate error metrics for the training set
    RMSE_train = round(float(format(np.sqrt(mean_squared_error(y_train_orig, y_train_predict_orig)), '.3f')), 2)
    MSE_train = round(mean_squared_error(y_train_orig, y_train_predict_orig), 2)
    MAE_train = round(mean_absolute_error(y_train_orig, y_train_predict_orig), 2)
    R2_train = round(r2_score(y_train_orig, y_train_predict_orig), 3)
    MAPE_train = round(mean_absolute_percentage_error(y_train_orig,y_train_predict_orig),3)

    print("TRAIN RESULTS")
    print('RMSE =', RMSE_train, '\nMSE =', MSE_train, '\nMAE =', MAE_train, '\nR2 =', R2_train, '\nMAPE = ', MAPE_train')
    """)


    # Creating the dataframe for the table data
    data = {
        "Metrics": ["RMSE", "MSE", "MAE", "R2", "MAPE"],
        "Train Results": [596.72, 356070.36, 436.31, 0.73, 0.252],
        "Test Results": [620.05, 384467.41, 447.14, 0.704, 0.266]
    }
    
    df = pd.DataFrame(data)

    # Displaying the table with streamlit functions
    st.dataframe(df)
    # Main Insights section
    st.subheader("Main Insights")
    st.markdown("""
    - Overfitting was prevented after 35 epochs, as indicated by the convergence of the training and validation loss curves.
    - The error metrics (RMSE, MSE, and MAE) indicate a solid model performance.
    - The MAPE error sits between 20-30% which presents a reliable forecast power.
    - The R² score sits at a respectable 70%, showing that the model explains a significant portion of the variance in the target variable.
    """)


# Use this function in your Streamlit app to display the ANN model results page.
if __name__ == '__main__':
    ann_model_result()
