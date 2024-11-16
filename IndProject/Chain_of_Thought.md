# Analysis and Thought Process Behind the Machine Learning Python Notebook: California Housing Market Value Prediction

## Dataset Overview
The dataset used in this project is the **California Housing Market Value** dataset from the 1990 census, sourced from the **Scikit-Learn library**. Initially, the **Keras library dataset** was considered, but it was found to lack column names and was stored in an array format, making it less suitable for this analysis. Thus, the Scikit-Learn version was chosen for better clarity and usability.

## Initial Data Preparation
After uploading the dataset, it became evident that some feature transformations were necessary:
- **Median Income**: This feature was initially represented in thousands of dollars. To standardize it, the values were multiplied by 1,000 to return it to its nominal value.
- **Median House Value** (target variable): Similar to the median income, this target variable was scaled by 1,000 for consistency.

### Data Cleansing
Thankfully, the dataset did not contain any null values, which simplified the preprocessing step. 

### Feature Selection
Certain features were excluded from the analysis:
- **Latitude and Longitude**: These geographical coordinates were discarded due to their potential negative impact on model performance, given their nature and potential complexities in interpretation.

### Attempted Feature Engineering
An API was initially considered for determining the area of the house and creating categorical values. However, this idea was shelved due to scalability issues.

## Exploratory Data Analysis (EDA)
- **Histograms and Boxplots**: Visualizations such as histograms and boxplots were created for various features to understand their distributions and identify potential outliers.
- **Correlation Matrix**: A correlation matrix was used to examine the relationships between features and the target variable.

### Outlier Removal Strategy
To handle outliers, a 1.5*IQR rule was applied to all features except the target variable and **Median Income**. Removing outliers from these variables was avoided to preserve the model's generalization capabilities.

## Data Scaling
Both the feature set and the target variable were scaled using **StandardScaler** to manage feature dimensionality effectively.

## Model Architecture: Artificial Neural Network (ANN)
### Layer Configuration
- **Input Layer**: Configured as a vector of shape `[number of features,]` to match the input data.
- **Hidden Layers**:
  - **First Hidden Layer**: Composed of 100 nodes.
  - **Second Hidden Layer**: Composed of 50 nodes.
- **Output Layer**: Configured as a single node to predict the target value.

### Compilation Details
The model was compiled with:
- **Loss Function**: Mean Squared Error (MSE). This is a standard choice for regression problems in ANNs as it penalizes large errors more heavily and provides a straightforward gradient for backpropagation.
- **Optimizer**: Configured with a learning rate of `1e-3`, which provides a balance between convergence speed and stability. A learning rate that is too high can cause the model to overshoot the optimal point, while one that is too low slows down the training process.

### Callbacks
Three callbacks were implemented to enhance training and prevent overfitting:
1. **Model Checkpoint**: Saves the best version of the model during training.
   ```python
   checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model1.keras", save_best_only=True)
   early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
   tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
### Fit
1. **Model Fit**: Fits with 100 epochs and a Validation split of 30%.
   ```python
   history = model1.fit(X_train_full, y_train_full, epochs=100,
                     validation_split=0.3,
                     callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
## Results
- When one inputs the features laid out (Population of block, Median Income of block, House Age, Average number of rooms, bedrooms and occupation) one gets the predicted value of the house.
- Outside of the notebook, the prediction app consider the time nuance since this data is from 1990 and applies a multiplying factor to estimate the price of the house for the year 2024.

## Insights and Analysis

### Training and Validation Loss
The training and validation loss graphs indicated that overfitting was successfully avoided, with optimal performance achieved before 40 epochs.

### Model Considerations
The number of hidden layer nodes significantly influences the model's forecasting power. While increasing the number of nodes may improve predictive accuracy, it can also increase computational requirements.

### Error Metrics
The model demonstrated solid performance, though improvements are still possible.

## Recommendations for Model Improvement
- **Inclusion of Additional Categorical Features**: Integrating location data or other relevant categories could enhance the model's forecasting ability.
- **Additional Numerical Features**: Features such as the number of floors or other property characteristics could provide more predictive power.



