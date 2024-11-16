import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn import datasets
import streamlit as st

def eda_page():
    # Load the California housing dataset
    data = datasets.fetch_california_housing(as_frame=True)
    x = pd.DataFrame(data["data"], columns=data["feature_names"])
    y_th = pd.Series(data["target"], name=data["target_names"][0])
    y = y_th * 1000  # Convert target values to scalar format (house prices in dollars)

    # Multiply all values in 'Median Income' by 1000
    x['MedInc'] = x['MedInc'] * 1000

    # Drop 'Latitude' and 'Longitude'
    x = x.drop(columns=["Latitude", "Longitude"])

    # Concatenate x and y to form the complete DataFrame
    y = y.reset_index(drop=True)
    df = pd.concat([x, y], axis=1)

    # Streamlit page title
    st.title("Exploratory Data Analysis (EDA) of California Housing Data")

    # 1. Data Preparation
    
    # Dataset Information Section
    st.subheader("Dataset Information")
    st.markdown("""
    - **Number of Instances**: 20,640
    - **Number of Attributes**: 8 numeric, predictive attributes and the target
    - **Attribute Information**:
        - `MedInc`: Median income in block group
        - `HouseAge`: Median house age in block group
        - `AveRooms`: Average number of rooms per household
        - `AveBedrms`: Average number of bedrooms per household
        - `Population`: Block group population
        - `AveOccup`: Average number of household members
        - `Latitude`: Block group latitude
        - `Longitude`: Block group longitude
    - **Missing Attribute Values**: None
    """)
    st.subheader("Data Preparation")
    st.markdown("""
    The dataset was prepared by removing `Latitude` and `Longitude` columns, detecting outliers, and ensuring `MedInc`
     retained critical data points even after cleaning. Below is the code used for this preparation. `Latitude` and `Longitude`were removed due to the nature of their values:
    """)
    st.code("""
    # Drop 'Latitude' and 'Longitude'
    x = x.drop(columns=["Latitude", "Longitude"])

    # Remove outliers using the IQR method, except for 'MedInc' after inspecting Box Plots
    x_cleaned = remove_outliers(x, x.columns.drop("MedInc"))
    """)
    st.subheader("Main Insights")
    st.markdown("""
    - The **heatmap** shows that there are no null values in the dataset.
    - The **correlation matrix** highlights a high correlation between `AveRooms` and `AveBedrms`, as expected, and a significant correlation between `MedInc` and the target variable (`Median House Value`).
    - The **histograms** indicate that some features, such as `MedInc`, have a wide dispersion, while others have a narrow spread. This suggests the importance of using a scaler for feature standardization.
    - The **scatter plot** reinforces what was seen in the correlation matrix, showing relationships between selected features and the target variable.
    - The **boxplots** illustrate the effect of outlier removal. It is important to note that outliers in `MedInc` were retained as they are intrinsic to the feature and removing them could negatively impact model generalization.
    """)

    # 2. Heatmap for Null Values
    with st.expander("Heatmap for Null Values"):
        st.markdown("This heatmap confirms that there are no null values in the dataset.")
        plt.figure(figsize=(10, 6))
        sns.heatmap(x.isnull(), cbar=False, cmap='viridis')
        plt.title('Heatmap for Null Values')
        st.pyplot(plt)

    # 3. Correlation Matrix
    with st.expander("Correlation Matrix"):
        st.markdown("This correlation matrix shows the strength of relationships between the different features.")
        plt.figure(figsize=(12, 10))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        st.pyplot(plt)

    # 4. Histograms of Features
    with st.expander("Histograms of Features"):
        st.markdown("These histograms show the distribution of key features in the dataset.")

        # Histogram for 'MedInc'
        st.markdown("### Histogram of Median Income")
        plt.figure(figsize=(10, 6))
        sns.histplot(x['MedInc'], kde=True, bins=30, color='blue')
        plt.title('Distribution of Median Income')
        plt.xlabel('Median Income')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Histogram for 'HouseAge'
        st.markdown("### Histogram of House Age")
        plt.figure(figsize=(10, 6))
        sns.histplot(x['HouseAge'], kde=True, bins=30, color='green')
        plt.title('Distribution of House Age')
        plt.xlabel('House Age')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Histogram for 'HouseAge'
        st.markdown("### Histogram for Average Rooms")
        plt.figure(figsize=(8, 4))
        sns.histplot(x['AveRooms'], kde=True, bins=30, color='green')
        plt.title('Distribution of Average Rooms')
        plt.xlabel('Average Rooms')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Histogram for 'HouseAge'
        st.markdown("### Histogram for Average Bedroooms")
        plt.figure(figsize=(8, 4))
        sns.histplot(x['AveBedrms'], kde=True, bins=30, color='green')
        plt.title('Distribution of Average Bedrooms')
        plt.xlabel('Average Bedrooms')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Histogram for 'HouseAge'
        st.markdown("### Histogram for Population")
        plt.figure(figsize=(8, 4))
        sns.histplot(x['Population'], kde=True, bins=30, color='green')
        plt.title('Distribution of Population')
        plt.xlabel('Population')
        plt.ylabel('Frequency')
        st.pyplot(plt)

        # Histogram for 'HouseAge'
        st.markdown("### Histogram for Average Occupation")
        plt.figure(figsize=(8, 4))
        sns.histplot(x['AveOccup'], kde=True, bins=30, color='green')
        plt.title('Distribution of Average Occupation')
        plt.xlabel('Average Occupation')
        plt.ylabel('Frequency')
        st.pyplot(plt)

    # 5. Scatter Plot
    with st.expander("Scatter Plots"):
        st.markdown("This scatter plot complements the correlation matrix by showing the high correlation between this feature and the target variable")

        # MedInc Scatter Plot
        st.markdown("### Scatter Plot MedInc vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['MedInc'], y=data.target, alpha=0.5, color='purple')
        plt.title('Median Income vs. Median House Value')
        plt.xlabel('Median Income')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

        # House Age Scatter Plot
        st.markdown("### Scatter Plot HouseAge vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['HouseAge'], y=data.target, alpha=0.5, color='purple')
        plt.title('House Age vs. Median House Value')
        plt.xlabel('House Age')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

        # AveRooms Scatter Plot
        st.markdown("### Scatter Plot AveRooms vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['AveRooms'], y=data.target, alpha=0.5, color='purple')
        plt.title('Average Rooms vs. Median House Value')
        plt.xlabel('Average Rooms')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

        # AveBedrooms Scatter Plot
        st.markdown("### Scatter Plot AveBedrms vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['AveBedrms'], y=data.target, alpha=0.5, color='purple')
        plt.title('Average Bedroooms vs. Median House Value')
        plt.xlabel('Average Bedrooms')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

        # Population Scatter Plot
        st.markdown("### Scatter Plot Population vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['Population'], y=data.target, alpha=0.5, color='purple')
        plt.title('Population vs. Median House Value')
        plt.xlabel('Population')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

        # AveOccup Scatter Plot
        st.markdown("### Scatter Plot AveOccup vs MedHouseVal")
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['AveOccup'], y=data.target, alpha=0.5, color='purple')
        plt.title('AveOccup vs. Median House Value')
        plt.xlabel('AveOccup')
        plt.ylabel('Median House Value')
        st.pyplot(plt)

    # 6. Target Variable Histogram
    with st.expander("Target Variable Histogram"):
        st.markdown("This histogram shows the different values for the target variable.")

        st.markdown("### MedHouseVal Histogram")
        plt.figure(figsize=(12, 8))
        sns.histplot(x=y)
        plt.title('Median House Value Histogram')
        plt.xlabel('Median House Value')
        plt.ylabel('Count')
        st.pyplot(plt)

    # 5. Boxplots for Outlier Analysis (Original Data)
    with st.expander("Boxplots for Outlier Analysis (Original Data)"):
        st.markdown("These boxplots help identify outliers in each feature before any data cleaning.")

        # Boxplot for 'MedInc'
        st.markdown("### Boxplot for Median Income")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='MedInc', color='lightblue')
        plt.title('Boxplot of Median Income')
        plt.xlabel('Median Income')
        st.pyplot(plt)

        # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for House Age")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='HouseAge', color='lightblue')
        plt.title('Boxplot of House Age')
        plt.xlabel('House Age')
        st.pyplot(plt)

                # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Rooms")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='AveRooms', color='lightblue')
        plt.title('Boxplot for Average Rooms')
        plt.xlabel('House Age')
        st.pyplot(plt)

                # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Bedroooms")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='AveBedrms', color='lightblue')
        plt.title('Boxplot for Average Bedroooms')
        plt.xlabel('House Age')
        st.pyplot(plt)

                # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Population")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='Population', color='lightblue')
        plt.title('Boxplot for Population')
        plt.xlabel('House Age')
        st.pyplot(plt)

                # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Occupation")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, y='AveOccup', color='lightblue')
        plt.title('Boxplot for Average Occupation')
        plt.xlabel('House Age')
        st.pyplot(plt)

    # 6. Outlier Detection and Removal
    st.subheader("Outlier Detection and Removal")
    st.markdown("""
    Outlier detection was performed using the IQR method. Outliers were removed from all features except `MedInc` to maintain important data points for this feature.
    """)
    st.code("""
    #Function to Count number of outliers per Feature
    def detect_outliers(data, feature):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
    return outliers
    # Function to systematically remove outliers using the IQR method
    def remove_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    # Apply outlier removal, retaining 'MedInc'
    x_cleaned = remove_outliers(x, x.columns.drop("MedInc"))
    x_cleaned = x_cleaned.reset_index(drop=True)
    y = y.reset_index(drop=True)
    df_cleaned = pd.concat([x_cleaned, y], axis=1)
    """)

    def detect_outliers(data, feature):
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        outliers = data[(data[feature] < (Q1 - 1.5 * IQR)) | (data[feature] > (Q3 + 1.5 * IQR))]
        return outliers

    for feature in x.columns:
        outliers = detect_outliers(df, feature)
        st.markdown(f"**Number of outliers in {feature}: {len(outliers)}**")


    def remove_outliers(df, columns):
        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df

    # Remove outliers from x but keep 'MedInc' intact
    x_cleaned = remove_outliers(x, x.columns.drop("MedInc"))
    st.markdown((f"{round((y.shape[0]-x_cleaned.shape[0])/y.shape[0]*100,2)}% of the data was removed"))
    st.markdown(f"The remaining rows are {x_cleaned.shape[0]} from {y.shape[0]}")
    # Reset index of cleaned x and y and concatenate
    x_cleaned = x_cleaned.reset_index(drop=True)
    df_cleaned = pd.concat([x_cleaned, y], axis=1)

    # 7. Boxplots After Outlier Removal
    with st.expander("Boxplots After Outlier Removal"):
        st.markdown("These boxplots display the distribution of features after removing outliers.")

        # Boxplot for 'MedInc' (Without Outliers)
        st.markdown("### Boxplot for Median Income (No Outliers Removed)")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='MedInc', color='lightgreen')
        plt.title('Boxplot of Median Income (No Outliers Removed)')
        plt.xlabel('Median Income')
        st.pyplot(plt)

        # Boxplot for 'HouseAge' (Without Outliers)
        st.markdown("### Boxplot for House Age (Without Outliers)")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='HouseAge', color='lightgreen')
        plt.title('Boxplot of House Age (Without Outliers)')
        plt.xlabel('House Age')
        st.pyplot(plt)

        # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Rooms")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='AveRooms', color='lightgreen')
        plt.title('Boxplot for Average Rooms')
        plt.xlabel('House Age')
        st.pyplot(plt)

        # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Bedroooms")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='AveBedrms', color='lightgreen')
        plt.title('Boxplot for Average Bedroooms')
        plt.xlabel('House Age')
        st.pyplot(plt)

        # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Population")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='Population', color='lightgreen')
        plt.title('Boxplot for Population')
        plt.xlabel('House Age')
        st.pyplot(plt)

        # Boxplot for 'HouseAge'
        st.markdown("### Boxplot for Average Occupation")
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df_cleaned, y='AveOccup', color='lightgreen')
        plt.title('Boxplot for Average Occupation')
        plt.xlabel('House Age')
        st.pyplot(plt)
if __name__ == '__main__':
    eda_page()