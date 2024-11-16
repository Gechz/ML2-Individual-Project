import streamlit as st
import importlib.util
from pathlib import Path

# Set up the app layout and default page configuration
st.set_page_config(page_title="Machine Learning II Individual Project", layout="wide")

# Display the main page content with a centered title and delivery date
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Machine Learning II Individual Project </h1>
        <h1>California Housing Market Artificial Neural Network</h1>
        <h3>Delivery Date: November 17, 2024</h3>
    </div>
    """,
    unsafe_allow_html=True
)

 # Team Members Section
st.header("Project Owner")
    
    # Define team members with specific images and roles
team_data = [
        {"name": "Gabriel Chapman", "Section": "Section A-1", "image": "pages/path_to_gabriel.jpg"}
    ]
    
    # Display each team member in a larger format
for member in team_data:
    col1, col2 = st.columns([1, 2])  # Adjust column ratio to make images larger
    with col1:
        st.image(member["image"], use_column_width=True)  # Larger image size
    with col2:
        st.subheader(member["name"])
        st.write(member["Section"])


# Pages in the side
pages = [
    ("Cover Page", "Main"),
    ("Introduction", "1_README"),
    ("EDA","2_Exploratory Data Analysis"),
    ("Setup", "3_Artificial Neural Network Setup"),
    ("Model Result", "4_ANN Model Result"),
    ("Simulation", "5_Simulation")
]


