import streamlit as st
import importlib.util
from pathlib import Path
import os


# Set up the app layout and default page configuration
st.set_page_config(page_title="Machine Learning II Individual Project", layout="wide")

# Display the main page content with a centered title and delivery date
st.markdown(
    """
    <div style="text-align: center;">
        <h1>Machine Learning II Individual Project </h1>
        <h1>California Housing Market Artificial Neural Network</h1>
        <h3>Date: November 17, 2024</h3>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("For more information about implementation, check README file in the corresponding [GitHub repo](https://github.com/Gechz/ML2-Individual-Project/tree/main).")
st.markdown("General thought process is laid out in the different pages. For more details, head to the [`Chain_of_Thought`document](https://github.com/Gechz/ML2-Individual-Project/blob/main/IndProject/Chain_of_Thought.md) in the corresponding [Github repo](https://github.com/Gechz/ML2-Individual-Project/tree/main).")
 # Team Members Section
st.header("Project Owner")
    
    # Define team members with specific images and roles
team_data = [
        {"name": "Gabriel Chapman", "Section": "Section A-1"}
    ]
    
    # Display each team member in a larger format
for member in team_data:
    col1, col2 = st.columns([1, 2])  # Adjust column ratio to make images larger
    with col1:
        st.image("IndProject/path_to_gabriel.jpg", use_container_width=True)  # Larger image size
    with col2:
        st.subheader(member["name"])
        st.write(member["Section"])


# Pages in the side
pages = [
    ("Cover Page", "Main"),
    ("EDA","2_Exploratory Data Analysis"),
    ("Setup", "3_Artificial Neural Network Setup"),
    ("Model Result", "4_ANN Model Result"),
    ("Simulation", "5_Simulation")
]


