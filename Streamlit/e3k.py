import streamlit as st
import base64
from streamlit_extras.stylable_container import stylable_container
from azure_utils import get_azure_workspace, get_model, list_models
from preprocessing import preprocess_prediction_data, preprocess_training_data


# Inject custom CSS for background color
st.markdown(
    """
<style>
    .stApp {
        background-color: lightblue;
        padding: 20px; /* Optional: add some padding for better spacing */
    }
</style>
    """,
    unsafe_allow_html=True
)

# Load the logo image (JPG)
logo_image_path = "/Users/maxmeiners/Downloads/Screenshot_2024-06-20_at_17.49.24-removebg-preview.png"

# Read and encode the image in base64
def get_image_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

logo_image_base64 = get_image_base64(logo_image_path)

# Define styles for the title
title_color = "#0D4A6F"  # Example color: blue
title_font_size = "45px"  # Example font size
header_font_size = "20px"  # Font size for headers
subheader_font_size = "25px"  # Font size for subheaders

# Display the logo at the top left of the main page with a specified width and title color
logo_image_html = f"""
<div style="display: flex; align-items: center;">
    <img src="data:image/jpg;base64,{logo_image_base64}" width="100"/>
    <h1 style="margin-left: 10px; color: {title_color}; font-size: {title_font_size};">
        Emotion Detection Platform
    </h1>
</div>
"""
st.markdown(logo_image_html, unsafe_allow_html=True)

# Define the page navigation function
def navigate_to(page):
    st.session_state.page = page

# Function to add a Home button to each page
def add_home_button():
    st.markdown("---")  # Adds a horizontal line for separation
    if st.button("🏠 Home"):
        navigate_to("Home")

# Define the Home page
def home_page():
    st.title("Home")
    st.write("Welcome to the Emotion Detection Platform! "
             "Please select an option below to get started.")
    col1, col2 = st.columns(2)

    # Styles for text and button in the first column
    col1_css = """
        button {
            background-color: #5495D6;
            color: white;
            border-radius: 20px;
            padding: 10px;
        }
        p {
            color: #c0fdfb;
            font-size: 16px;
            margin-bottom: 10px;
        }
    """

    # Styles for text and button in the second column
    col2_css = """
        button {
            background-color: #5495D6;
            color: white;
            border-radius: 20px;
            padding: 10px;
        }
        p {
            color: #c0fdfb;
            font-size: 16px;
            margin-bottom: 10px;
        }
    """

    with col1:
        with stylable_container(key="edp_button", css_styles=col1_css):
            st.markdown("<p></p>", unsafe_allow_html=True)
            if st.button("Go to Emotion Data Preprocessing"):
                navigate_to("EDP")

    with col2:
        with stylable_container(key="model_training_button", css_styles=col2_css):
            st.markdown("<p></p>", unsafe_allow_html=True)
            if st.button("Go to Model Training"):
                navigate_to("Model Training")

# Define the Model Training page
def model_training_page():
    st.markdown(f"<h2 style='font-size: {header_font_size};'>Model Training</h2>", unsafe_allow_html=True)
    st.write("This is the Model Training page.")
    add_home_button()

# Define the EDP (Emotion Detection Predictor) page
def edp_page():
    st.markdown(f"<h2 style='font-size: {header_font_size};'>Emotion Detection Predictor (EDP)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='font-size: {subheader_font_size};'>Upload your audio or video file</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mov"])
        
        if "ws" not in st.session_state:
            st.session_state.ws = get_azure_workspace()
        
        model_names = list_models(st.session_state.ws)
        selected_model = st.selectbox("Choose a model", model_names)
        
        if uploaded_file and selected_model:
            st.write("File uploaded successfully.")
            # Preprocess and predict using Azure model
            result = process_and_predict(uploaded_file, selected_model)
            st.write(result)
    
    with col2:
        st.markdown(f"<h3 style='font-size: {subheader_font_size};'>Prediction Output</h3>", unsafe_allow_html=True)
        # This space will show the prediction results
    
    add_home_button()

def process_and_predict(uploaded_file, selected_model):
    try:
        # Authenticate and connect to the workspace
        ws = st.session_state.ws
        
        # Load the model
        model = get_model(ws, selected_model)
        
        # Preprocess the file (dummy preprocessing step)
        # Add actual preprocessing logic here
        preprocessed_data = uploaded_file.read()  # Placeholder for actual preprocessing
        
        # Predict using the model
        # Replace with actual prediction logic
        # Example: using a web service or local model
        prediction = "dummy prediction result"
        
        return prediction
    
    except Exception as e:
        return f"An error occurred: {e}"

# Define the Prediction Result page
def prediction_result_page():
    st.markdown(f"<h2 style='font-size: {header_font_size};'>Prediction Result</h2>", unsafe_allow_html=True)
    st.write("This is the Prediction Result page.")
    add_home_button()

# Main function to handle the page rendering
def main():
    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Page rendering logic
    if st.session_state.page == "Home":
        home_page()
    elif st.session_state.page == "Model Training":
        model_training_page()
    elif st.session_state.page == "EDP":
        edp_page()
    elif st.session_state.page == "Prediction Result":
        prediction_result_page()

# Run the app
if __name__ == "__main__":
    main()
