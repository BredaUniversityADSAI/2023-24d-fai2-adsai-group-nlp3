import streamlit as st
import base64
import requests

# Inject custom CSS for background color and button styles
st.markdown(
    """
<style>
    .stApp {
        background-color: lightblue;
        padding: 20px;
    }
    .stButton>button {
        background-color: #5495D6;
        color: white;
        border-radius: 20px;
        padding: 10px;
    }
</style>
    """,
    unsafe_allow_html=True
)

# Load the logo image (JPG)
logo_image_path = "images/Screenshot_2024-06-20_at_17.49.24-removebg-preview.png"

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

def home_page():
    st.title("Home")
    welcome_text = """
        <div style="color: black;">
            Welcome to the Emotion Detection Platform! 
            This platform is designed to provide valuable insights into viewer engagement and 
            preferences for TV series by predicting the distribution of emotions in the shows, 
            enabling you to make data-driven decisions. Begin by training your model using text 
            input data and choosing your own hyperparameters. Once your model is ready, you can 
            apply it to analyze audio and video files and see predictions at the sentence level.
        </div>
    """
    st.markdown(welcome_text, unsafe_allow_html=True)
    col1, _ = st.columns(2)

    with col1:
        if st.button("Go to Model Training"):
            st.session_state.training_successful = False  # Reset success state when navigating to training
            navigate_to("Model Training")

# Define the Model Training page
def model_training_page():
    st.markdown(f"<h2 style='font-size: {header_font_size};'>Model Training</h2>", unsafe_allow_html=True)
    st.write("This is the Model Training page.")
    
    val_size = st.number_input("Validation Size", min_value=0.0, max_value=1.0, value=0.2)
    epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
    lr = st.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.001, format="%.4f")
    early_stopping_patience = st.number_input("Early Stopping Patience", min_value=1, max_value=100, value=10)
    model_name = st.text_input("Model Name", value="RoBERTa_model")
    uploaded_file = st.file_uploader("Choose training data file", type=["csv", "json"])  # Assume CSV or JSON for training data
    
    if "training_successful" not in st.session_state:
        st.session_state.training_successful = False  # Initialize the training_successful state
    
    if st.button("Start Training"):
        result = train_model(val_size, epochs, lr, early_stopping_patience, model_name, uploaded_file)
        if isinstance(result, dict) and result.get("message") == "Model trained successfully":
            st.session_state.training_successful = True  # Set flag to True if training was successful
            st.session_state.model_name = model_name  # Store model name in session state
            st.success("Model training completed successfully!")
        else:
            st.error(f"Failed to train model: {result}")

    if st.session_state.training_successful:
        if st.button("Go to Emotion Detection Predictor"):
            navigate_to("EDP")

    add_home_button()

def train_model(val_size, epochs, lr, early_stopping_patience, model_name, uploaded_file):
    try:
        # Specify the FastAPI server URL for the training endpoint
        fastapi_url = f"http://194.171.191.226:3999/train/{val_size}/{epochs}/{lr}/{early_stopping_patience}"
        
        files = {'train_data': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        params = {'model_name': model_name}
        
        response = requests.post(fastapi_url, files=files, params=params)
        
        if response.status_code == 200:
            training_result = response.json()
            return training_result
        
        return f"Failed to start training. Status code: {response.status_code}, detail: {response.text}"
    
    except Exception as e:
        return f"An error occurred: {e}"

# Define the EDP (Emotion Detection Predictor) page
def edp_page():
    st.markdown(f"<h2 style='font-size: {header_font_size};'>Emotion Detection Predictor (EDP)</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"<h3 style='font-size: {subheader_font_size};'>Upload your audio or video file</h3>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "mov"])
        
        # Input fields for query parameters
        target_sr = st.number_input("Target Sample Rate", value=32000)
        segment_length = st.number_input("Segment Length", value=0.03)
        min_fragment_len = st.number_input("Minimum Fragment Length", value=300)
        vad_aggressiveness = st.number_input("VAD Aggressiveness", value=0)
        use_fp16 = st.checkbox("Use FP16", value=True)
        transcript_model_size = st.selectbox("Transcript Model Size", options=["tiny", "base", "small", "medium", "large"], index=0)
        
        # Use the model name from session state
        selected_model = st.session_state.get("model_name", "RoBERTa_model")
        
        if uploaded_file and st.button("Predict"):
            st.write("File uploaded successfully.")
            # Preprocess and predict using FastAPI
            result = process_and_predict(uploaded_file, selected_model, target_sr, segment_length, min_fragment_len, vad_aggressiveness, use_fp16, transcript_model_size)
            st.write(result)
    
    with col2:
        st.markdown(f"<h3 style='font-size: {subheader_font_size};'>Prediction Output</h3>", unsafe_allow_html=True)
        # This space will show the prediction results
    
    add_home_button()

def process_and_predict(uploaded_file, selected_model, target_sr, segment_length, min_fragment_len, vad_aggressiveness, use_fp16, transcript_model_size):
    try:
        # Specify the FastAPI server URL for the predict endpoint
        fastapi_url = "http://194.171.191.226:3999/predict"
        
        # Upload the file to FastAPI
        files = {'audio_file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        # Prepare query parameters
        params = {
            'model_name': selected_model,
            'target_sr': target_sr,
            'segment_length': segment_length,
            'min_fragment_len': min_fragment_len,
            'vad_aggressiveness': vad_aggressiveness,
            'use_fp16': use_fp16,
            'transcript_model_size': transcript_model_size
        }
        
        # Send the POST request to FastAPI
        response = requests.post(fastapi_url, files=files, params=params)
        
        if response.status_code == 200:
            prediction = response.json()
            return prediction
        
        return f"Failed to get prediction. Status code: {response.status_code}, detail: {response.text}"
    
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
