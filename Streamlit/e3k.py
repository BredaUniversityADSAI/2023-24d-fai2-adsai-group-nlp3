import streamlit as st
from azure_utils import get_azure_workspace, get_model, list_models
from preprocessing import preprocess_prediction_data, preprocess_training_data


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
    st.write("Welcome to the Emotion Detection Platform!")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Go to Emotion Data Preprocessing"):
            navigate_to("EDP")
    
    with col2:
        if st.button("Go to Model Training"):
            navigate_to("Model Training")

# Define the Model Training page
def model_training_page():
    st.title("Model Training")
    st.write("This is the Model Training page.")
    add_home_button()

# Define the EDP (Emotion Detection Predictor) page
def edp_page():
    st.title("Emotion Detection Predictor (EDP)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Upload your audio or video file")
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
        st.header("Prediction Output")
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
    st.title("Prediction Result")
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
