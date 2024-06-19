import logging
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import shutil
import os

from cli import episode_preprocessing, predict, model_output_information

app = FastAPI()

# Logger setup
logger = logging.getLogger("main")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("fastapi_logs.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Pydantic model for the response
class PredictionResponse(BaseModel):
    predicted_emotions: list
    confidence_scores: list

@app.post("/predict", response_model=PredictionResponse)
async def predict_emotions(
    audio_file: UploadFile = File(...),
    target_sr: int = Form(32000),
    segment_length: float = Form(0.03),
    min_fragment_len: int = Form(300),
    vad_aggressiveness: int = Form(0),
    use_fp16: bool = Form(True),
    transcript_model_size: str = Form("large"),
    model_path: str = Form(...),
    tokenizer_model: str = Form("roberta-base"),
    max_length: int = Form(128),
    decoder_path: str = Form(...)
):
    # Save uploaded file to a temporary location
    temp_file_path = f"temp_{audio_file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    # Setting up arguments
    class Args:
        input_path = temp_file_path
        target_sr = target_sr
        segment_length = segment_length
        min_fragment_len = min_fragment_len
        vad_aggressiveness = vad_aggressiveness
        use_fp16 = use_fp16
        transcript_model_size = transcript_model_size
        model_path = model_path
        tokenizer_model = tokenizer_model
        max_length = max_length
        decoder_path = decoder_path

    args = Args()
    
    # Preprocess the episode
    data_df = episode_preprocessing(args)

    # Make predictions
    predicted_emotions, highest_probabilities = predict(args, data_df)
    
    # Generate model output information
    pie_chart_path = "emotion_distribution_pie_chart.png"
    model_output_information(predicted_emotions, highest_probabilities)

    # Clean up temporary file
    os.remove(temp_file_path)

    return JSONResponse(content={
        "predicted_emotions": predicted_emotions,
        "confidence_scores": highest_probabilities
    })

@app.get("/pie-chart")
def get_pie_chart():
    pie_chart_path = "emotion_distribution_pie_chart.png"
    return FileResponse(pie_chart_path, media_type="image/png", filename="pie_chart.png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
