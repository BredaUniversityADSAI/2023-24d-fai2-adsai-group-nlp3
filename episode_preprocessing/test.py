import datetime
import io
import wave

from tqdm import tqdm

import numpy as np
import pandas as pd
import librosa
import webrtcvad
import whisper
from pydub import AudioSegment
import soundfile as sf


audio = AudioSegment.from_file("ep_1.mov")

wav_file = io.BytesIO()
audio.export(wav_file, format="wav")

audio, sample_rate = librosa.load(wav_file, sr=32_000)
print(f"got audio with sample rate: {sample_rate}")

audio = librosa.to_mono(audio)
print("converted to mono")


segment_len = int(0.03 * sample_rate) # 30ms
num_segments = len(audio) // segment_len
print(f"segment length: {segment_len} frames, number of segments: {num_segments}")

segments = [audio[i * segment_len:(i + 1) * segment_len] for i in range(num_segments)]
print(f"split into {len(segments)} segments")

# instantiate VAD with aggressiveness 0/3
vad = webrtcvad.Vad(0)

segment_file = io.BytesIO()

results = []
for segment in tqdm(segments):
    sf.write("test.wav", segment, sample_rate, format="wav", subtype='PCM_16')
    segment_bytes_file = wave.open("test.wav", mode="rb")
    segment_bytes = segment_bytes_file.readframes(segment_len)
    result = vad.is_speech(segment_bytes, sample_rate)
    results.append(result)

results = np.array(results)
print(results)
np.save("segments.npy", results)