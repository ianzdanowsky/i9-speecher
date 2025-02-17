import time
import numpy as np
import uvicorn
import io
import librosa
import zhconv
from fastapi import FastAPI, File, Form, UploadFile
from whisper_model import WhisperModel

app = FastAPI()

whisper = WhisperModel(model_name="medium")

@app.post('/audioToText')
def audio_to_text(
    timestamp: str = Form(),
    audio: UploadFile = File()
):

    audio.filename = f"{timestamp}.wav"

    bt = audio.file.read()

    memory_file = io.BytesIO(bt)

    data, sample_rate = librosa.load(memory_file)

    resample_data = librosa.resample(data, orig_sr=sample_rate, target_sr=16000)

    transcribe_start_time = time.time()
    text = whisper.transcribe(resample_data)
    transcribe_end_time = time.time()

    response = {
        'status': 'ok',
        'text': text,
        'transcribe_time': transcribe_end_time - transcribe_start_time,
    }

    print(response)

    return response

uvicorn.run(app, host="0.0.0.0", port=9090)