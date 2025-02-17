import whisper

class WhisperModel:
    def __init__(self, model_name='turbo'):
        self.model = whisper.load_model(name=model_name, in_memory=True)

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        return result["text"]