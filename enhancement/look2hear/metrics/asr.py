from faster_whisper import WhisperModel

class ASR:
    def __init__(self) -> None:
        model_size = "medium.en"
        self.model = WhisperModel(model_size, device="cuda", download_root="/home/likai/data5/huggingface_models")
        
    
    def __call__(self, audio):
        segments, _ = self.model.transcribe(audio.cpu().numpy(), language="en", vad_filter=True)
        text = " ".join([seg.text for seg in segments])
        return text