from transformers import WhisperProcessor, WhisperForConditionalGeneration

def nutn_kws():
    MODEL = "NUTN-KWS/Whisper-Taiwanese-model-v0.5"
    processor = WhisperProcessor.from_pretrained(MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL)
    model.config.use_cache = False
    return processor, model
