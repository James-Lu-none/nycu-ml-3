from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

def openai_whisper_small():
    # 0.2B parameters
    model_name = "openai/whisper-small"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    return processor, model

def openai_whisper_medium():
    # 0.8B parameters
    model_name = "openai/whisper-medium"

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
    model.config.use_cache = False
    return processor, model
